---
title: "All-to-All with Unknown Receive Counts: Four Ways DeepEP Does Dispatch"
description: "MoE routing is only settled after the gate scores tokens, so the receiver does not know how much data is coming. This post compares four DeepEP dispatch designs using one set of questions and turns the differences into NIC bytes and communication time."
date: 2026-11-05
draft: false
lang: en
translationKey: deepep-dispatch-move-count
category: technical
---

<link rel="stylesheet" href="/blog/fused-indextopk/article.css" />

[Chinese version](/blog/deepep-dispatch-move-count-zh/)

Expert parallelism (EP) for MoE does an all-to-all in every layer, sending each token to the GPUs that hold the experts it picked. This step is called dispatch. It differs from a typical all-to-all in one way: the receiver doesn't know in advance how much data it will get. `MPI_Alltoallv` requires the receiver to know each source's count when it makes the call, but MoE routing is only settled after the gate has scored the tokens, and it can change with every layer and every step.

DeepEP handles this in four ways: the normal and low-latency modes in V1, and direct and hybrid in V2. This post first describes the conditions they work under, then goes through the four designs one by one, and finally uses official data to turn the differences into communication time.

> This post is based on the DeepEP source code, using the last V1 commit `29d31c0` and the current V2 commit `f6aa38f`, and covers dispatch only. Examples use hidden size 7168, 8 experts per token and 8 GPUs per node. This is the DeepSeek-V3 configuration on H800. The official DeepEP V1 latency table and the ISCA 2025 paper both use these numbers, which makes calibration possible.

## TL;DR

- **One problem, four designs.** V1 normal mode exchanges counts first and sends data afterwards. V1 low-latency mode reserves buffers for the worst case, and the count, sent last, also signals that the data has arrived. V2 reserves buffers too, but uses a barrier to detect arrival and a copy epilogue to place the data, so counts and data can travel at the same time. Hybrid adds per-node deduplication and two-hop forwarding on top of this.
- **The designs differ because of load and network.** Prefill has large batches and cares mostly about bandwidth. Decode has small batches, needs CUDA graph and cannot wait for the CPU. Meanwhile, NVLink inside a node is much faster than IB between nodes.
- **When decode crosses nodes, most of the time goes to the NIC.** A simple byte model fits the official latency table for V1 low-latency mode, and at EP128 the bandwidth term is about 80% of the total. As a rough estimate, hybrid cuts this term by somewhere between 17% and a half, depending on routing.
- **V2 no longer supports zero-SM EP.** Judging from the mechanism, hybrid forwarding and NVLink transfers both need SMs, so those SMs can't go back to computation while data is still on its way.

## 1. What the receiver needs to know

The receiver's communication buffer is just a piece of memory. Other GPUs write into it directly, so the receiver gets no notification and doesn't know how many entries are coming. Below, the data for one token going to one destination is called a record.

To use this data, the receiver must know three things: whether everything has arrived, or it may read slots that have not been written yet; where each record belongs, because GEMM needs the tokens of an expert to sit next to each other; and how large the output should be, including how many rows each expert has.

Three kinds of numbers are involved, and they are easy to confuse:

| Number | Meaning | Sent across GPUs? |
|---|---|---|
| Slot index | Which slot of the remote buffer this record goes into | No. When a buffer segment has a single writer, the sender can count locally |
| Count | The total number of records going to a GPU or an expert | Yes. The receiver relies on it to answer the questions above |
| Progress pointer | Head or tail of a ring queue or pipeline | Yes. Used for flow control or to notify the next stage |

## 2. Load and network

DeepSeek's production system deploys prefill and decode separately and gives them different EP sizes ([ISCA 2025 paper](https://arxiv.org/abs/2505.09343), §2.3.1). Prefill has large batches, so bandwidth is the bottleneck and fixed overhead gets amortized. An [LMSYS post](https://lmsys.org/blog/2025-05-05-large-scale-ep/) also notes that CUDA graph is typically off in this phase. Decode has small batches, so kernel launches and waiting take up a large share of the time. It therefore uses CUDA graph, and capturing a graph requires fixed shapes and no waiting on the CPU.

On the network side, GPUs in the same NVLink domain can write to each other directly, while traffic between nodes has to go through the NIC. The paper's H800 numbers put NVLink at about 4 times the bandwidth of IB (§4.3).

The four designs map onto these two conditions as follows:

| | One interconnect domain | Two-level network |
|---|---|---|
| **Large batch** | V1 normal (intranode), V2 direct | V1 normal (internode), V2 hybrid |
| **Small batch** | V1 low-latency, V2 direct | V1 low-latency, V2 hybrid |

V1 first splits into two sets of kernels by load. V2 splits only by network into direct and hybrid, and leaves the differences in load to parameters.

## 3. V1 normal: exchange counts first

Since the receiver doesn't know how much is coming, the most direct fix is to tell it first. Normal mode therefore has three steps.

The first step is counting. A kernel started by `get_dispatch_layout` scans the top-k and computes the token counts per GPU, per node and per expert, along with a token-to-GPU table. Then [`notify_dispatch`](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode.cu#L93) exchanges these counts, writes the totals into pinned memory the CPU can read, and computes prefix-sum matrices in GPU memory.

The second step is allocation. The CPU [polls the pinned memory](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/deep_ep.cpp#L1142-L1162) until the counts arrive, then allocates an output of the exact size.

Only the third step sends data. The data kernel is organized into channels. A channel takes two SMs and handles a contiguous range of tokens. Each channel starts by sending its start and end positions to the receiver. The receiver adds the source GPU's prefix sum to the channel start to get the first row, then increments it for each record and [writes the data straight into the output](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode.cu#L1129-L1143). The output is ordered by source GPU and channel, and grouping by expert is left to the caller.

The data takes two hops. Across nodes, each node gets one copy, sent to the GPU with the same index on the target node, which then forwards it over NVLink to each target GPU. The buffers in between are ring queues. When a queue is full, the sender waits for the head to advance, so buffer size doesn't depend on the total token count. The V1 README also notes that this design saves memory but [adds complexity and potential deadlocks](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/README.md#L312).

The price is CPU synchronization: an extra round of count exchange before any data moves, plus the GPU waiting on the CPU. Output shapes also differ every time, which LMSYS describes as “symbolic shapes that are incompatible with CUDA Graph”. The `num_worst_tokens` option can skip the CPU sync, but the docs say it is [for intranode only](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/deep_ep/buffer.py#L354-L355).

## 4. V1 low-latency: reserve for the worst case

Decode cannot wait for the CPU, so it cannot exchange counts in advance either. Low-latency mode instead reserves buffers for the worst case and gives each segment a single writer. The receive buffer has this shape:

```
[local expert, source GPU, Tmax, one 7408 B message]
```

The data from source GPU r to expert e has [its own segment](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L261-L263), sized for Tmax. Only r writes to this segment, so r can assign slots with [a local atomic add](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L256), with no coordination between GPUs.

One kernel has a send phase and a receive phase.

In the send phase, data warps send each token to the segment of each of its top-k experts, using RDMA across nodes and direct NVLink writes within a node. Meanwhile, the last warp of each SM scans the top-k and counts the records going to each expert. Once all requests for an expert have been issued, it [writes the count into the remote count table](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L337-L339). The value written is −n−1, so even n = 0 doesn't write a zero.

Counting can happen alongside the data, but the count must be sent last, because the receiver takes the count to mean the data has arrived. The sender doesn't know whether the data has landed. It relies on channel ordering:

| | Data | Count | Why the count arrives later |
|---|---|---|---|
| Across nodes | RDMA write | RDMA atomic add | Both use the same QP, indexed by the target local expert |
| Within a node | Plain write | Release write | Release makes earlier writes visible first, and the receiver reads with acquire |

In the receive phase, the receiver polls the count table until it reads a non-zero value. It then uses an atomic add to get where this batch starts in that expert's output, and copies and compacts the data. The per-expert counts go to masked GEMM.

This design has one more benefit. With `return_recv_hook` on, the kernel [runs only the send phase](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/deep_ep.cpp#L1649), and the receive phase starts when the hook is called. Over RDMA, the NIC moves the data while it is in flight, so no SM is occupied.

Its drawbacks are the lack of deduplication and the memory use. A token going to two experts on the same GPU is sent twice, so every token is sent 8 times. Each expert also needs a worst-case buffer, which is acceptable for decode with its small Tmax but far too large at prefill scale.

## 5. V2 direct: counts and data travel together

V2 uses one set of kernels for both kinds of load. It has to work without waiting for the CPU, deduplicate per GPU, and still give prefill an exact output size. Its receive buffer is partitioned by source GPU, dropping the expert dimension:

```
[source GPU, Tmax, one record]
```

Only source GPU r writes to its own segment, so slots still come from a local atomic add. Two groups of warps work at the same time inside one kernel.

Notify warps do the counting. They scan the top-k and [accumulate per expert and per GPU](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L100-L105), deduplicating before the per-GPU count. After the SMs' counts are combined, SM0 sends them to each target GPU. SM0 then waits for the counts from other GPUs, writes the totals to host memory if CPU sync is on, and finally computes two prefix sums: the cumulative records from each source GPU, and the first output row of each expert.

Dispatch warps send the data. Each record carries its top-k and source index, so a receiver can tell which experts it belongs to just by reading it. Before sending, the warp deduplicates per GPU and gets a slot with [a local atomic add](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L340-L341). The NVLink path uses TMA to write directly to remote GPU memory. The RDMA path first writes to a registered send buffer, and then the NIC sends it.

When both groups are done, the kernel reaches [an end barrier](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L393) and then triggers the copy epilogue. The copy epilogue uses prefix sums to find each record in the buffer and reads its top-k. For each expert on this GPU, it does an atomic add to [get the output row](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh#L122).

As a result, the count no longer has to signal arrival. The barrier detects arrival, the copy epilogue places the data, and the CPU reads the output size if it needs it. Whether the counts arrive before or after the data makes no difference to the result.

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 312" role="img" aria-label="When counting and moving happen in V1 normal, V1 low-latency and V2" style="width:100%;height:auto;display:block;font-family:'Spectral',Georgia,serif">
<style>
.dd1-title{fill:var(--text);font-size:15px;font-weight:500}
.dd1-row{fill:var(--secondary);font-size:13px;font-style:italic}
.dd1-count{fill:var(--note-bg);stroke:var(--technical);stroke-width:1.3}
.dd1-move{fill:var(--technical)}
.dd1-idle{fill:var(--quiet-line)}
.dd1-ct{fill:var(--text);font-size:13px}
.dd1-mt{fill:var(--background);font-size:13px}
.dd1-it{fill:var(--secondary);font-size:13px;font-style:italic}
.dd1-wall{stroke:var(--reflection);stroke-width:1.5;stroke-dasharray:5 4}
.dd1-wt{fill:var(--reflection);font-size:12.5px;font-style:italic}
</style>
<text class="dd1-title" x="0" y="15">V1 normal: count first, then move</text>
<text class="dd1-row" x="0" y="43">count</text>
<rect class="dd1-count" x="56" y="26" width="104" height="26" rx="4"/>
<text class="dd1-ct" x="108" y="44" text-anchor="middle">layout kernel</text>
<rect class="dd1-count" x="164" y="26" width="104" height="26" rx="4"/>
<text class="dd1-ct" x="216" y="44" text-anchor="middle">notify kernel</text>
<rect class="dd1-count" x="272" y="26" width="128" height="26" rx="4"/>
<text class="dd1-ct" x="336" y="44" text-anchor="middle">CPU waits, allocates</text>
<text class="dd1-row" x="0" y="75">move</text>
<rect class="dd1-move" x="404" y="58" width="316" height="26" rx="4"/>
<text class="dd1-mt" x="562" y="76" text-anchor="middle">data kernel writes each record to its final row</text>
<text class="dd1-title" x="0" y="117">V1 low-latency: move and count together, send the count last</text>
<text class="dd1-row" x="0" y="145">count</text>
<rect class="dd1-count" x="56" y="128" width="228" height="26" rx="4"/>
<text class="dd1-ct" x="170" y="146" text-anchor="middle">last warp scans top-k</text>
<rect class="dd1-count" x="288" y="128" width="92" height="26" rx="4"/>
<text class="dd1-ct" x="334" y="146" text-anchor="middle">send count</text>
<text class="dd1-row" x="0" y="177">move</text>
<rect class="dd1-move" x="56" y="160" width="228" height="26" rx="4"/>
<text class="dd1-mt" x="170" y="178" text-anchor="middle">data warps put records</text>
<rect class="dd1-idle" x="288" y="160" width="176" height="26" rx="4"/>
<text class="dd1-ct" x="376" y="178" text-anchor="middle">in flight: the NIC moves</text>
<rect class="dd1-move" x="468" y="160" width="252" height="26" rx="4"/>
<text class="dd1-mt" x="594" y="178" text-anchor="middle">see the count, then copy and compact</text>
<text class="dd1-title" x="0" y="219">V2: move and count in parallel, a barrier decides arrival</text>
<text class="dd1-row" x="0" y="247">count</text>
<rect class="dd1-count" x="56" y="230" width="344" height="26" rx="4"/>
<text class="dd1-ct" x="228" y="248" text-anchor="middle">notify warps: count, exchange, publish</text>
<text class="dd1-row" x="0" y="279">move</text>
<rect class="dd1-move" x="56" y="262" width="344" height="26" rx="4"/>
<text class="dd1-mt" x="228" y="280" text-anchor="middle">dispatch warps put records</text>
<line class="dd1-wall" x1="412" y1="226" x2="412" y2="292"/>
<text class="dd1-wt" x="412" y="304" text-anchor="middle">end barrier</text>
<rect class="dd1-move" x="424" y="262" width="296" height="26" rx="4"/>
<text class="dd1-mt" x="572" y="280" text-anchor="middle">copy epilogue places rows using prefix sums</text>
</svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">Figure 1: How counting and data transfer line up in time across the three designs</figcaption>
</figure>

## 6. V2 hybrid: deduplicate per node

Direct mode only deduplicates per GPU, which saves little at large EP. Under uniform routing at EP128, a token's 8 experts land on 7.89 different GPUs on average, almost the same as sending 8 copies.

Section 4.3 of the paper offers another route: tokens going to the same node are sent once over IB and then forwarded over NVLink. The IB cost then depends only on the number of target nodes M, falling from 8t to Mt. DeepSeek-V3 was designed around this with node-limited routing. It splits 256 experts into 8 groups, places each group on one node, and guarantees that a token goes to at most 4 nodes. Hybrid follows this path:

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 214" role="img" aria-label="V2 hybrid dispatch: one copy per node over the NIC, then one copy per GPU over NVLink" style="width:100%;height:auto;display:block;font-family:'Spectral',Georgia,serif">
<style>
.dd2-node{fill:var(--note-bg);stroke:var(--note-border);stroke-width:1.2}
.dd2-key{fill:var(--note-bg);stroke:var(--technical);stroke-width:1.6}
.dd2-t{fill:var(--text);font-size:15px;font-weight:500}
.dd2-s{fill:var(--secondary);font-size:13px;font-style:italic}
.dd2-arrow{stroke:var(--secondary);stroke-width:1.3;fill:none}
.dd2-head{fill:var(--secondary)}
.dd2-l{fill:var(--text);font-size:13.5px}
.dd2-n{fill:var(--secondary);font-size:12.5px;font-style:italic}
.dd2-wall{stroke:var(--quiet-line);stroke-width:1.2;stroke-dasharray:5 4}
</style>
<defs><marker id="dd2-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path class="dd2-head" d="M0,1 L10,5 L0,9 z"/></marker></defs>
<text class="dd2-s" x="72" y="16" text-anchor="middle">node A</text>
<text class="dd2-s" x="262" y="16" text-anchor="middle">node B</text>
<line class="dd2-wall" x1="214" y1="6" x2="214" y2="136"/>
<rect class="dd2-node" x="0" y="42" width="144" height="70" rx="7"/>
<text class="dd2-t" x="72" y="71" text-anchor="middle">Source GPU i</text>
<text class="dd2-s" x="72" y="93" text-anchor="middle">send buffer [Tmax]</text>
<line class="dd2-arrow" x1="146" y1="77" x2="268" y2="77" marker-end="url(#dd2-a)"/>
<text class="dd2-l" x="206" y="66" text-anchor="middle">Rail, NIC</text>
<rect class="dd2-key" x="272" y="42" width="158" height="70" rx="7"/>
<text class="dd2-t" x="351" y="71" text-anchor="middle">Relay GPU i</text>
<text class="dd2-s" x="351" y="93" text-anchor="middle">[src node, channel, slot]</text>
<rect class="dd2-node" x="540" y="26" width="180" height="26" rx="5"/>
<text class="dd2-s" x="630" y="44" text-anchor="middle">Target GPU j1</text>
<rect class="dd2-node" x="540" y="64" width="180" height="26" rx="5"/>
<text class="dd2-s" x="630" y="82" text-anchor="middle">Target GPU j2</text>
<rect class="dd2-node" x="540" y="102" width="180" height="26" rx="5"/>
<text class="dd2-s" x="630" y="120" text-anchor="middle">Target GPU j3</text>
<line class="dd2-arrow" x1="432" y1="70" x2="536" y2="39" marker-end="url(#dd2-a)"/>
<line class="dd2-arrow" x1="432" y1="77" x2="536" y2="77" marker-end="url(#dd2-a)"/>
<line class="dd2-arrow" x1="432" y1="84" x2="536" y2="115" marker-end="url(#dd2-a)"/>
<text class="dd2-l" x="484" y="20" text-anchor="middle">Lsa, NVLink</text>
<text class="dd2-l" x="206" y="160" text-anchor="middle">Hop 1: one copy per node</text>
<text class="dd2-n" x="206" y="180" text-anchor="middle">one writer per segment</text>
<text class="dd2-n" x="206" y="198" text-anchor="middle">slot = channel tail</text>
<text class="dd2-l" x="484" y="160" text-anchor="middle">Hop 2: one copy per GPU</text>
<text class="dd2-n" x="484" y="180" text-anchor="middle">several forward warps share a segment</text>
<text class="dd2-n" x="484" y="198" text-anchor="middle">slot = atomic add, then copy epilogue</text>
</svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">Figure 2: The two hops of hybrid</figcaption>
</figure>

The first hop crosses nodes. One warp corresponds to one channel, and tokens are assigned to channels in turn. Each token is [deduplicated per node](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L412), written to the send buffer, and sent through the Rail team to the GPU with the same index on the target node. Data for the local node skips the NIC.

The second hop stays inside the node. A forward warp on the relay GPU starts as soon as it [sees the tail advance](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L517), without waiting for the first hop to finish. It reads the top-k, deduplicates per GPU, and uses TMA to write to the target GPUs over NVLink.

Counts take the same two hops. Per-GPU and per-expert counts go through Rail to the GPU with the same index, which sums them and sends the results over NVLink to the GPUs in its node.

The two hops have different writers, so they assign slots differently:

| | First hop | Second hop |
|---|---|---|
| Target buffer | `[source node, channel, per-channel limit]` | `[relay GPU, nodes × Tmax]` |
| Writers | One warp per segment | Several forward warps on one relay share a segment |
| Slot | The channel's own tail, no atomics | [Atomic add](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L585) |
| Arrival | Tail plus a finish flag | [NVLink barrier](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L658-L662) |

The tail here serves a different purpose from the progress pointers in V1 normal mode. V1 uses them for flow control. Hybrid's buffers are reserved for the worst case and never wrap, so its tail exists to let the second hop start early.

## 7. The four designs side by side

| | V1 normal | V1 low-latency | V2 direct | V2 hybrid |
|---|---|---|---|---|
| A record | Hidden, scales, top-k, source | Source, hidden, scales | Hidden, scales, top-k, source | Same as direct |
| Copies | One per node, then one per GPU | One per expert | One per GPU | One per node, then one per GPU |
| What is counted | Per GPU, node, expert, channel | Per expert | Per GPU, per expert | Per GPU, per expert |
| Who counts | A separate kernel | Last warp of each SM | Notify warps | Notify warps |
| How counts travel | Exchanged before the data | Written to the count table after data is issued | With the data, SM0 sends to each GPU | With the data, summed and forwarded by the same-index GPU |
| Where counts live | Pinned memory, GPU prefix sums | Target GPU's count table | GPU prefix sums, host memory if needed | Same as direct |
| Who reads them | CPU, receiver warps | Receiver warps | Copy epilogue, CPU if needed | Same as direct |
| Arrival | Range known, stop when complete | Seeing the count | End barrier | Tail on hop one, barrier on hop two |
| Buffers | Ring queues, independent of token total | Worst case per expert | Worst case per source GPU | Worst case, on two levels |

Reading across the table, copies and path go together: with one hop you can at most deduplicate per GPU, and per-node deduplication requires a relay. Reading down, whether slots need atomics depends only on how many writers a segment has, and this holds for all four designs.

## 8. From differences to time

A simple model turns these differences into time:

```
communication time ≈ fixed overhead + bytes at the bottleneck port ÷ effective bandwidth
bytes at the bottleneck port = tokens × copies × record size
```

Start by calibrating against the [official table](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/README.md#L32-L37) for V1 low-latency mode. The test used H800 and CX7, with 128 tokens per GPU and 7408 B per message. Traffic within a node goes over NVLink, so only (EP−8)/EP of the bytes pass through the NIC:

| EP | Bytes through the NIC | Model (µs) | Official (µs) |
|---|---|---|---|
| 16 | 3.79 MB | 117 | 118 |
| 32 | 5.69 MB | 159 | 155 |
| 64 | 6.64 MB | 179 | 173 |
| 128 | 7.11 MB | 190 | 192 |
| 256 | 7.35 MB | 195 | 194 |

The fit gives a fixed overhead of about 35 µs and an effective bandwidth of about 46 GB/s. At EP128 the bandwidth term is about 155 µs, 80% of the total, so NIC bytes are the main bottleneck in this configuration. The ratio of combine to dispatch latency is 1.92 (369 ÷ 192), and the ratio of their message sizes is 1.94 (14336 ÷ 7408), which points the same way.

Section 2.3.2 of the paper uses the same arithmetic: dispatch plus combine is (1 B + 2 B) × 32 × 9 × 7 KB ÷ 50 GB/s, about 121 µs. The 9 copies are 8 routed experts without deduplication plus 1 shared expert. Section 4.3 takes 40 GB/s as the effective IB bandwidth, close to the fit above.

The same model gives a rough estimate for hybrid. A V2 record is [7552 B](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/layout/ep/token.cuh#L33-L40). The premise of node-limited routing matters here: the paper assumes expert groups are deployed per node and discusses training, while the global policy that the official [EPLB](https://github.com/deepseek-ai/EPLB) uses for decode ignores expert groups. So for decode at EP128, we can't simply assume a token goes to at most 4 nodes. Below are estimates under two routing assumptions, with 16 nodes at EP128:

| Design and routing | Copies through the NIC | Bytes through the NIC | Bandwidth term |
|---|---|---|---|
| V1 low-latency | 7.5 | 7.11 MB | About 155 µs |
| V2 hybrid, uniform routing | About 6.12 | 5.91 MB | About 129 µs |
| V2 hybrid, exactly 4 nodes per token | About 3.75 | 3.62 MB | About 79 µs |

So how much hybrid saves depends mainly on routing, from about 17% to about a half. Actual results will also depend on the routing distribution, and the extra hop and barrier may add to the fixed overhead.

## 9. What V2 gives up

The V2 README says [buffers are larger than in V1](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L29) and that [0-SM RDMA low-latency EP is no longer supported](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L30), without explaining the latter.

Zero SM requires that only the NIC is working while data is in flight. Judging from the mechanism, two kinds of V2 work can't meet this. One is NVLink transfer, where SMs perform the writes, so nothing keeps writing once the kernel exits. The V1 docs also say that the low-latency NVLink path is [“somehow incompatible”](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/deep_ep/buffer.py#L54-L55) with hooks. The other is hybrid forwarding, which reads the top-k, deduplicates and writes to target GPUs token by token, none of which a NIC can do. Section 4.4.1 of the paper also notes that message handling and NVLink forwarding both occupy SMs, with up to 20 SMs on H800 used for communication during training.

Waiting for counts, the end barrier and the copy epilogue are not fundamental obstacles. They could move to after the hook, as the V1 receive phase does. So a pure RDMA direct path could in principle be split into two phases as well, but V2 doesn't provide one.

In exchange, the official numbers are [up to 1.3x peak performance and up to 4x fewer SMs](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L55), with only 6 SMs for the 8×4 topology.

## Summary

| Design | Handling unknown receive counts | Suited to | Cost |
|---|---|---|---|
| V1 normal | Exchange counts first | Large batches that can wait for the CPU | CPU wait across nodes, no graph capture |
| V1 low-latency | Reserve for the worst case, count signals arrival | Small batches needing capture and overlap | 8 copies per token, large buffers |
| V2 direct | Reserve buffers, counts travel with data | One interconnect domain, both loads | An extra copy epilogue, occupied SMs |
| V2 hybrid | Adds per-node deduplication and two hops | Two-level network, both loads | One more hop, SMs for forwarding |

Beyond these four designs, when DeepInfer deploys V4.1 it [uses Mega MoE if the EP group is inside an NVLink domain](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L184-L188), and DeepEP only across nodes. Mega MoE faces the same unknown receive counts: it exchanges counts inside the kernel, then has the receiver pull the data, as covered in the [previous post](/blog/mega-moe-push-pull-en/).
