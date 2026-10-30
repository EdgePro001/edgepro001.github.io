---
title: "From Push to Pull: A Brief Analysis of the Communication Schemes in DeepGEMM Mega MoE and DeepEP"
description: "DeepEP pushes, Mega MoE pulls. Starting from whether the writer knows the final address, a comparison of the two: five constraints, from scatter to gather, and how the SMs divide the work."
date: 2026-10-30
draft: false
lang: en
translationKey: mega-moe-push-pull
category: technical
---

<link rel="stylesheet" href="/blog/fused-indextopk/article.css" />

[Chinese version](/blog/mega-moe-push-pull-zh/)

Expert parallelism (EP) for MoE does three things in every layer: send each token to the GPUs that hold the experts it picked (dispatch), run two matrix multiplications, and send the results back to the token's own GPU to be summed (combine).

DeepSeek has two implementations of this:

- **DeepEP** handles only the communication. Dispatch and combine are each one kernel; the matrix multiplications in between are left to DeepGEMM.
- **Mega MoE** (inside DeepGEMM) puts dispatch, both matrix multiplications and combine into a single kernel.

While reading the source I noticed one detail: **in DeepEP's dispatch, the sender pushes the data over; in Mega MoE, the receiver pulls it. Yet both of them push in combine.** This post starts from that detail, compares what the two implementations share and where they differ, and asks why they differ.

> This post is based on the DeepGEMM and DeepEP source code and covers the SM100 implementation of Mega MoE. Examples use the DeepSeek V4.1-Flash shape: 256 routed experts, 6 picked per token, expert parallelism across 8 GPUs.

## TL;DR

- **Same job, two approaches.** In DeepEP's dispatch, the sender pushes tokens over and they are reordered on arrival. In Mega MoE, the receiver first exchanges counts, then pulls each token straight to where the matrix multiplication will read it. Both push in combine.
- **Why only dispatch flips: does the writer know the final address at the moment it writes?** In combine, the address traveled with the token, so it pushes. In dispatch, the final position can only be computed once every GPU's counts are in. DeepEP doesn't wait, pushing first and reordering later; Mega MoE waits, then pulls each token straight into place.
- **Pulling turns scatter into gather.** The pull walks the matrix in row order, so low blocks fill first and the GEMM can start while the pull is still going. This only pays off when communication and computation share one kernel, which is exactly why DeepEP doesn't do it.
- **Mega MoE pays a little so data lands in place:** an extra count exchange, about 1.35× the NVLink traffic, and NVLink-domain only. In return, data arrives exactly where the computation reads it.

## 1. What the two have in common

**DeepEP + DeepGEMM** runs several kernels one after another:

1. **Dispatch**: each source GPU **pushes** its tokens into the receive buffer on the destination GPU. The buffer is split by source GPU, one region each. At the same time the GPUs exchange counts of how many tokens each expert receives.
2. **Reorder**: the destination GPU rearranges what it received into a matrix with each expert's tokens stored contiguously, and hands it to the GEMM.
3. **GEMM**: DeepGEMM runs both matrix multiplications in a separate kernel.
4. **Combine**: the GPU holding the expert **pushes** the results back to the source GPU, which sums them locally.

**Mega MoE** uses one kernel:

1. **Exchange counts**: each GPU first tells the destination GPU who is coming and how many.
2. **Pull**: the destination GPU **pulls** tokens from the source GPUs and places them directly where the first GEMM layer will read them.
3. **GEMM**: both matrix multiplications run while the pull is still going on.
4. **Combine**: when the second layer finishes, its results are **pushed** straight back to the source GPU, which sums them locally.

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 300" role="img" aria-label="Dispatch: DeepEP pushes, Mega MoE pulls" style="width:100%;height:auto;display:block;font-family:'Spectral',Georgia,serif">
<style>
.mm1-lane{fill:var(--text);font-size:15.5px;font-style:italic}
.mm1-node{fill:var(--note-bg);stroke:var(--note-border);stroke-width:1.2}
.mm1-key{fill:var(--note-bg);stroke:var(--technical);stroke-width:1.6}
.mm1-t{fill:var(--text);font-size:15px;font-weight:500}
.mm1-s{fill:var(--secondary);font-size:13px;font-style:italic}
.mm1-arrow{stroke:var(--secondary);stroke-width:1.3;fill:none}
.mm1-head{fill:var(--secondary)}
.mm1-lbl{fill:var(--secondary);font-size:13px}
.mm1-wall{stroke:var(--reflection);stroke-width:1.5;stroke-dasharray:5 4}
.mm1-wallt{fill:var(--reflection);font-size:12.5px;font-style:italic}
.mm1-cell{fill:none;stroke:var(--secondary);stroke-width:0.9}
.mm1-fill{fill:var(--technical);opacity:0.85}
.mm1-foot{fill:var(--secondary);font-size:13px;font-style:italic}
</style>
<defs><marker id="mm1-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path class="mm1-head" d="M0,1 L10,5 L0,9 z"/></marker></defs>
<text class="mm1-lane" x="0" y="18">DeepEP + DeepGEMM · the sender pushes</text>
<rect class="mm1-node" x="0" y="32" width="120" height="70" rx="7"/>
<text class="mm1-t" x="60" y="61" text-anchor="middle">Source tokens</text>
<text class="mm1-s" x="60" y="83" text-anchor="middle">in token order</text>
<line class="mm1-arrow" x1="122" y1="67" x2="169" y2="67" marker-end="url(#mm1-a)"/>
<text class="mm1-lbl" x="146" y="59" text-anchor="middle">push</text>
<rect class="mm1-node" x="172" y="32" width="140" height="70" rx="7"/>
<text class="mm1-t" x="242" y="61" text-anchor="middle">Receive buffer</text>
<text class="mm1-s" x="242" y="83" text-anchor="middle">split by source rank</text>
<line class="mm1-arrow" x1="314" y1="67" x2="361" y2="67" marker-end="url(#mm1-a)"/>
<text class="mm1-lbl" x="338" y="59" text-anchor="middle">reorder</text>
<rect class="mm1-node" x="364" y="32" width="158" height="70" rx="7"/>
<text class="mm1-t" x="443" y="52" text-anchor="middle">Expert matrix</text>
<rect class="mm1-fill" x="373" y="68.8" width="30" height="7.2"/><rect class="mm1-cell" x="373" y="60" width="30" height="16"/>
<rect class="mm1-fill" x="409" y="70.4" width="30" height="5.6"/><rect class="mm1-cell" x="409" y="60" width="30" height="16"/>
<rect class="mm1-fill" x="445" y="68" width="30" height="8"/><rect class="mm1-cell" x="445" y="60" width="30" height="16"/>
<rect class="mm1-fill" x="481" y="69.6" width="30" height="6.4"/><rect class="mm1-cell" x="481" y="60" width="30" height="16"/>
<text class="mm1-s" x="443" y="94" text-anchor="middle">all blocks fill slowly</text>
<line class="mm1-wall" x1="556" y1="26" x2="556" y2="110"/>
<text class="mm1-wallt" x="556" y="126" text-anchor="middle">kernel boundary</text>
<line class="mm1-arrow" x1="524" y1="67" x2="587" y2="67" marker-end="url(#mm1-a)"/>
<rect class="mm1-node" x="590" y="32" width="130" height="70" rx="7"/>
<text class="mm1-t" x="655" y="61" text-anchor="middle">DeepGEMM</text>
<text class="mm1-s" x="655" y="83" text-anchor="middle">after dispatch ends</text>
<text class="mm1-lane" x="0" y="170">Mega MoE · the receiver pulls, in one kernel</text>
<rect class="mm1-node" x="0" y="184" width="120" height="70" rx="7"/>
<text class="mm1-t" x="60" y="213" text-anchor="middle">Source tokens</text>
<text class="mm1-s" x="60" y="235" text-anchor="middle">stay in place</text>
<line class="mm1-arrow" x1="122" y1="219" x2="169" y2="219" marker-end="url(#mm1-a)"/>
<rect class="mm1-key" x="172" y="184" width="140" height="70" rx="7"/>
<text class="mm1-t" x="242" y="213" text-anchor="middle">① Exchange counts</text>
<text class="mm1-s" x="242" y="235" text-anchor="middle">who, and how many</text>
<line class="mm1-arrow" x1="314" y1="219" x2="361" y2="219" marker-end="url(#mm1-a)"/>
<text class="mm1-lbl" x="338" y="211" text-anchor="middle">② pull</text>
<rect class="mm1-key" x="364" y="184" width="158" height="70" rx="7"/>
<text class="mm1-t" x="443" y="204" text-anchor="middle">Expert matrix</text>
<rect class="mm1-fill" x="373" y="212" width="30" height="16"/><rect class="mm1-cell" x="373" y="212" width="30" height="16"/>
<rect class="mm1-fill" x="409" y="212" width="30" height="16"/><rect class="mm1-cell" x="409" y="212" width="30" height="16"/>
<rect class="mm1-fill" x="445" y="220" width="30" height="8"/><rect class="mm1-cell" x="445" y="212" width="30" height="16"/>
<rect class="mm1-cell" x="481" y="212" width="30" height="16"/>
<text class="mm1-s" x="443" y="246" text-anchor="middle">low blocks fill first</text>
<line class="mm1-arrow" x1="524" y1="219" x2="587" y2="219" marker-end="url(#mm1-a)"/>
<text class="mm1-lbl" x="556" y="211" text-anchor="middle">block full</text>
<rect class="mm1-key" x="590" y="184" width="130" height="70" rx="7"/>
<text class="mm1-t" x="655" y="213" text-anchor="middle">GEMM</text>
<text class="mm1-s" x="655" y="235" text-anchor="middle">block by block</text>
<text class="mm1-foot" x="0" y="292">Combine is a push on both sides: results go back to the source rank and are summed there.</text>
</svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">Figure 1. Data flow in DeepEP and Mega MoE</figcaption>
</figure>

The two have a lot in common:

- Both run dispatch → two matrix multiplications → combine;
- Both filter by expert before sending: a token goes only to the GPUs that hold its chosen experts, never to all of them;
- Both read and write other GPUs' memory directly through symmetric memory;
- Both exchange counts first and then synchronize across GPUs;
- **Both push in combine.**

One detail differs: the unit of sending. Several of a token's 6 experts may sit on the same GPU. DeepEP deduplicates per GPU, sending one copy to each GPU and expanding it on arrival ([source](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L102-L104)). Mega MoE does not deduplicate: every token–expert pair gets its own row, so two experts on the same GPU pull the same token twice ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L383-L390)). With uniform routing, DeepEP sends 4.44 copies per token on average and Mega MoE sends 6, so Mega MoE uses about 1.35× the NVLink traffic. As we will see, this is one of the prices Mega MoE pays so that data can be computed on as soon as it lands.

## 2. Difference one: push versus pull

### Only dispatch is reversed

DeepEP's combine writes results back to the source GPU using the origin information recorded during dispatch ([source](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/combine.cuh#L95-L107)). Mega MoE does the same ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1336-L1351)).

If pulling were always better, combine should pull too; if pushing were always better, dispatch should not have changed. So the answer must lie in some difference between the two transfers.

### The test: when the write happens, does the writer know the final address?

**In combine**, the destination address is fixed by three numbers: the source GPU, the source token index, and which of the token's choices this is. All three travelled with the token to the expert's GPU during dispatch ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L607)). Writing back only needs a local table lookup and depends on no other GPU. The writer knows the address, so it pushes.

**In dispatch**, the destination address is "row k of expert E's matrix", where k is the number of tokens, across all source GPUs, that also chose E and come before this one. Computing k requires every source GPU's count for E. At the moment of writing, no GPU knows it.

That leaves two options:

- **DeepEP doesn't wait.** Data is pushed right away into a region split by source GPU, a position that can be computed from local information alone. Counts and data travel together, and the data is reordered locally after it arrives.
- **Mega MoE waits.** It finishes exchanging counts first, lets the destination GPU compute k itself, and then pulls each token straight to its final position, with no reorder afterwards.

### How Mega MoE "waits"

Before any data moves, Mega MoE writes two things to the destination GPU ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L368-L428)):

- **A list**: one 4-byte integer per token–expert pair, recording "token i on source GPU r chose expert E".
- **Lengths**: how many tokens each source GPU sends to each expert, plus a signal that every GPU has finished writing.

Both are needed. Without the list, the destination does not know where to read from; without the lengths, it does not know where to stop, or whether the source has finished writing. After that, all GPUs synchronize once more before the pull begins.

This is the price of pulling: before any data moves, there is an extra exchange that must finish first. According to comments in the source, the counting takes about 6.5 µs and writing the list about 2 µs ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L374-L383)).

### Five constraints: why each side is right

Neither design is better in general; each is sensible under its own constraints. I see five:

1. Mega MoE includes the computation, while DeepEP handles only communication;
2. Mega MoE also has a shared expert to compute, and can compute it while waiting for counts;
3. Mega MoE is itself the user of its buffer, so it knows when space frees up and can stop and wait on its own; DeepEP cannot;
4. The side that pulls naturally knows when all the data has arrived;
5. Pulling needs direct access to other GPUs' memory, which only works within one NVLink domain.

Mapped to the source:

| Constraint | DeepEP | Mega MoE |
|---|---|---|
| ① Where the dispatch consumer is | In another kernel; the kernel boundary is a global barrier, so arrival order has no value to it and it should only optimize "the whole transfer ends as early as possible" → don't wait for counts; push first, reorder later | In the same kernel and able to wait per block ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L713-L716)) → arrival order = compute order is what gives value → pull |
| ② The gap while waiting for counts at the start | Nothing in the kernel to compute, so waiting is pure idle time | Issue the shared-expert L1 first (and as much L2 as fits in one wave), since it doesn't depend on dispatch; the comment calls it the "dispatch gap" ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L442-L452)) |
| ③ Can the peer's memory be addressed directly | Must support cross-node RDMA/hybrid, so this cannot be assumed | Only within an NVLink domain ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L552-L575); DeepInfer's enabling condition: [source](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L184-L188)) |
| ④ Can the buffer be bounded | The receiver cannot make the sender stop → allocate for the worst case, `[num GPUs, Tmax, record]` | The one waiting for a free slot is the receiver itself, reading its local `l1_empty_count` ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L544-L550)) → natural backpressure → a ring pool sized by the maximum number of blocks alive at once ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/mega_moe.hpp#L56-L63)) |
| ⑤ Who knows "the data has all arrived" | Push: the receiver only knows through a completion signal from every sender, and the number of signals grows with EP size (MoK cites waiting on up to 71 peers) | Pull: arrival is a local fact for the receiver (its local `l1_full_count` fills up, [source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)). Note Mega MoE is not free of cross-GPU signals: they are concentrated in a single sync at the start (count exchange + NVLink barrier, [source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L393-L428)), after which every arrival check is local |

Two notes:

- **On ②**: the shared expert fills only part of this gap. The warps that post-process GEMM results (the epilogue) must first wait for dispatch to pass the first cross-GPU sync ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L977)), and TMEM, which holds matrix-multiply results, has only 2 slots ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L177)). So each SM can run at most 2 shared-expert tasks during the gap before it has to wait. The timeline in section 4 shows this.
- **On ⑤**: this point comes from Cursor's Mixture-of-Kittens. In a training setting (NVL72, EP64) they independently made the same choice, pull for dispatch and push for combine, for reasons that largely match ①, ④ and ⑤. Two teams reaching the same choice in different settings is more convincing than my reasoning alone.

The table leads to one rule: **the direction of communication is decided by who knows the final address and who needs to control the pace.** When only the receiver can compute the final address and it wants data in the order it will compute, pull; when the sender already knows the final address, push. That is why both sides push in combine.

One more point: pulling and fusion reinforce each other. Fusion is what makes the order of pulling matter; pulling in turn lets the receiver control the pace, so the buffer can be a small ring that is reused, which is what lets one kernel handle a large batch of tokens. In short, Mega MoE pulls data just in time to use it.

## 3. Difference two: from scatter to gather

Constraint ① says pulling is only worth it when arrival order equals compute order. So how does pulling make data arrive in compute order?

### Blocks exist only in the final matrix

The GEMM does not process an expert all at once. It splits the expert's tokens into blocks of BLOCK_M rows and computes them block by block. Whether a block can start depends only on whether all of its BLOCK_M tokens have arrived.

On its way from the source GPU to the GEMM, a token passes through at most three layouts:

| Layout | Where | Row order | Has blocks? |
|---|---|---|---|
| ① Source input `x` | Source GPU HBM | Source token index | No |
| ② DeepEP receive buffer | Destination GPU HBM | `[source GPU][slot]`, slots assigned by the source ([source](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L316-L347)) | No |
| ③ Expert pool = L1's A matrix | Destination GPU HBM | By local expert, each expert padded to a multiple of BLOCK_M | Yes, one block per BLOCK_M rows |

Mega MoE goes ① → ③; DeepEP goes ① → ② → ③.

### A parallel loop finishes in the order of the index it walks

**Mega MoE**: the pull loop's index is the row number of ③ ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L448-L449)). All pulling warps start at row 0 of ③ and move forward, so **blocks with lower row numbers fill first**. The scheduler also hands out GEMM tasks starting from the lowest blocks ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L358)). Pull order and compute order match. Each time a token is pulled, that block's counter goes up by one ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)); when the loading warp sees the counter full, it starts computing that block ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L713-L716)).

**DeepEP**: on the source GPU, the loop index is the source token index ([source](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L276-L279)); on the destination GPU, the reorder loop scans ② row by row and claims a position in ③ with one atomic add per row ([source](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh#L117-L122)). Both steps walk the input in order and scatter into the output, so all blocks of ③ fill a little at a time, together, and almost none is complete until the end.

| | Who runs the loop | Loop index | What finishes first |
|---|---|---|---|
| Pull = gather | Receiver | Output position | The low end of the output |
| Push = scatter | Sender | Input position | The low end of the input is sent first; the output fills sparsely everywhere |

Gather has one requirement: you must be able to work out, from an output position, where the input is. The list and lengths in section 2 are the price paid for that.

For DeepEP, the order in which blocks fill does not matter. Its consumer, DeepGEMM, is another kernel that cannot start until DeepEP has finished entirely, so nobody could use an early block anyway. DeepEP only needs the whole transfer to finish as soon as possible, which is exactly constraint ①.

### A small example

GPU 1 holds experts E0 and E1, and BLOCK_M = 2. Tokens on GPU 0: a picks E0, b picks E1, c picks E0. Tokens on GPU 1: d picks E0, f picks E1.

| Row of ③ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Token | a (GPU 0) | d (GPU 1) | c (GPU 0) | empty | b (GPU 0) | f (GPU 1) |
| Block | E0 block 0 | E0 block 0 | E0 block 1 | E0 block 1 | E1 block 0 | E1 block 0 |

- Mega MoE pulls rows 0, 1, 2, 4, 5 in that order. E0's block 0 fills first and can be computed right away.
- DeepEP's reorder scans ② in the order a, b, c, d, f, so E0's block 0 is not full until it reaches d. And which token comes first inside E0 depends on which atomic add wins.

One more detail: within an expert, tokens from different source GPUs take turns (a, d, c rather than a, c, d). This is Mega MoE's round-robin rule ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L480-L528)). It doesn't decide which block fills first; it decides which GPU each block reads from, so that reads at any moment are spread across all source GPUs instead of piling onto one.

This gives a more general lesson: **to let the consumer start as early as possible, organize the producer's loop in the order the consumer needs.** The same idea is behind output-stationary GEMM, where each thread block owns a fixed tile of the output.

## 4. Difference three: how the SMs divide the work

Once the receiver is the one pulling, the next question is which resources on the GPU it uses to pull.

### Three ways to divide the work

| How work is divided | Example | Where the idle time goes |
|---|---|---|
| By time: one kernel finishes, the next starts | DeepEP dispatch → DeepGEMM → DeepEP combine | Tensor cores idle during communication; NVLink idle during compute |
| By SM: some SMs communicate, others compute | When DeepEP's combine runs alongside the shared expert, DeepEP takes only some of the SMs (DeepInfer caps it at `min(theoretical, total − min(32, total//4))`, [source](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/parallel_utils/userbuffer.py#L40-L51)); Cursor's MoK also sets aside dedicated communication SMs | Tensor cores on the communication SMs stay idle; the split is fixed and needs tuning |
| By warp, inside each SM | Mega MoE: every SM has 4 pull warps, 4 GEMM-related warps (two that load data, one that issues the matrix-multiply instructions, one scheduler) and 8 epilogue warps | Spread across every SM, and the roles that sit idle hold very few resources |

Note that splitting work inside the SM does not mean an SM computes the tokens it pulled itself. When a pull warp finishes a token, it updates a global counter ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)), and the scheduler decides dynamically which SM computes that block ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L344-L370)). Both approaches hand work over through global counters. The real difference is **how resources are split and where the idle time goes**.

### Mega MoE's resource budget

**Registers**: 48 per thread for the pull warps, 40 for the GEMM-related warps, 208 for the epilogue warps ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L328-L331)). That adds up to 128×48 + 128×40 + 256×208 = 64512, exactly filling the register file ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L332-L335)).

**Shared memory**: pulling takes about 17 KB, 1 KB of expert counts plus a 4 KB staging buffer for each of the 4 warps. Each extra GEMM pipeline stage needs about 19.5 KB ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/mega_moe.hpp#L179-L184), for BLOCK_M = 32), so pulling costs roughly one pipeline stage.

In other words, Mega MoE leaves pulling with the bare minimum and gives most of each SM to the matrix multiplications and the epilogue. This works because pulling is done mostly by the TMA hardware and needs almost no computation of its own.

### When each kind of warp is busy

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 272" role="img" aria-label="When each warp role is busy inside the Mega MoE kernel" style="width:100%;height:auto;display:block;font-family:'Spectral',Georgia,serif">
<style>
.mm2-ph{fill:var(--secondary);font-size:13px;font-style:italic}
.mm2-div{stroke:var(--quiet-line);stroke-width:1.2}
.mm2-role{fill:var(--text);font-size:15px;font-weight:500}
.mm2-n{fill:var(--secondary);font-size:12.5px;font-style:italic}
.mm2-busy{fill:var(--technical)}
.mm2-idle{fill:var(--quiet-line)}
.mm2-bt{fill:var(--background);font-size:13.5px}
.mm2-it{fill:var(--secondary);font-size:13.5px;font-style:italic}
.mm2-note{fill:var(--secondary);font-size:12.5px;font-style:italic}
</style>
<!-- track 120..720; phases: start 120-160 | wait 160-265 | pull+compute 265-520 | rest 520-615 | combine 615-720 -->
<text class="mm2-ph" x="140" y="16" text-anchor="middle">start</text>
<text class="mm2-ph" x="212" y="16" text-anchor="middle">wait for counts</text>
<text class="mm2-ph" x="392" y="16" text-anchor="middle">pull ∥ compute</text>
<text class="mm2-ph" x="567" y="16" text-anchor="middle">rest of compute</text>
<text class="mm2-ph" x="667" y="16" text-anchor="middle">combine</text>
<line class="mm2-div" x1="160" y1="26" x2="160" y2="212"/>
<line class="mm2-div" x1="265" y1="26" x2="265" y2="212"/>
<line class="mm2-div" x1="520" y1="26" x2="520" y2="212"/>
<line class="mm2-div" x1="615" y1="26" x2="615" y2="212"/>
<text class="mm2-role" x="0" y="52">Pull warps</text><text class="mm2-n" x="0" y="70">×4</text>
<rect class="mm2-busy" x="121" y="34" width="143" height="32" rx="3"/>
<text class="mm2-bt" x="192" y="55" text-anchor="middle">exchange counts</text>
<rect class="mm2-busy" x="266" y="34" width="168" height="32" rx="3"/>
<text class="mm2-bt" x="350" y="55" text-anchor="middle">pull rows in order</text>
<rect class="mm2-idle" x="436" y="34" width="178" height="32" rx="3"/>
<text class="mm2-it" x="525" y="55" text-anchor="middle">wait for layer-2 writes</text>
<rect class="mm2-busy" x="616" y="34" width="104" height="32" rx="3"/>
<text class="mm2-bt" x="668" y="55" text-anchor="middle">clean up</text>
<text class="mm2-role" x="0" y="98">Scheduler</text><text class="mm2-n" x="0" y="116">×1</text>
<rect class="mm2-busy" x="121" y="80" width="63" height="32" rx="3"/>
<rect class="mm2-idle" x="186" y="80" width="78" height="32" rx="3"/>
<text class="mm2-it" x="225" y="101" text-anchor="middle">wait</text>
<rect class="mm2-busy" x="266" y="80" width="348" height="32" rx="3"/>
<text class="mm2-bt" x="440" y="101" text-anchor="middle">shared expert first, then both layers interleaved</text>
<rect class="mm2-idle" x="616" y="80" width="104" height="32" rx="3"/>
<text class="mm2-role" x="0" y="144">Load + MMA</text><text class="mm2-n" x="0" y="162">×3</text>
<rect class="mm2-busy" x="121" y="126" width="83" height="32" rx="3"/>
<text class="mm2-bt" x="162" y="147" text-anchor="middle">≤ 2 tasks</text>
<rect class="mm2-idle" x="206" y="126" width="58" height="32" rx="3"/>
<rect class="mm2-busy" x="266" y="126" width="348" height="32" rx="3"/>
<text class="mm2-bt" x="440" y="147" text-anchor="middle">two GEMM layers</text>
<rect class="mm2-idle" x="616" y="126" width="104" height="32" rx="3"/>
<text class="mm2-role" x="0" y="190">Epilogue</text><text class="mm2-n" x="0" y="208">×8</text>
<rect class="mm2-busy" x="121" y="172" width="38" height="32" rx="3"/>
<rect class="mm2-idle" x="161" y="172" width="103" height="32" rx="3"/>
<text class="mm2-it" x="212" y="193" text-anchor="middle">blocked</text>
<rect class="mm2-busy" x="266" y="172" width="348" height="32" rx="3"/>
<text class="mm2-bt" x="440" y="193" text-anchor="middle">finish both GEMM layers</text>
<rect class="mm2-busy" x="616" y="172" width="104" height="32" rx="3"/>
<text class="mm2-bt" x="668" y="193" text-anchor="middle">sum results</text>
<rect class="mm2-busy" x="121" y="230" width="14" height="10" rx="2"/>
<text class="mm2-note" x="141" y="239">busy</text>
<rect class="mm2-idle" x="186" y="230" width="14" height="10" rx="2"/>
<text class="mm2-note" x="206" y="239">waiting or idle</text>
<text class="mm2-note" x="720" y="239" text-anchor="end">Schematic, not to scale.</text>
<text class="mm2-note" x="121" y="262">≤ 2 tasks: before the epilogue is released, at most two shared-expert tasks can run.</text>
</svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">Figure 2. When each kind of warp is busy (8192 tokens per GPU, schematic)</figcaption>
</figure>

- **Pull warps**: exchange counts first ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L368-L428)), then pull in row order ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L449-L619)). When done, they stop and wait for all second-layer results to be written ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L623)), then clean up the workspace and do the second cross-GPU sync, overlapping with combine ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L621-L689)).
- **Scheduler warp**: schedules the shared expert's first layer ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L445-L452)), waits for the counts ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L455)), then schedules each expert's first layer, and after that interleaves the two layers.
- **Load and MMA warps**: work whenever there is a task; once scheduling ends, they sit idle through the whole combine phase.
- **Epilogue warps**: blocked at the start ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L977)), then process the results of both layers, and finally turn into combine warps that do the summing ([source](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1365) onward).

Every kind of warp has a stretch of idle time: the pull warps near the end of compute, the GEMM-related warps during combine, the epilogue warps at the start. Mega MoE can accept this because the roles that sit idle hold very few resources, while the most resource-hungry role, the epilogue warps, stays busy from the start of compute almost to the end. The same warp also changes jobs over time: epilogue warps later do combine, and pull warps later do cleanup.

This also explains why Mega MoE doesn't set aside a group of SMs just for pulling: then the idle part would be whole SMs, tensor cores and all their registers included.

## 5. Summary

Looking back, almost every difference between Mega MoE and DeepEP is a facet of the same choice: **pay a little elsewhere so that data lands exactly where the computation needs it.**

- **Pull instead of push**: an extra serial count exchange, in return for data arriving in compute order;
- **No per-GPU deduplication**: about 1.35× the NVLink traffic, in return for computing as soon as data lands, with no second expansion;
- **Splitting work by warp inside each SM**: every SM gives up a little, in return for every tensor core doing work;
- **A reused ring buffer**: the receiver has to control the pace, so it only works within an NVLink domain where peer memory is directly addressable.

DeepEP doesn't do this not because it overlooked something, but because its consumer lives outside the kernel and it has to work across nodes, so blocks arriving early would do it no good. Both designs are sensible under their own constraints.

A few questions remain open for me: why the epilogue warps are blocked at the start (the source comment only says it must not run alongside the barrier before pulling); whether the serial exchange at the start could be made cheaper; and how much each of these differences matters under decode and under prefill. For the last one, I plan to use "how many tokens each expert receives" in a follow-up post.

## References

- DeepGEMM: [PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304) (introduces Mega MoE), [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316) (official benchmarks)
- Cursor, [Mixture-of-Kittens: our open-source MoE megakernel for NVL72s](https://cursor.com/blog/mixture-of-kittens) (2026-08)
- 木子知, [DeepGEMM Mega MOE算子详解](https://zhuanlan.zhihu.com/p/2046275229597672923) (in Chinese)
