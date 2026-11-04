---
title: "接收量未知的 all-to-all：DeepEP dispatch 的四种做法"
description: "MoE 的路由要等 gate 打分后才确定，接收方事先不知道会收到多少。本文用同一组问题比较 DeepEP 的四种 dispatch 做法，并把差别换算成网卡字节和通信时间。"
date: 2026-11-05
draft: false
lang: zh
translationKey: deepep-dispatch-move-count
category: technical
---

<link rel="stylesheet" href="/blog/fused-indextopk/article.css" />

[English version](/blog/deepep-dispatch-move-count-en/)

MoE 的专家并行（EP）每一层都要做一次 all-to-all，把 token 发到它选中的专家所在的卡上，这一步叫 dispatch。这次 all-to-all 和常见的有一点不同：接收方事先不知道自己会收到多少数据。`MPI_Alltoallv` 要求接收方在调用时就知道每个来源发多少，而 MoE 的路由要等 gate 打完分才确定，每层、每一步都可能不一样。

DeepEP 处理这个问题有四种做法：V1 的普通模式和低延迟模式，V2 的 direct 和 hybrid。本文先介绍它们面对的条件，再逐一讲这四种做法，最后用官方数据把差别换算成通信时间。

> 本文基于 DeepEP 的源码，V1 取最后一版 `29d31c0`，V2 取当前的 `f6aa38f`，只讨论 dispatch。举例时取 hidden 7168、每个 token 选 8 个专家、每节点 8 张卡。这是 DeepSeek-V3 在 H800 上的配置，DeepEP V1 的官方延迟表和 ISCA 2025 论文都用这组参数，便于校准。

## TL;DR

- **同一个问题，四种做法。** V1 普通模式先交换计数，再传数据。V1 低延迟模式按最坏情况预留缓冲区，最后发出的计数同时表示数据已经到齐。V2 也预留缓冲区，但用屏障判断到齐，用 copy epilogue 摆放数据，所以计数可以和数据同时传。hybrid 在此基础上按节点去重，分两跳转发。
- **做法的差别来自负载和网络。** prefill 的 batch 大，主要看带宽；decode 的 batch 小，要用 CUDA graph，不能等 CPU。节点内的 NVLink 又比跨节点的 IB 快得多。
- **decode 跨节点时，时间主要花在网卡上。** V1 低延迟模式的官方延迟表可以用一个简单的字节模型拟合，EP128 时带宽部分约占八成。粗略估算，hybrid 能把这部分减少 17% 到一半，具体要看路由。
- **V2 不再支持零 SM。** 从机制上看，hybrid 的中转和 NVLink 传输都要 SM 来做，数据还在路上时，这部分 SM 不能释放给计算。

## 一、接收方需要知道什么

接收方的通信缓冲区只是一块内存。别的卡直接把数据写进来，接收方不会收到通知，也不知道会来多少条。下文把一个 token 发往一个目标的那份数据称为一条记录。

要用这些数据，接收方必须知道三件事：数据收齐了没有，否则可能读到还没写入的格子；每条记录该放在哪，因为 GEMM 要求同一个专家的 token 连续排列；输出要分配多大，每个专家有多少行。

这里会用到三种数字，它们很容易混淆：

| 数字 | 含义 | 要不要跨卡传 |
|---|---|---|
| 槽位号 | 这条记录写进对方缓冲区的第几格 | 不用。一段缓冲区只有一个写入者时，发送方自己计数就够了 |
| 计数 | 发往某张卡或某个专家的总条数 | 要。接收方靠它回答上面的问题 |
| 进度指针 | 环形队列或流水线上的 head、tail | 要。用来做流控，或者通知下一级 |

## 二、负载和网络

DeepSeek 的线上系统把 prefill 和 decode 分开部署，并给它们不同的 EP 规模（[ISCA 2025 论文](https://arxiv.org/abs/2505.09343) §2.3.1）。prefill 的 batch 大，瓶颈在带宽，固定开销可以被摊薄，[LMSYS 的文章](https://lmsys.org/blog/2025-05-05-large-scale-ep/)也提到这个阶段通常不开 CUDA graph。decode 的 batch 小，kernel 的发射和等待占比高，所以要用 CUDA graph，而录制要求形状固定，也不能等 CPU。

网络方面，同一个 NVLink 域内的卡可以直接互写，跨节点则要走网卡。论文给出的 H800 数字是，NVLink 和 IB 的带宽相差约 4 倍（§4.3）。

四种做法和这两个条件的关系如下：

| | 一个互联域 | 两级网络 |
|---|---|---|
| **大 batch** | V1 普通（intranode）、V2 direct | V1 普通（internode）、V2 hybrid |
| **小 batch** | V1 低延迟、V2 direct | V1 低延迟、V2 hybrid |

V1 先按负载分成两套 kernel。V2 只按网络分 direct 和 hybrid，负载的差别交给参数处理。

## 三、V1 普通：先交换计数

既然接收方不知道会来多少，最直接的办法就是先把数量告诉它。普通模式因此分成三步。

第一步是计数。`get_dispatch_layout` 启动的 kernel 扫描 top-k，算出每卡、每节点、每专家的 token 数，以及 token 到卡的映射表。接着，[`notify_dispatch`](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode.cu#L93) 交换这些计数，把总数写进 CPU 可读的锁页内存，并在显存里算好前缀和矩阵。

第二步是分配。CPU [轮询锁页内存](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/deep_ep.cpp#L1142-L1162)，等到计数到齐，再按精确条数分配输出。

第三步才是传数据。数据 kernel 按 channel 组织，一个 channel 占两个 SM，负责一段连续的 token。每个 channel 开始时先把自己的起止位置发给接收方。接收方用来源卡的前缀和加上 channel 的起点，算出起始行，之后每收到一条就加一，把数据[直接写进输出](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode.cu#L1129-L1143)。输出按来源卡和 channel 排列，按专家分组的工作留给调用方。

数据走两跳。跨节点时每个节点只发一份，到了目标节点的同号卡，再经 NVLink 转发给每张目标卡。中间的缓冲区是环形队列，队列满时发送方等待 head 前进，所以缓冲区大小和 token 总数无关。V1 的 README 也提到，这种设计省内存，但[比较复杂，还可能死锁](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/README.md#L312)。

这种做法要付出 CPU 同步的代价：传数据之前多一轮计数交换，GPU 还要等 CPU。输出的形状每次都不同，LMSYS 称之为 “symbolic shapes that are incompatible with CUDA Graph”。`num_worst_tokens` 选项可以跳过 CPU 同步，但文档写明它[只适用于节点内](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/deep_ep/buffer.py#L354-L355)。

## 四、V1 低延迟：按最坏情况预留

decode 不能等 CPU，也就没法事先交换计数。低延迟模式的办法是按最坏情况预留缓冲区，并且让每一段只有一个写入者。接收缓冲区的形状是：

```
[本地专家, 来源卡, Tmax, 一条消息 7408 B]
```

来源卡 r 发给专家 e 的数据有[专属的一段](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L261-L263)，大小按 Tmax 预留。这一段只有 r 会写，所以 r 用[本卡的原子加](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L256)就能分配槽位，不用和别的卡协调。

一个 kernel 分成发送和接收两个阶段。

发送阶段，数据 warp 把每个 token 按 top-k 发往各专家对应的段，跨节点用 RDMA，同节点用 NVLink 直接写。与此同时，每个 SM 的最后一个 warp 扫描 top-k，数出发往各专家的条数。发往某个专家的请求全部发出后，再[把计数写到对方的计数表](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/kernels/internode_ll.cu#L337-L339)。写入的值是 −n−1，这样 n 为 0 时写入的也不是 0。

计数可以和数据同时数，但必须最后发，因为接收方一看到计数，就认为数据已经到齐。发送方并不知道数据有没有到，它依靠的是通道的顺序：

| | 数据 | 计数 | 计数为什么一定后到 |
|---|---|---|---|
| 跨节点 | RDMA 写 | RDMA 原子加 | 两者用同一个 QP，编号都是目标本地专家号 |
| 同节点 | 普通写 | 带 release 的写 | release 保证之前的写先可见，接收方用 acquire 读 |

接收阶段，接收方轮询计数表，读到非 0 的值，再用一次原子加拿到这批数据在该专家输出中的起点，然后把它们拷贝压紧。每个专家的条数交给 masked GEMM。

这个设计还有一个好处。开启 `return_recv_hook` 后，kernel [只运行发送阶段](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/csrc/deep_ep.cpp#L1649)，接收阶段等调用 hook 时才启动。走 RDMA 时，数据在路上由网卡传输，不占用 SM。

它的缺点是不去重，也费内存。一个 token 去同一张卡上的两个专家，也要发两份，所以每个 token 固定发 8 份。每个专家都要按最坏情况预留缓冲区，这在 Tmax 较小的 decode 还能接受，放到 prefill 的规模就太大了。

## 五、V2 direct：计数和数据同时传

V2 用一套 kernel 同时服务两种负载，既要能不等 CPU，又要能按卡去重，还要让 prefill 拿到精确的大小。它的接收缓冲区按来源卡分区，少了专家这一维：

```
[来源卡, Tmax, 一条记录]
```

来源卡 r 的那一段只有 r 自己写，所以槽位仍然来自本卡原子加。一个 kernel 里有两组 warp 同时工作。

notify warp 负责计数。它们扫描 top-k，[分别按专家和按卡累加](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L100-L105)，按卡累加前先去重。各 SM 的计数汇总后，由 SM0 发给各目标卡。SM0 再等别人发来的计数，开启 CPU 同步时把总数写进主机内存，最后算出两组前缀和：每张来源卡的累计条数，以及每个专家在输出里的起始行。

dispatch warp 负责传数据。每条记录都带上 top-k 和来源编号，所以接收方读到记录，就知道它属于哪些专家。发送前先按卡去重，用[本卡原子加](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L340-L341)拿到槽位。NVLink 路径用 TMA 直接写到对方显存；RDMA 路径先写进已注册的发送缓冲区，再由网卡发出。

两组 warp 都结束后，kernel 走到[结束屏障](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L393)，然后触发 copy epilogue。copy epilogue 用前缀和定位缓冲区里的每条记录，读出它的 top-k，再对每个落在本卡的专家做一次原子加，[拿到输出的行号](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh#L122)。

这样一来，计数就不用再表示数据到齐了。判断到齐靠屏障，摆放数据靠 copy epilogue，输出大小由 CPU 按需读取，计数先到还是后到都不影响结果。

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
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 1：三种做法里，计数和传数据的时间关系</figcaption>
</figure>

## 六、V2 hybrid：按节点去重

direct 模式只按卡去重，EP 大时省不了多少。按均匀路由算，EP128 时一个 token 的 8 个专家平均落在 7.89 张不同的卡上，和不去重的 8 份差不多。

论文 §4.3 给出了另一条思路：发往同一节点的 token 只经 IB 发一次，再经 NVLink 转发，这样 IB 上的成本就只取决于目标节点数 M，从 8t 降到 Mt。DeepSeek-V3 为此设计了节点受限路由，把 256 个专家分成 8 组，每组放在一个节点上，并保证一个 token 最多去 4 个节点。hybrid 走的就是这条路径：

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
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 2：hybrid 的两跳</figcaption>
</figure>

第一跳跨节点。一个 warp 对应一个 channel，token 轮流分给各 channel。每个 token [按节点去重](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L412)，写进发送缓冲区，再经 Rail 组发给目标节点的同号卡。本节点的数据不走网卡。

第二跳在节点内。中转卡上的转发 warp [看到 tail 前进](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L517)就开始工作，不用等第一跳全部结束。它读出 top-k，按卡去重，再用 TMA 经 NVLink 写给目标卡。

计数也走这两跳。每卡和每专家的计数先经 Rail 发给同号卡，由它汇总后再经 NVLink 发给本节点各卡。

两跳的写入者不同，分配槽位的方式也不同：

| | 第一跳 | 第二跳 |
|---|---|---|
| 目标缓冲区 | `[来源节点, channel, 每 channel 上限]` | `[中转卡, 节点数 × Tmax]` |
| 写入者 | 一段只有一个 warp 写 | 同一张中转卡上的多个转发 warp 写同一段 |
| 槽位 | channel 自己的 tail，不用原子操作 | [原子加](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L585) |
| 判断到齐 | tail 加结束标志 | [NVLink 屏障](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh#L658-L662) |

这里的 tail 和 V1 普通模式的进度指针用途不同。V1 用它做流控，hybrid 的缓冲区按最坏情况预留，不会绕回，tail 是为了让第二跳尽早开始。

## 七、四种做法的对照

| | V1 普通 | V1 低延迟 | V2 direct | V2 hybrid |
|---|---|---|---|---|
| 一条记录 | hidden、量化系数、top-k、来源 | 来源、hidden、量化系数 | hidden、量化系数、top-k、来源 | 同 direct |
| 发几份 | 每节点一份，再每卡一份 | 每专家一份 | 每卡一份 | 每节点一份，再每卡一份 |
| 数什么 | 每卡、每节点、每专家、每 channel | 每专家 | 每卡、每专家 | 每卡、每专家 |
| 谁来数 | 单独的 kernel | 每 SM 最后一个 warp | notify warp | notify warp |
| 计数怎么传 | 传数据之前单独交换 | 数据发出后写计数表 | 和数据同时，SM0 发给各卡 | 和数据同时，经同号卡汇总转发 |
| 存在哪 | 锁页内存、显存前缀和 | 目标卡计数表 | 显存前缀和，按需写主机内存 | 同 direct |
| 谁来读 | CPU、接收 warp | 接收 warp | copy epilogue，按需 CPU | 同 direct |
| 怎么判断到齐 | 起止已知，收够即止 | 看到计数 | 结束屏障 | 第一跳看 tail，第二跳靠屏障 |
| 缓冲区 | 环形队列，与 token 总数无关 | 每专家按最坏情况预留 | 每来源卡按最坏情况预留 | 同样预留，分两级 |

横着看这张表，发送份数和路径是绑在一起的：只走一跳，最多按卡去重；要按节点去重，就得中转。竖着看则能发现，槽位要不要原子操作，只取决于一段缓冲区有几个写入者，这一点四种做法都一样。

## 八、换算成时间

这些差别可以用一个简单的模型换算成时间：

```
通信时间 ≈ 固定开销 + 瓶颈端口字节 ÷ 有效带宽
瓶颈端口字节 = token 数 × 份数 × 每条记录的大小
```

先用 V1 低延迟模式的[官方表](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/README.md#L32-L37)来校准。测试用的是 H800 和 CX7，每卡 128 个 token，每条消息 7408 B。同节点的流量走 NVLink，所以只有 (EP−8)/EP 的字节经过网卡：

| EP | 经过网卡的字节 | 模型（µs） | 官方数据（µs） |
|---|---|---|---|
| 16 | 3.79 MB | 117 | 118 |
| 32 | 5.69 MB | 159 | 155 |
| 64 | 6.64 MB | 179 | 173 |
| 128 | 7.11 MB | 190 | 192 |
| 256 | 7.35 MB | 195 | 194 |

拟合出的固定开销约为 35 µs，有效带宽约为 46 GB/s。EP128 时带宽部分约 155 µs，占总时间的八成，可见这种配置下的主要瓶颈是网卡字节。combine 和 dispatch 的延迟之比是 1.92（369 ÷ 192），消息大小之比是 1.94（14336 ÷ 7408），也说明了这一点。

论文 §2.3.2 的算法相同：dispatch 加 combine 为 (1 B + 2 B) × 32 × 9 × 7 KB ÷ 50 GB/s，约 121 µs。这里的 9 份是不去重的 8 个路由专家加 1 个共享专家。论文 §4.3 把 IB 的有效带宽取为 40 GB/s，和上面的拟合结果接近。

同样的模型也能粗略估算 hybrid。V2 的一条记录是 [7552 B](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/layout/ep/token.cuh#L33-L40)。这里要注意节点受限路由的前提：论文说的是专家组按节点部署，讨论的场景是训练，而官方 [EPLB](https://github.com/deepseek-ai/EPLB) 用于 decode 的 global 策略并不考虑专家组。所以在 EP128 的 decode 上，不能直接假设一个 token 最多去 4 个节点。下面按两种路由分别估算，EP128 共 16 个节点：

| 方案和路由 | 经过网卡的份数 | 经过网卡的字节 | 带宽部分 |
|---|---|---|---|
| V1 低延迟 | 7.5 | 7.11 MB | 约 155 µs |
| V2 hybrid，均匀路由 | 约 6.12 | 5.91 MB | 约 129 µs |
| V2 hybrid，每个 token 正好去 4 个节点 | 约 3.75 | 3.62 MB | 约 79 µs |

也就是说，hybrid 能省下多少，主要看路由，少则约 17%，多则约一半。实际结果还会受路由分布影响，多出的一跳和屏障也可能增加固定开销。

## 九、V2 的代价

V2 的 README 写明[缓冲区比 V1 更大](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L29)，也[不再支持零 SM 的 RDMA 低延迟 EP](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L30)，但没有解释后者的原因。

零 SM 的前提是数据在路上时只有网卡在工作。从机制上看，V2 有两类工作做不到这一点。一类是 NVLink 传输，它由 SM 执行写操作，kernel 退出后就没有人继续写了。V1 的文档也写过，低延迟模式的 NVLink 路径与 hook [“somehow incompatible”](https://github.com/deepseek-ai/DeepEP/blob/29d31c095796f3c8ece47ee9cdcc167051bbeed9/deep_ep/buffer.py#L54-L55)。另一类是 hybrid 转发，要逐个 token 读 top-k、去重、写目标卡，这些网卡做不了。论文 §4.4.1 也提到，消息处理和 NVLink 转发都要占用 SM，训练时 H800 最多有 20 个 SM 用于通信。

等计数、结束屏障和 copy epilogue 倒不是根本障碍，它们可以挪到 hook 之后再做，V1 的接收阶段就是这样。所以纯 RDMA 的直连路径原则上也能拆成两段，只是 V2 没有提供。

作为交换，官方给出的数字是[峰值带宽最高提升 1.3 倍，SM 占用最多减到四分之一](https://github.com/deepseek-ai/DeepEP/blob/b306af0/README.md#L55)，8×4 的拓扑只用 6 个 SM。

## 总结

| 做法 | 怎么处理接收量未知 | 适用条件 | 代价 |
|---|---|---|---|
| V1 普通 | 先交换计数 | 大 batch，可以等 CPU | 跨节点要等 CPU，无法录制 |
| V1 低延迟 | 按最坏情况预留，计数表示到齐 | 小 batch，要录制和重叠 | 每 token 8 份，缓冲区大 |
| V2 direct | 预留缓冲区，计数和数据同时传 | 一个互联域，两种负载 | 多一趟 copy epilogue，占用 SM |
| V2 hybrid | 在此基础上按节点去重、两跳转发 | 两级网络，两种负载 | 再多一跳，中转占用 SM |

这四种做法之外，DeepInfer 部署 V4.1 时，EP 组[在 NVLink 域内用的是 Mega MoE](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L184-L188)，跨节点才用 DeepEP。Mega MoE 同样要处理接收量未知：它在 kernel 内先交换计数，再由接收方把数据拉过来，详见[上一篇文章](/blog/mega-moe-push-pull-zh/)。
