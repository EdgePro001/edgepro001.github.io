---
title: "“推拉”之间：DeepGEMM Mega MoE 和 DeepEP 通信方案简析"
description: "同一件事，DeepEP 推、Mega MoE 拉。从“写方知不知道最终地址”出发，比较两者的异同：五个约束、从 scatter 到 gather、以及 SM 怎么分工。"
date: 2026-10-30
draft: false
lang: zh
translationKey: mega-moe-push-pull
category: technical
---

<link rel="stylesheet" href="/blog/fused-indextopk/article.css" />

[English version](/blog/mega-moe-push-pull-en/)

MoE 的专家并行（EP）在每一层都做三件事：把 token 发到它选中的专家所在的卡上（dispatch），算两层矩阵乘，再把结果发回 token 原来的卡上求和（combine）。

DeepSeek 为这件事写了两套实现：

- **DeepEP**：只管通信。dispatch 和 combine 各是一个 kernel，中间的矩阵乘交给 DeepGEMM。
- **Mega MoE**（在 DeepGEMM 里）：把 dispatch、两层矩阵乘和 combine 放进同一个 kernel。

读源码时我注意到一个细节：**DeepEP 的 dispatch 是发送方把数据推过去，Mega MoE 是接收方把数据拉过来；但 combine 两边都是推。** 这篇文章从这个细节出发，比较两套实现的异同，并回答它们为什么不同。

> 本文基于 DeepGEMM 和 DeepEP 的源码，讨论 Mega MoE 的 SM100 实现。举例时取 DeepSeek V4.1-Flash 的形状：256 个路由专家，每个 token 选 6 个，8 卡专家并行。

## TL;DR

- **同一件事，两种做法。** DeepEP 的 dispatch 是发送方把 token 推过去，到了再重排；Mega MoE 是接收方先交换计数，再把 token 直接拉到矩阵乘要读的位置。combine 两边都是推。
- **为什么只有 dispatch 反了：看写的那一刻，写方知不知道最终地址。** combine 的地址早已随 token 带到，所以推。dispatch 的最终位置要等所有卡的计数到齐才能算出：DeepEP 选择不等，先推后重排；Mega MoE 选择先等，再直接拉到位。
- **拉的本质是从 scatter 变成 gather。** 拉取按矩阵的行号依次进行，行号小的块先满，矩阵乘可以边拉边算。但这只在通信和计算处于同一个 kernel 时才有价值，这正是 DeepEP 不这样做的原因。
- **Mega MoE 用一点代价，换数据“直接落位”：** 多一段计数交换，多约 1.35 倍的 NVLink 流量，只能在 NVLink 域内使用；换来的是数据一到就落在计算要读的位置。

## 一、相同之处：两者都在做什么

**DeepEP + DeepGEMM** 用几个 kernel 依次完成：

1. **dispatch**：源卡把 token **推**到目标卡的接收缓冲区。缓冲区按源卡分区，每张源卡一块。同时交换“每个专家收到多少 token”的计数。
2. **重排**：目标卡把收到的 token 重新排成按专家连续摆放的矩阵，交给 GEMM。
3. **GEMM**：DeepGEMM 在另一个 kernel 里算两层矩阵乘。
4. **combine**：专家所在的卡把结果**推**回源卡，源卡在本地求和。

**Mega MoE** 只用一个 kernel：

1. **交换计数**：每张卡先告诉目标卡“谁要来”和“有几个”。
2. **拉取**：目标卡把 token 从源卡**拉**过来，直接放到第一层 GEMM 要读的位置上。
3. **GEMM**：边拉边算两层矩阵乘。
4. **combine**：第二层算完，直接把结果**推**回源卡，源卡在本地求和。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-2 0 724 300" role="img" aria-label="Dispatch: DeepEP pushes, Mega MoE pulls" style="width:100%;height:auto;display:block;font-family:'Spectral',Georgia,serif">
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
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 1：DeepEP 与 Mega MoE 的数据流</figcaption>
</figure>

两者有很多相同点：

- 流程都是 dispatch → 两层矩阵乘 → combine；
- 都在发送前按专家筛选，只发到被选中的专家所在的卡，不会广播给所有卡；
- 都靠对称内存直接读写别的卡的显存；
- 都要先交换计数，再做跨卡同步；
- **combine 都是推。**

有一处细节不同：发送的粒度。一个 token 选中的 6 个专家里，可能有几个在同一张卡上。DeepEP 按卡去重，同一张卡只发一份，到了再展开（[源码](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L102-L104)）。Mega MoE 不去重，每个“token–专家”对各占一行，同一张卡上的两个专家会把这个 token 拉两次（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L383-L390)）。按均匀路由算，DeepEP 平均每个 token 发 4.44 份，Mega MoE 发 6 份，Mega MoE 多用了约 1.35 倍的 NVLink 流量。后面会看到，这是 Mega MoE 为“拉到位就能算”付出的代价之一。

## 二、不同之一：推与拉

### 只有 dispatch 反了

DeepEP 的 combine 根据 dispatch 时记下的来源信息，把结果写回源卡（[源码](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/combine.cuh#L95-L107)）。Mega MoE 也是这样做的（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1336-L1351)）。

如果拉总是更好，combine 也该改成拉；如果推总是更好，dispatch 就不该改。所以答案一定藏在这两次传输的某个差别里。

### 判据：写的那一刻，写方知不知道最终地址

**combine 的目标地址**由三个数确定：源卡、源 token 编号、它是这个 token 的第几个选择。这三个数在 dispatch 时已经随 token 带到了专家所在的卡上（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L607)）。写回时只需查本地的表，不依赖任何别的卡。写方知道地址，所以推。

**dispatch 的目标地址**是“专家 E 的矩阵里的第 k 行”。k 等于所有源卡里排在它前面、也选了 E 的 token 数。要算出 k，必须先拿到所有源卡对 E 的计数。写入的那一刻，没有任何一张卡知道它。

于是有两种做法：

- **DeepEP：不等。** 数据立刻推到“按源卡分区”的位置，这个位置只用本卡的信息就能算出来。计数和数据同时传，数据到了之后再在本地重排一次。
- **Mega MoE：等。** 先把计数交换完，目标卡自己算出 k，再把 token 直接拉到最终位置，不用再重排。

### Mega MoE 怎么“等”

在数据动之前，Mega MoE 先往目标卡写两样东西（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L368-L428)）：

- **名单**：每个“token–专家”对写一个 4 字节的整数，记录“源卡 r 的第 i 个 token 选了专家 E”。
- **长度**：每张源卡给每个专家送了几个，外加一个“所有卡都写完了”的完成信号。

两样缺一不可。没有名单，目标卡不知道去哪里读；没有长度，目标卡不知道读到哪里为止，也不知道源卡是否写完。之后所有卡再做一次跨卡同步，才开始拉。

这是拉要付的代价：数据还没开始动，就先多了一段必须串行完成的交换。按源码注释，其中计数约 6.5 us，写名单约 2 us（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L374-L383)）。

### 五个约束：为什么两边各自都对

两种设计都不是绝对更好，而是在各自的约束下都合理。我归纳出五点：

1. Mega MoE 集成了计算，DeepEP 只负责通信；
2. Mega MoE 还要算 shared expert，可以趁等计数的时候先算；
3. Mega MoE 自己就是缓冲区的使用者，知道缓冲区什么时候空出来，可以自己停下来等，DeepEP 做不到；
4. 拉的一方自然知道数据什么时候到齐；
5. 拉要直接访问别的卡的显存，只能在同一个 NVLink 域里做。

对应到源码：

| 约束 | DeepEP | Mega MoE |
|---|---|---|
| ① dispatch 的消费者在哪 | 另一个 kernel；kernel 边界即全局屏障，到达顺序对它无价值，只该优化“整次搬运最早结束” → 不等计数，先推，后重排 | 同一个 kernel，可按块等待（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L713-L716)）→ 到达顺序 = 计算顺序才有价值 → 拉 |
| ② 开头等计数的空隙 | kernel 里没有计算可填，等就是纯空等 | 先发不依赖 dispatch 的 shared expert L1（及一波能放下的 L2），注释称 “dispatch gap”（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L442-L452)） |
| ③ 能否直接寻址对端显存 | 要支持跨节点 RDMA/hybrid，不能假设 | 只在 NVLink 域内（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L552-L575)；DeepInfer 的启用条件见[源码](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L184-L188)） |
| ④ 缓冲区能否有界 | 接收方无法让发送方停 → 按最坏情况开满 `[卡数, Tmax, 记录]` | 等槽空的是接收方自己，读本地 `l1_empty_count`（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L544-L550)）→ 天然反压 → 环形池，容量按最多同时活着的块数算（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/mega_moe.hpp#L56-L63)） |
| ⑤ 谁知道“数据到齐了” | 推：接收方只能靠每个发送方的完成信号才知道到齐，信号数随 EP 规模增长（MoK 称最多等 71 个 peer） | 拉：到齐是接收方的本地事实（本地 `l1_full_count` 加满，[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)）。注意 Mega MoE 并非零跨卡信号：跨卡同步集中在开头一次（计数交换 + NVLink barrier，[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L393-L428)），之后的到齐判断全部本地化 |

两点补充：

- **关于 ②**：shared expert 只能填满这段空隙的一部分。负责收尾处理（epilogue）的 warp 一开始要等 dispatch 通过第一次跨卡同步（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L977)），而存放矩阵乘结果的 TMEM 只有 2 份（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L177)）。所以每个 SM 在空隙里最多只能先算 2 个 shared expert 任务，之后就得等。第四节的时间线里能看到这一点。
- **关于 ⑤**：这一条来自 Cursor 的 Mixture-of-Kittens。他们在训练场景（NVL72，EP64）里独立做出了同样的选择：dispatch 拉，combine 推，理由和 ①④⑤ 基本一致。两个团队在不同场景下得出同一个选择，这比我一个人的推理更有说服力。

由这张表可以得到一条规律：**通信方向由“谁知道最终地址”和“谁需要控制节奏”决定。** 最终地址只有接收方能算出来、而且接收方要按自己的计算顺序取数时，用拉；发送方已经知道最终地址时，用推。所以两边的 combine 都是推。

还要补一句：拉和融合是互相成就的。融合让“按什么顺序拉”有了意义；拉又让接收方能控制节奏，缓冲区可以做成循环复用的有限大小，一个 kernel 才吃得下大批 token。用四个字概括 Mega MoE，就是“急拉急用”。

## 三、不同之二：从 scatter 到 gather

约束 ① 说，到达顺序等于计算顺序时，拉才有价值。那么，拉是怎么让数据按计算顺序到达的？

### 块只存在于最终的矩阵里

GEMM 不会一次算完一个专家，而是把这个专家的 token 切成若干块，每块 BLOCK_M 行，一块一块地算。一块能不能开始算，只看这一块的 BLOCK_M 个 token 是否都已经到了。

一个 token 从源卡到 GEMM，最多经过三种摆放方式：

| 排布 | 在哪 | 行的顺序 | 有没有块 |
|---|---|---|---|
| ① 源卡输入 `x` | 源卡 HBM | 源 token 号 | 没有 |
| ② DeepEP 接收缓冲区 | 目标卡 HBM | `[源卡][槽]`，槽由源卡分配（[源码](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L316-L347)） | 没有 |
| ③ 专家池 = L1 的 A 矩阵 | 目标卡 HBM | 按本地专家排，专家内补齐到 BLOCK_M | 有，每 BLOCK_M 行一块 |

Mega MoE 走 ① → ③，DeepEP 走 ① → ② → ③。

### 循环按谁的下标走，谁就按那个顺序完成

**Mega MoE**：拉取循环的下标就是 ③ 的行号（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L448-L449)）。所有负责拉取的 warp 从 ③ 的第 0 行开始往后拉，所以**行号小的块先满**。调度器分配 GEMM 任务时，也是从行号小的块开始（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L358)）。拉的顺序和算的顺序一致。每拉完一个 token，这一块的计数器就加 1（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)）；加载数据的 warp 看到计数满了，就开始算这一块（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L713-L716)）。

**DeepEP**：在源卡上，循环的下标是源 token 号（[源码](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch.cuh#L276-L279)）；到了目标卡，重排的循环按 ② 的顺序逐行扫描，每扫一行就用一次原子加在 ③ 里占一个位置（[源码](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh#L117-L122)）。两步都是“按输入的顺序走、往输出里散”，所以 ③ 的各个块是同时、零星地被填上的，几乎都要到最后才满。

| | 谁跑循环 | 循环下标 | 什么先完成 |
|---|---|---|---|
| 拉 = gather | 接收方 | 输出位置 | 输出的低端先完成 |
| 推 = scatter | 发送方 | 输入位置 | 输入的低端先发完，输出到处零星 |

gather 有一个前提：要能从输出的位置反推出输入在哪里。第二节说的名单和长度，就是为这个前提付的代价。

对 DeepEP 来说，块按什么顺序满并不重要。它的下游 DeepGEMM 是另一个 kernel，要等 DeepEP 全部结束才能开始，哪一块先到齐都没人能提前用。所以 DeepEP 只需要让整次搬运尽早结束，这正是约束 ①。

### 一个小例子

卡 1 上有专家 E0 和 E1，BLOCK_M = 2。卡 0 的 token：a 选 E0、b 选 E1、c 选 E0；卡 1 的 token：d 选 E0、f 选 E1。

| ③ 的行 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| token | a（卡 0） | d（卡 1） | c（卡 0） | 空 | b（卡 0） | f（卡 1） |
| 块 | E0 第 0 块 | E0 第 0 块 | E0 第 1 块 | E0 第 1 块 | E1 第 0 块 | E1 第 0 块 |

- Mega MoE 按行 0、1、2、4、5 的顺序拉，E0 的第 0 块最先满，可以马上开始算。
- DeepEP 重排时按 ② 的顺序扫 a、b、c、d、f，E0 的第 0 块要扫到 d 才满。而且 E0 内部谁排前面，取决于原子加谁先抢到。

还有一个细节：同一个专家里，来自不同源卡的 token 是轮流排的（a、d、c，而不是 a、c、d）。这是 Mega MoE 的轮转规则（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L480-L528)）。它决定的不是哪一块先满，而是每一块去哪张卡读：让同一时刻的读请求分散到所有源卡上，不会全挤在一张卡上。

由此可以得到一条更一般的经验：**想让下游尽早开工，就按下游需要的顺序来组织上游的循环。** GEMM 里每个线程块负责一块固定输出（output-stationary），也是同样的道理。

## 四、不同之三：SM 怎么分工

确定了由接收方来拉，下一个问题是：接收方用 GPU 上的哪些资源来拉？

### 三种分工方式

| 分工方式 | 例子 | 闲置放在哪里 |
|---|---|---|
| 按时间分：一个 kernel 做完再做下一个 | DeepEP dispatch → DeepGEMM → DeepEP combine | 通信时 tensor core 全闲，计算时 NVLink 全闲 |
| 按 SM 分：一部分 SM 通信，一部分 SM 计算 | DeepEP 的 combine 与 shared expert 同时跑时，DeepEP 只占一部分 SM（DeepInfer 的上限是 `min(理论值, 总数 − min(32, 总数//4))`，[源码](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/parallel_utils/userbuffer.py#L40-L51)）；Cursor 的 MoK 也划出专门的通信 SM | 通信 SM 上的 tensor core 一直闲着；比例固定，需要调 |
| 在 SM 内按 warp 分 | Mega MoE：每个 SM 有 4 个拉取 warp、4 个 GEMM 相关 warp（两个搬数据、一个发矩阵乘指令、一个调度）、8 个收尾 warp | 分摊到每个 SM，而且会闲着的角色只占很少的资源 |

注意，“在 SM 内分工”并不意味着一个 SM 拉来的 token 就由这个 SM 来算。拉取 warp 拉完一个 token，更新的是全局计数器（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L613)），由哪个 SM 来算这一块，是调度器动态决定的（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L344-L370)）。两种分工方式都要通过全局计数器交接。真正的区别在于**资源怎么切、闲置放在哪里**。

### Mega MoE 的资源账

**寄存器**：拉取 warp 每个线程 48 个，GEMM 相关 warp 40 个，收尾 warp 208 个（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L328-L331)）。加起来 128×48 + 128×40 + 256×208 = 64512，正好用满整个寄存器堆（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L332-L335)）。

**共享内存**：拉取要占约 17 KB，包括专家计数 1 KB 和 4 个 warp 各 4 KB 的中转缓冲。而 GEMM 每多一级流水大约要 19.5 KB（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/mega_moe.hpp#L179-L184)，按 BLOCK_M = 32 算），所以拉取大约占掉了一级流水。

也就是说，Mega MoE 只给拉取留了最少的资源，把每个 SM 的大部分留给矩阵乘和收尾。这样做行得通，是因为拉取主要靠 TMA 硬件完成，本身几乎不需要计算。

### 每类 warp 什么时候在忙

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
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 2：每类 warp 的忙闲时间线（每卡 8192 个 token，示意）</figcaption>
</figure>

- **拉取 warp**：先交换计数（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L368-L428)），再按矩阵行号的顺序拉取（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L449-L619)）。拉完后停下来，等所有第二层的结果写完（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L623)），最后清理工作区并做第二次跨卡同步，这一段和 combine 同时进行（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L621-L689)）。
- **调度 warp**：先安排 shared expert 的第一层（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L445-L452)），等计数到齐（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L455)），再安排各专家的第一层，之后两层交替。
- **搬数据和发矩阵乘指令的 warp**：有任务就做；调度结束后，整个 combine 阶段都闲着。
- **收尾 warp**：开头被挡住（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L977)），之后处理两层矩阵乘的结果，最后变成 combine warp，负责求和（[源码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1365) 起）。

每类 warp 都有一段是闲着的：拉取 warp 在收尾阶段，GEMM 相关的 warp 在 combine 阶段，收尾 warp 在开头。Mega MoE 能接受这些闲置，是因为会闲着的角色只占很少的资源，而最吃资源的收尾 warp 从开始计算到最后几乎一直在忙。同一个 warp 还会随时间换工作：收尾 warp 后来去做 combine，拉取 warp 后来去做清理。

这也回答了 Mega MoE 为什么不单独划一批 SM 来拉取：那样的话，闲着的就是整个 SM，包括它的 tensor core 和全部寄存器。

## 五、小结

回头看，Mega MoE 和 DeepEP 的每一处不同，几乎都是同一个选择的不同侧面：**用一点别的代价，换数据直接落在计算需要的位置。**

- **拉而不推**：多一段串行的计数交换，换来数据按计算顺序到达；
- **不按卡去重**：多用约 1.35 倍的 NVLink 流量，换来拉到位就能算，不必再展开一次；
- **在 SM 内按 warp 分工**：每个 SM 都让出一点资源，换来所有 tensor core 都在算；
- **循环复用的缓冲区**：要求接收方自己控制节奏，所以只能在能直接访问对端显存的 NVLink 域内使用。

DeepEP 不这样做，不是因为考虑不周，而是它的下游在 kernel 之外，它还要支持跨节点，“按块到达”对它没有价值。两种设计都是在各自约束下的合理选择。

还有几个问题我没有弄清楚：收尾 warp 为什么一开始要被挡住（源码注释只说“不能和拉取前的同步同时进行”）；开头那段串行交换能不能做得更便宜；以及上面这些差异在 decode 和 prefill 两种负载下分别有多重要。最后一个问题，我打算下一篇用“每个专家分到多少 token”来分析。

## 参考

- DeepGEMM：[PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304)（引入 Mega MoE）、[PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316)（官方基准）
- Cursor，[Mixture-of-Kittens: our open-source MoE megakernel for NVL72s](https://cursor.com/blog/mixture-of-kittens)（2026-08）
- 木子知，[DeepGEMM Mega MOE算子详解](https://zhuanlan.zhihu.com/p/2046275229597672923)
