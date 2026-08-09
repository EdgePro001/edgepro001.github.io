---
title: "Sparse Attention 的 Algorithm–Infra Co-design"
description: "从 NSA 的 block-level selection 到 DSA 的 token-level selection：逐 shape 推导算法数据流、GPU program ownership、访存与 backward reduction。"
date: 2026-08-10
draft: false
lang: zh
translationKey: sparse-attention-co-design
---

## 1. Opening

全文沿着一个主视角展开：

> **先把算法改写成每一维、每一步依赖都显式可见的 tensor dataflow；再追踪
> 这些逻辑 tensor 如何变成一个 GPU program 能够独占、搬运和计算的 tile。**

沿着这条链条，我们先拆解 NSA 与 DSA 的算法数据流和 kernel mapping，再用两组
H100 实验检验从实现分析中自然产生的两个问题，最终得到五条相互关联的结论。

1. **Sparsity ratio 不是 execution plan。** 它只说明算法逻辑上保留了多少
   query–KV interactions，不说明为了发现、表示与执行这些 interactions 还要付出
   多少工作。selector/Top-K、metadata、间接 KV load、一次 load 能被复用多少次、
   backward 中的 many-to-one reduction，以及这些工作能否重新组织成规则 MMA
   tiles，都会进入真实成本。
2. **NSA 在 load 之前制造规则性。** 一个 selection ID 指向逻辑上连续的
   64-token index 区间；不同 selected blocks 的 base address 仍可离散，但 block
   内的 64 行可以从一个 base 通过规则 affine offsets 展开，而不必为每个 token
   单独保存一个间接索引。同时，同一 KV group 的 16 个 query heads 共享 selection
   IDs。本文审阅的 FLA 社区实现进一步把这两个轴直接映射成
   $[\text{query heads},\text{KV tokens}]=[16,64]$ 的 interaction tile。
3. **Forward 与 Backward 需要不同的稀疏图与 infra 设计。** Query-centric
   mapping 可以为 $O^{\mathrm{slc}}$ 和 $dQ^{\mathrm{slc}}$ 提供唯一
   owner，但 $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 必须对所有选择同一 KV
   block 的 queries 做 many-to-one reduction。本文审阅的 FLA 社区实现将
   forward adjacency `query → selected blocks` 转置为 CSR 形式的
   `KV block → selecting queries`，再让一个 KV-block program 成为对应
   $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 的唯一 writer，从而避免昂贵的
   浮点 atomics 或 partial-gradient buffers。代价是额外的 CSR 构造、离散
   query gather、可变 fan-in 与 CTA load imbalance。
4. **DSA 在 load 之后恢复规则性。** DSA 将最终选择细化到动态
   token level，但没有让 128 个 main heads 分别维护独立的随机地址：
   对于同一个 query，它们共享同一组 Top-2048 IDs。公开的 H100
   sparse-prefill kernel 使用两个 CTA 覆盖这 128 个 heads；每个 CTA 处理
   64 个 heads，将一条 selected KV row 搬入 SMEM 后在 CTA 内复用 64 次。
   因此只有 producer 面对离散的 HBM row bases，Tensor Core consumers 仍可
   读取规则的 $[64,576]$ SMEM tile。在冻结的
   $T_Q=128,\kappa=2048$ H100 microbenchmark 中，对每个 query 保持 selected set
   不变，只将地址有序的 rows 改成八种预先冻结的随机排列，相对
   ascending 的 latency 点估计只相差 $0.10\%$--$0.47\%$，每个
   paired 95% CI 都包含 1。concentration sweep 则不单调；只有 2K/4K
   的 tight windows 出现亚百分比的 pointwise 小信号。这说明该固定
   operating point 下的 kernel 对所测地址模式较稳健，不等于“随机访存免费”。
5. **DSA 没有消灭 $O(L^2)$。**

   **算法上，**对长度为 $L$ 的 causal sequence，Lightning Indexer 仍需计算
   全部 causal query--history pairs，其数量为

   $$
   \frac{L(L+1)}{2}=\Theta(L^2).
   $$

   公开 unfused 路径中的 exact Top-K 还要扫描随上下文增长的 FP32
   logits carrier。固定 $\kappa=2048$ 只把高维 main Attention 的 pair count
   限制为 $\Theta(L\kappa)$，没有把整个 DSA Attention module 变成线性复杂度。

   **实现上，**在冻结的 H100、$B=1,C=4096,\kappa=2048$ 公开组件
   合成链中，causal-prefill cumulative discovery 在 32K、64K 和 2M
   分别为 FlashMLA 的 $0.727\times$、$1.267\times$ 和 $33.60\times$。64K
   是首个预设的 discovery-dominant checkpoint，测得的离散 bracket 为
   $[32\mathrm K,64\mathrm K]$，不作插值。从 256K 到 2M，Discovery 与
   sparse-prefill 的端点有效指数分别为 $2.027$ 与 $1.031$：这个
   operating point 展示了隐藏二次项如何重新成为瓶颈，并不代表 production
   DSA 存在固定的 64K crossover。

因此全文的中心命题可以压缩成两句话：

> **Sparsity becomes executable when the algorithm exposes a reuse axis that one
> GPU program can own. Whether it becomes speed depends on the cost of discovering,
> regularizing, and reducing that sparsity.**

本文不再重讲 naive dense Attention 或 FlashAttention。我们的起点是更靠后的
问题：当算法交给 kernel 的不再是规则矩阵，而是 block IDs 或 token IDs 时，
GPU 到底如何执行？

为避免混淆证据，后文只在必要处使用六类标记：`[PAPER]` 表示论文陈述，
`[CODE]` 表示官方公开代码可核对，`[CODE/community]` 表示社区实现展示了一条
可行路径，`[DERIVATION]` 表示由已声明的 shape、dtype 或执行顺序推出，
`[MEASURED]` 表示本文在冻结环境中的正式测量，`[GAP]` 表示公开资料不足。
NSA 论文使用的原始高性能 kernel 没有完整公开，因此代码级
backward 分析会明确标为社区实现。DSA 的 Indexer scorer 与 FlashMLA
sparse-forward kernel 已经公开，但 production Indexer–Top-K fusion、完整运行时
orchestration 以及 training backward 仍不可见。后文的 Nsight 结果只作为
diagnostic evidence，不会替代正式 CUDA-event latency。

## 2. 统一 Tensor Lens：从逻辑矩阵到可执行 tile

> **Takeaway.** 后文最重要的不是“一个矩阵有多大”，而是区分算法逻辑上存在的
> tensor、真正写入 HBM 的 tensor，以及一个 CTA/program 当前消费的 tile。

除非特别说明，本文省略 batch 轴，并使用 token-major 的概念 shape：

| 符号 | 含义 |
|---|---|
| $T_Q,T_K$ | 本次调用中的 query 与 KV token 数 |
| $T$ | causal self-attention 的完整序列长度 |
| $H_q,H_{kv}$ | query heads 与 KV heads/groups 数 |
| $G=H_q/H_{kv}$ | 一个 KV head/group 服务的 query-head 数 |
| $t,h$ | query-token 位置与 query-head ID |
| $r=(t,h)$ | 一条独立的 Attention softmax row |
| $g(h)=\lfloor h/G\rfloor$ | GQA 中 query head $h$ 对应的 KV head/group |
| $d_k,d_v$ | 每个 head 的 Q/K 与 V/O 维度 |
| $B_R,B_{\mathrm{kv}}$ | 一个 program 独占的 softmax-row 数与 KV-token tile 长度 |
| $\alpha$ | Attention score 的缩放因子 |

一次 FMA 计作 2 FLOPs。除非特别注明，byte count 使用 BF16，即每个元素
2 bytes；解析 byte count 不包含 cache-line overfetch、allocator、page/TLB
与跨设备通信。

**Logical shape.** 固定一个 query head $h\in[0,H_q)$，它在 GQA 中
使用第 $g(h)$ 个 KV head/group：
$$
Q^{(h)}:[T_Q,d_k],\qquad
K^{(g(h))}:[T_K,d_k],\qquad
V^{(g(h))}:[T_K,d_v].
$$

数学语义是

$$
\begin{aligned}
S^{(h)}
&=\alpha Q^{(h)}(K^{(g(h))})^\top,
&
[T_Q,d_k][d_k,T_K]
&\rightarrow[T_Q,T_K],\\
P^{(h)}
&=\operatorname{softmax}_{T_K}(S^{(h)}),
&
[T_Q,T_K]
&\rightarrow[T_Q,T_K],\\
O^{(h)}
&=P^{(h)}V^{(g(h))},
&
[T_Q,T_K][T_K,d_v]
&\rightarrow[T_Q,d_v].
\end{aligned}
$$

因果 mask 或 sparse-selection mask 在这里省略；所有不可见位置的
score 在 softmax 前视为 $-\infty$。

$S/P:[T_Q,T_K]$ 首先只是 **logical shape**：它定义每个输出依赖哪些输入，
不意味着 kernel 必须把完整 score 或 probability tensor 写入 HBM。后文始终
区分三层：

1. **logical shape**：算法语义中的完整关系；
2. **materialized shape**：真正写入 HBM、供另一 kernel 读取的 tensor；
3. **tile shape**：一个 CTA/program 搬到片上并交给 MMA 的局部工作集。

**Tile shape.** Kernel 可以沿 query tokens 分块，也可以让多个 query
heads 共享同一个 KV tile。因此我们不把 program 的 row 轴固定解释为
query-token 轴，而用 $B_R$ 表示它独占的独立 softmax rows 数。若 $i$
表示这个 row tile，$j$ 表示当前 KV tile，则
$$
Q_i:[B_R,d_k],\qquad
K_j:[B_{\mathrm{kv}},d_k],\qquad
V_j:[B_{\mathrm{kv}},d_v],
$$

其中这 $B_R$ 条 rows 必须能够共享当前的 $K_j,V_j$。这正是后文
NSA 与 DSA 各自暴露的复用轴。对应的两个局部 contraction 为

$$
\begin{aligned}
S_{ij}
&=\alpha Q_iK_j^\top,
&[B_R,d_k][d_k,B_{\mathrm{kv}}]
&\rightarrow[B_R,B_{\mathrm{kv}}],\\
\Delta O_i
&=\widetilde P_{ij}V_j,
&[B_R,B_{\mathrm{kv}}][B_{\mathrm{kv}},d_v]
&\rightarrow[B_R,d_v].
\end{aligned}
$$

NSA 与 DSA 的核心差别，在算法层面体现于
$K_j,V_j$ 怎样被选出，而在 infra 层面体现于它们怎样被搬进片上，
以及这次 load 能被谁复用。固定 head 并沿 query tokens 分块时，
$B_R$ 就是 query-token tile 长度；NSA selection forward 中 $B_R=G=16$；
DSA FlashMLA head64 CTA 中 $B_R=B_H=64$。

**Online softmax.** 只要一个 program 独占这 $B_R$ 条 softmax rows，
就可以为它们保留三个状态：
$$
m:[B_R],\qquad
z:[B_R],\qquad
O_{\rm acc}:[B_R,d_v].
$$

初始状态为

$$
m^{(0)}=-\infty,\qquad z^{(0)}=0,\qquad O_{\rm acc}^{(0)}=0.
$$

$m$ 是已访问 logits 的 row-wise maximum，$z$ 是相对 $m$ 的指数和，
$O_{\rm acc}$ 是同一基准下的 value numerator。新 score tile
$S_{ij}:[B_R,B_{\mathrm{kv}}]$ 到达时，先计算

$$
\widehat m_j
=\max_{b=0,\ldots,B_{\mathrm{kv}}-1}S_{ij}[:,b]
\in\mathbb R^{B_R},
$$

$$
m_{\rm new}=\max(m_{\rm old},\widehat m_j)
\in\mathbb R^{B_R},
$$

再将旧状态换到新的 row-wise maximum 基准：

$$
\rho=\exp(m_{\rm old}-m_{\rm new})
\in\mathbb R^{B_R}.
$$

当前 tile 在这一基准下的未归一化权重为

$$
\widetilde P_{ij}
=\exp\!\left(S_{ij}-m_{\rm new}[:,\mathrm{None}]\right)
\in\mathbb R^{B_R\times B_{\mathrm{kv}}}.
$$

然后完成两个状态更新：

$$
z_{\rm new}
=\rho z_{\rm old}
+\sum_{b=0}^{B_{\mathrm{kv}}-1}\widetilde P_{ij}[:,b]
\in\mathbb R^{B_R},
$$

$$
O_{{\rm acc},{\rm new}}
=\rho[:,\mathrm{None}]O_{{\rm acc},{\rm old}}
+\widetilde P_{ij}V_j
\in\mathbb R^{B_R\times d_v}.
$$

遍历所有 KV tiles 后：

$$
O_i
=
\frac{O_{\rm acc}}{z[:,\mathrm{None}]}
\in\mathbb R^{B_R\times d_v},\qquad
\operatorname{LSE}_i=m+\log z
\in\mathbb R^{B_R}.
$$

因此 selected blocks 或 selected rows 可以分批访问，而无须物化完整
$P$。在精确算术下，改变 KV tiles 的遍历顺序不改变数学结果；
在有限精度下，不同 reduction order 可能带来微小数值差异。NSA 与 DSA
改变的是“哪些 KV tile 进入这条流”，不是 softmax 的数学语义。

从这里开始，我们对每条 sparse path 只反复问五件事：

1. **Stage/materialization：** 哪些 tensor 只在逻辑上存在，哪些会跨 kernel
   写入 HBM？
2. **Grid/ownership：** grid 中有多少 programs？一个 program 独占哪些
   output rows，并由谁持有 $m,z,O_{\rm acc}$？
3. **Tile/reuse：** program 消费什么 tile？一次搬入的 KV row/tile 被多少
   query rows 或 heads 消费？
4. **Irregularity boundary：** 动态 index 在哪里出现，又在哪里被重排为
   规则 tile？
5. **Full cost：** selector、Top-K、metadata、gather、load balance 与
   backward reduction 留下了什么成本？

两种算法将给出两种答案：

$$
\boxed{
\begin{aligned}
\text{NSA: }&
\text{在 HBM load 之前，把稀疏关系限制为连续 token blocks；}\\
\text{DSA: }&
\text{保留 token-level 间接寻址，在 load 之后于 SMEM 重建规则 tile。}
\end{aligned}}
$$

## 3. NSA：先让算法产生可执行的 Block Workload

> **Takeaway.** Native Sparse Attention（NSA）让一个 selection ID 指向连续的 64-token 区间，使这个语义
> block 可以被分解成整数个规则 kernel tiles。本文分析的 FLA 路径进一步取
> $B_{\mathrm{kv}}=\ell'=64$，让一个 tile 恰好覆盖一个 block，并让同一 KV group 的
> 16 个 query heads 共享这段 K/V。

> **先说明本文使用的实现。** NSA 论文公开了算法、Triton kernel 的高层
> program mapping，以及 8×A100 上的性能结果，但没有公开论文实验所用的
> 原始 kernel。下面的算法公式来自论文；online Top-k、group-centric forward
> 和 CSR backward 的代码级分析来自当前 FLA 社区实现。FLA 展示了一条可行的
> 实现路径，但它简化了论文中的 learned overlapping compression，也不是
> DeepSeek 的官方实现。`[PAPER] [CODE/community] [GAP]`

NSA 将长程注意力拆成三种访问模式：

$$
\text{压缩的全局上下文}
\;+\;
\text{动态选出的关键 blocks}
\;+\;
\text{固定局部窗口}.
$$

真正关键的推导链不是“三路相加”，而是

$$
\text{信息需求}
\longrightarrow
\text{block IDs}
\longrightarrow
\text{共享选块的 GQA workload}
\longrightarrow
\text{规则的 }16\times64\text{ matmul tile}.
$$

### 3.1 算法契约：三路稀疏各自提供什么

**记号。** 省略 batch 轴，使用 sequence-major layout。对长度为 $T$ 的序列，
$$
Q\in\mathbb{R}^{T\times H_q\times d_k}.
$$

NSA 面向 GQA。论文默认配置采用

$$
H_q=64,\qquad H_{kv}=4,\qquad G=H_q/H_{kv}=16,\qquad d_k=192,\qquad d_v=128.
$$

第 $g$ 个 KV head 对应的 query-head group 为

$$
\mathcal H_g=\{gG,\ldots,(g+1)G-1\}.
$$

三条分支的算法参数是：

| 分支 | 稀疏单位 | 论文默认配置 | 每条 query 的可见规模 |
|---|---:|---:|---:|
| compression | overlapping window | $\ell=32,d=16$ | 约 $L_t/d$ |
| selection | contiguous token block | $\ell'=64,n=16$ | 至多 $n\ell'=1024$ |
| sliding | local token window | $w=512$ | 至多 512 |

其中 $L_t=t+1$ 是位置 $t$ 的因果前缀长度。它包含

$$
C_t=\max\left(0,\left\lfloor\frac{L_t-\ell}{d}\right\rfloor+1\right)
$$

个完整 compression windows，以及

$$
M_t=\left\lceil\frac{L_t}{\ell'}\right\rceil
$$

个因果可见的 selection blocks。

**三路使用独立表示。** 论文为三条分支提供独立的 K/V projection：

$$
K^c\in\mathbb{R}^{T\times H_{kv}\times d_k},\qquad V^c\in\mathbb{R}^{T\times H_{kv}\times d_v},\qquad c\in\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}.
$$

因此，三路即使访问了相同 token，也不能先求 token union，再当成一次 attention。它们使用不同表示、分别归一化，最后才组合 branch outputs。

**Compression。** 每个 KV head 将长度为 $\ell=32$ 的 overlapping window 经 learned compressor 映射为一行 compressed K/V，相邻窗口 stride 为 $d=16$，完整序列产生约 $C_T\approx T/d$ 行 compressed K/V。

位置 $t$、group $g$ 的局部 shapes 为

$$
Q_{t,\mathcal H_g,:}\in\mathbb{R}^{G\times d_k},\qquad
\widetilde K^{\mathrm{cmp}}_{0:C_t,g,:}\in\mathbb{R}^{C_t\times d_k},\qquad
\widetilde V^{\mathrm{cmp}}_{0:C_t,g,:}\in\mathbb{R}^{C_t\times d_v}.
$$

$$
[G,d_k][d_k,C_t]\rightarrow[G,C_t],\qquad [G,C_t][C_t,d_v]\rightarrow[G,d_v].
$$

固定 stride $d$ 时，所有 query 的 compression Attention pair count 仍为

$$
\sum_{t=0}^{T-1}\Theta(C_t)=\Theta(T^2/d).
$$

在相同 feature dimension 下，它相对 dense causal pair count 约减少 $d$ 倍，
但仍不是线性复杂度。这里比较的是算术规模，不是包含 compressor、routing 与
kernel efficiency 后的 wall-clock speedup。

**Sliding window。** 局部分支只访问

$$
\mathcal W_t=\{\max(0,t-w+1),\ldots,t\},\qquad w=512,
$$

提供固定、连续且易于 tiled attention 处理的局部上下文。

**三路组合。** 三条分支分别完成 softmax，得到

$$
O^{\mathrm{cmp}}_{t,h,:},\ O^{\mathrm{slc}}_{t,h,:},\ O^{\mathrm{win}}_{t,h,:}\in\mathbb{R}^{d_v},
$$

再由独立 sigmoid gates 组合：

$$
O_{t,h,:}=\gamma^{\mathrm{cmp}}_{t,h}O^{\mathrm{cmp}}_{t,h,:}+\gamma^{\mathrm{slc}}_{t,h}O^{\mathrm{slc}}_{t,h,:}+\gamma^{\mathrm{win}}_{t,h}O^{\mathrm{win}}_{t,h,:}.
$$

这些 gate 不要求和为 1。执行 DAG 上，sliding branch 可以与
compression/routing chain 并行，selection Attention 则必须等待 block IDs。
至于 compression probabilities 如何到达 routing，属于实现选择：可以物化、
与 Top-K 融合，或者像后文分析的 FLA 路径一样，根据 $Q$、compressed $K$
与 LSE 分块重算。

### 3.2 Routing：从 Compression Probability 到 Top-16 Block IDs

Selection 不训练一套完全独立的 token router。它复用 compression attention 已经学到的 probability，把 compressed-window 重要性重新映射到 64-token selection blocks。

**空间重映射。** 令

$$
P^{\mathrm{cmp}}_{t,h}\in\mathbb{R}^{C_t}
$$

为 query head $h$ 的 compression probability。论文 Eq. 9 将它变成 selection-block score：

$$
R^{\mathrm{slc}}_{t,h}[j]
=
\sum_{m=0}^{\ell'/d-1}
\sum_{u=0}^{\ell/d-1}
P^{\mathrm{cmp}}_{t,h}
\!\left[\frac{\ell'}{d}j-m-u\right],
\qquad j\in\{0,\ldots,M_t-1\}.
$$

越界的 compressed-window index 视为零。论文配置满足

$$
\ell'/d=4,\qquad \ell/d=2.
$$

这个双重求和处理了 overlapping compression windows 与 selection blocks 之间并非一一对应的问题。

**GQA group reduction 与 Top-$n$。** 同一 KV head 的 $G=16$ 个 query heads 必须共享 selection IDs，因此

$$
\bar R^{\mathrm{slc}}_{t,g}[j]
=
\sum_{h\in\mathcal H_g}R^{\mathrm{slc}}_{t,h}[j],
\qquad
\bar R^{\mathrm{slc}}_{t,g}\in\mathbb{R}^{M_t},
$$

$$
I^{\mathrm{slc}}_{t,g}
=
\operatorname{TopKIndex}\!\left(\bar R^{\mathrm{slc}}_{t,g},n\right),
\qquad n=16.
$$

论文 Sec. 4 的具体配置还规定：16 个 selection slots 中固定激活 1 个 initial
block 和最近 2 个 local blocks，其余 slots 才由上述动态 ranking 填充。
`[PAPER]`

固定 initial block 会确定地产生一个 $\Theta(T)$ fan-in 的热门 row；两个 local
slots 则随 query 向前移动，仅由 current + previous 规则产生时，一个具体 block
最多服务约 $2\ell'=128$ 个 query positions。动态 ranking 还可能形成额外热点。
`[DERIVATION]`

对成熟位置，

$$
I^{\mathrm{slc}}_{t,g}
=(j_0,\ldots,j_{n-1})\in\mathbb{Z}^{16}.
$$

它是带有 slot 顺序的 index tensor，而不是数学集合。序列开头可能只有少于 16
个因果有效 blocks；kernel 可维持 16 个 selection slots，但必须用 $-1$ 或
block count 标记 padding。“固定 16 个 slots”不等于“始终存在 16 个有效
blocks”。

完整 shape 链为

```text
compression probabilities       [T,Hq,≈T/16]
  → Eq. 9 spatial remap          [T,Hq,≈T/64]
  → reduce G=16 query heads      [T,Hkv,≈T/64]
  → Top-16                       [T,Hkv,16]
```

**隐藏的物化成本。** 标准 FlashAttention 不会把完整 probability matrix 写回 HBM，通常只返回 output 与 LSE。因此，“复用 compression probability”并不自动回答 probability 如何到达 Top-k。

在 $T=65536$ 上，完整窗口数实际为 $C_T=4095$。为说明固定 shape
allocation 的上界，令 padded capacity

$$
\bar C=\left\lceil T/d\right\rceil=4096.
$$

若按该 capacity 物化 per-head FP32 compressed scores，

$$
[T,H_q,\bar C],
$$

一层需要

$$
65536\times64\times4096\times4=64\ \mathrm{GiB}.
$$

若在 Eq. 9 remap **之前**先对每个 GQA group 求和，物化

$$
[T,H_{kv},\bar C],
$$

仍需

$$
65536\times4\times4096\times4=4\ \mathrm{GiB}.
$$

若讨论的是 remap **之后**的 group-reduced block scores，令
$M_T=\lceil T/\ell'\rceil=1024$，其 shape 应为
$$
[T,H_{kv},M_T],
$$

对应

$$
65536\times4\times1024\times4=1\ \mathrm{GiB}.
$$

这三个数字描述不同的数据流位置，不能混用。

当前 FLA 给出了一种可行机制：保存 compression LSE，根据 Q 与 compressed K
分块重算 probability，并用 bitonic merge 在线维护 Top-$n$，从而不物化上述
二次 score tensor。这条路径以重算换显存：Top-k kernel 必须再次用 Q 扫描
compressed K，因此 routing 仍包含一次 $\Theta(T^2/d)$ 的 QK scan 与候选
merge。它消除的是二次 score tensor 的 HBM materialization，不是二次 pair
scoring 本身。

其配套路径还把 compression 简化为 non-overlap mean pooling、共享 K/V，并对
GQA group size 施加至少 16 且为 2 的幂等实现约束；这些都不是论文算法的通用
定义。因此，FLA 只能证明一种 infrastructure 路径可行，不能证明论文作者采用
了同样实现。`[CODE/community] [GAP]`

### 3.3 Selection Forward：逻辑 Gather，物理 Block Tile

Routing 输出离散 block IDs：

$$
I^{\mathrm{slc}}_{t,g}
=(j_0,\ldots,j_{n-1})\in\mathbb{Z}^{n}.
$$

每个 $j_s$ 指向 selection K/V 中一段连续的 64-token block：

$$
\mathcal B_{j_s}=\{j_s\ell',\ldots,(j_s+1)\ell'-1\},\qquad \ell'=64.
$$

**逻辑视图。** 对成熟 query，如果 16 个 slots 都有效，可以写成

$$
\widetilde K^{\mathrm{slc}}_{t,g}
=
\operatorname{Cat}_{s=0}^{n-1}
K^{\mathrm{slc}}[\mathcal B_{j_s},g,:]
\in\mathbb{R}^{1024\times192},
$$

$$
\widetilde V^{\mathrm{slc}}_{t,g}
=
\operatorname{Cat}_{s=0}^{n-1}
V^{\mathrm{slc}}[\mathcal B_{j_s},g,:]
\in\mathbb{R}^{1024\times128}.
$$

逻辑 attention shapes 为

$$
[16,192][192,1024]\rightarrow[16,1024],\qquad [16,1024][1024,128]\rightarrow[16,128].
$$

这里的 1024 是 padded capacity，不是所有位置的有效 token 数。序列开头可能有无效 slots；包含当前位置的 causal block 也可能只有部分 token 可见。实现必须同时应用 ID mask、range mask 与 causal mask。

这里的 $[1024,d]$ 只是 Cat 之后的**逻辑 shape**。在原始 KV 中，它对应
16 段彼此离散、每段内部 token index 连续的 64-token blocks；它不是一段连续
的 1024-token source range。若物理布局为 token-major
$[T,H_{kv},D]$，固定 head 的相邻 token rows 具有固定 stride，也不一定组成
字节上完全连续的 $64D$ 区间。真正被算法改善的是寻址结构：一个 block 只需
一个离散 base，再由规则 affine offsets 展开 64 行，而不是为 64 个 tokens
分别做一次间接索引。

如果真的为每条 query/group 构造这两个 tensors，external gather 既要处理 16
个不规则 block bases，又要将它们组装为 compact selected-KV tensor 写回 HBM。
仅 K/V payload 就有

$$
1024(192+128)\times2=655{,}360\ \mathrm{bytes}=640\ \mathrm{KiB}
$$

（BF16）。它会制造额外写入，并在 attention 中再次读取同一数据。正确做法是只保存 IDs，在 kernel 内按 block 加载原始 K/V。

**Program ownership。** 一般情况下，Selection forward 中一个 program 的
完整 owner 是

$$
\boxed{
\text{一个 query position}
\times
\text{一个 KV group}
\times
\text{一个 value-feature tile}
}.
$$

等价地，它由后文的 $(a,t,g,\nu)$ 唯一标识。只有当
$N_V=1$ 时，value-feature 轴才退化，此时才可以简写为“一个 query
position $\times$ 一个 KV group”。

先把这一段源码中的输入 shape 与 grid 符号全部展开。对 fixed-length 输入，
FLA 使用

$$
\begin{aligned}
Q&:[B,T_Q,H_q,d_k],\\
K^{\mathrm{slc}}&:[B,T_K,H_{kv},d_k],\\
V^{\mathrm{slc}}&:[B,T_K,H_{kv},d_v],\\
I^{\mathrm{slc}}&:[B,T_Q,H_{kv},N_{\mathrm{slc}}],\\
\texttt{block\_counts}&:[B,T_Q,H_{kv}]
\ \text{or a scalar capacity},\\
O^{\mathrm{slc}}&:[B,T_Q,H_q,d_v],\\
\mathrm{LSE}^{\mathrm{slc}}&:[B,T_Q,H_q].
\end{aligned}
$$

其中：

- $B$ 是 batch size；
- $T_Q,T_K$ 是每条序列在本次调用中的 query 与 KV token 数；
- $H_q,H_{kv}$ 是 query-head 与 KV-head 数；
- $G=H_q/H_{kv}$ 是一个 KV head 服务的 query-head 数；
- $d_k,d_v$ 是每个 head 的 Q/K feature 与 V/output feature 维度；
- $N_{\mathrm{slc}}$ 是 selection-ID 末维的 slot capacity，源码记作
  `S`，本文配置为 16；
- $I^{\mathrm{slc}}$ 保存 block IDs；`block_counts` 若为 tensor，则记录
  每个 query/group 的有效 slot 数，若为 scalar capacity，kernel 扫描全部
  $N_{\mathrm{slc}}$ 个 slots；
- $B_V$ 是一个 program 沿 value feature 轴计算的 tile 宽度；
- $N_V=\lceil d_v/B_V\rceil$ 是覆盖完整 value 维所需的 tile 数。

本文使用的 NSA 数值配置是

$$
H_q=64,\qquad H_{kv}=4,\qquad G=16,\qquad d_k=192,\qquad d_v=128.
$$

FLA 还假设 $Q$ 对应每条 KV sequence 最后的 $T_Q$ 个 tokens。因此
fixed-length 路径中，query slot $t$ 在 KV sequence 内的绝对位置不是
简单的 $t$，而是

$$
p_t=T_K-T_Q+t.
$$

后面的 causal mask 与 selected-block 起点都相对于 $p_t$ 判断。

源码中的 launch grid 是一个三维 **program-count tuple**：

$$
\boxed{
\mathrm{grid}
=
\left(T_Q,\ N_V,\ B\times H_{kv}\right)
}.
$$

源码将三个 program IDs 命名为 `i_t`、`i_v` 与 `i_bh`。为避免
`i_v` 和 value tensor $V$ 混淆，本文把第二个 ID 改记为 $\nu$：

$$
t\in[0,T_Q),\qquad
\nu\in[0,N_V),\qquad
i_{bh}\in[0,B\times H_{kv}).
$$

它们依次表示 query position、value-feature tile，以及被压平的
batch/KV-head 位置。第三个 ID 按

$$
a=\left\lfloor\frac{i_{bh}}{H_{kv}}\right\rfloor,\qquad
g=i_{bh}\bmod H_{kv}
$$

还原成 batch index $a$ 与 KV-head/group index $g$。这里特意用 $a$
表示 batch，避免和后文的 KV block ID $b$ 混淆。第 $g$ 个 group
服务的 query heads 仍是

$$
\mathcal H_g=\{gG,\ldots,(g+1)G-1\}.
$$

再定义 program $\nu$ 负责的 value-feature 区间

$$
\mathcal D_\nu
=
\left\{
f\in\mathbb Z
\;\middle|\;
\nu B_V
\le f
<
\min\!\left((\nu+1)B_V,d_v\right)
\right\}.
$$

于是 program $(t,\nu,i_{bh})$，等价地说 $(a,t,g,\nu)$，是下面这个输出
slice 的唯一 writer：

$$
O^{\mathrm{slc}}
\left[a,t,\mathcal H_g,\mathcal D_\nu\right]
\in
\mathbb{R}^{G\times|\mathcal D_\nu|}.
$$

物理 accumulator 按 $[G,B_V]$ 分配；最后一个不满 $B_V$ 的 tile 通过
feature mask 处理。由此也能看清 axis 顺序：去掉 batch flattening 后 grid 是
$(T_Q,N_V,H_{kv})$，而不是 $(T_Q,H_{kv},N_V)$。

如果 $N_V>1$，同一个 $(a,t,g)$ 会展开成 $N_V$ 个 programs；它们为各自
的 value-feature slice 重算相同的 QK/softmax，但写入的
$\mathcal D_\nu$ 两两不交，因此 output 没有写冲突。LSE 不沿 value 维切分，
源码只允许 $\nu=0$ 的 program 写回
$\mathrm{LSE}^{\mathrm{slc}}[a,t,\mathcal H_g]$。

论文配置 $d_v=128$，当前 FLA 也取 $B_V=128$，所以

$$
N_V=1,\qquad \mathcal D_0=\{0,\ldots,127\}.
$$

此时每个 $(a,t,g)$ 恰好只对应一个 program，它独占
$[G,d_v]=[16,128]$ 的完整 selection output。该 program 随后加载并常驻
逻辑 query tile

$$
Q_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{G\times d_k}
=
\mathbb{R}^{16\times192}.
$$

公开的 variable-length 路径采用 packed $B=1$ 表示。若第 $i$ 条序列分别
含 $T_Q^{(i)}$ 个 query tokens 与 $T_K^{(i)}$ 个 KV tokens，则

$$
T_Q^{\mathrm{total}}=\sum_iT_Q^{(i)},\qquad
T_K^{\mathrm{total}}=\sum_iT_K^{(i)},
$$

$$
\begin{aligned}
Q&:[1,T_Q^{\mathrm{total}},H_q,d_k],\\
K^{\mathrm{slc}}&:[1,T_K^{\mathrm{total}},H_{kv},d_k],\\
V^{\mathrm{slc}}&:[1,T_K^{\mathrm{total}},H_{kv},d_v].
\end{aligned}
$$

此时 grid 变为

$$
\left(T_Q^{\mathrm{total}},N_V,H_{kv}\right).
$$

第一个 program ID $u\in[0,T_Q^{\mathrm{total}})$ 是 packed query slot；
`token_indices_q[u]` 将它映射为
$(i_{\mathrm{seq}},t_{\mathrm{local}})$。令

$$
\mathrm{bos}_Q(i)
=
\texttt{cu\_seqlens\_q}[i]
$$

表示第 $i$ 条序列在 packed Q 中的起始 offset，则物理 output owner 是

$$
O^{\mathrm{slc}}
\left[
0,\,
\mathrm{bos}_Q(i_{\mathrm{seq}})+t_{\mathrm{local}},\,
\mathcal H_g,\,
\mathcal D_\nu
\right].
$$

该 query 在本序列 KV 中的 causal absolute position 相应为

$$
p_{i_{\mathrm{seq}},t_{\mathrm{local}}}
=
T_K^{(i_{\mathrm{seq}})}
-
T_Q^{(i_{\mathrm{seq}})}
+
t_{\mathrm{local}}.
$$

因此 varlen 只替换了 fixed-length 的 $(a,t)$ 定位方式；$\nu$、$g$
两条 grid 轴与“一个 program 唯一写一个 output slice”的 ownership 逻辑
没有改变。

随后循环 16 个 selection slots。本文用 $B_{\mathrm{kv}}$ 表示一次内循环
加载并计算的 KV-token row 数；在当前 FLA 路径中，它对应源码参数 `BS`，并
恰好等于 NSA 的 selection block length $\ell'$：

$$
B_{\mathrm{kv}}=\texttt{BS}=\ell'=64.
$$

对一个有效 block ID $j_s$，program 计算 $\mathrm{base}=j_s\ell'$，并加载

$$
K^{\mathrm{slc}}_{j_s}
\in\mathbb{R}^{B_{\mathrm{kv}}\times d_k}
=
\mathbb{R}^{64\times192},
\qquad
V^{\mathrm{slc}}_{j_s}
\in\mathbb{R}^{B_{\mathrm{kv}}\times d_v}
=
\mathbb{R}^{64\times128}.
$$

这是算法层面的逻辑 tile。在当前 FLA 的 Hopper 分支中，

$$
\texttt{BK}
=
\min\!\left(256,\operatorname{nextPow2}(d_k)\right)
=
256,
$$

所以为了匹配 Triton dot 的 feature tile，实际构造

$$
b_Q:[16,256],\qquad
b_K:[256,64],\qquad
b_V:[64,128],
$$

其中 feature 维 $192\rightarrow256$ 的尾部通过 mask/padding 处理。因此逻辑
contraction 是

$$
[16,192][192,64]\rightarrow[16,64],\qquad [16,64][64,128]\rightarrow[16,128].
$$

而 QK 的物理 padded execution shape 是

$$
[16,256][256,64]\rightarrow[16,64].
$$

这组 $192\rightarrow256$ 的执行 shape 是 **当前 FLA Hopper 路径**的
结论。非 Hopper 分支将 `BK` 上限设为 128；当 $d_k=192$ 时会得到两个
feature tiles，并触发源码的 `NK == 1` 断言，因此不能把这里的
$[16,256][256,64]$ 直接外推到该路径。

动态性只存在于每轮加载前的 block base；进入片上 tile 后，计算重新变成固定 shape。

这里特意不用 $B_K$ 表示 token tile。NSA 论文中的 token-tile 记号 $B_K$ 在当前 FLA 源码中对应 `BS`；FLA 的 `BK` 表示 padded feature width，例如

$$
d_k=192\longrightarrow \texttt{BK}=256.
$$

混用两套记号会把“64 个 tokens”误读成“256 个 feature elements”。

**在线归一化。** Program 对 16 个 blocks 使用前文统一介绍的 exact online-softmax merge，只维护每个 query head 的 running max、normalizer 与 output accumulator，不写出 $P^{\mathrm{slc}}\in\mathbb{R}^{16\times1024}$。所有 selected blocks 完成后直接写回

$$
O^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{16\times128}.
$$

因此 output 拥有唯一 writer，forward 不需要对 output 做 atomic reduction。

**为什么 head sharing 是执行契约。** 对一个 64-token block，QK 与 PV 的乘加量约为

$$
2\,G\,B_{\mathrm{kv}}(d_k+d_v)
=
2\times16\times64\times(192+128)
=
655{,}360\ \mathrm{FLOPs}.
$$

若 K/V 为 BF16，本轮主要 K/V payload 为

$$
2\,B_{\mathrm{kv}}(d_k+d_v)
=
2\times64\times(192+128)
=
40{,}960\ \mathrm{bytes}.
$$

只看这份被 16 个 heads 共享的 K/V payload，局部 arithmetic intensity 为

$$
\frac{
2\,G\,B_{\mathrm{kv}}(d_k+d_v)
}{
2\,B_{\mathrm{kv}}(d_k+d_v)
}
=
G
=
16\ \mathrm{FLOP/byte}.
$$

这不是整 kernel 的 roofline 数字：它没有计入 Q、output、indices、LSE、mask 和 launch 开销。它只说明，同一 K/V block 被越多 query heads 复用，一次不连续寻址带来的搬运就越容易被规则 matmul 摊薄。

对成熟 query，selection 主循环有 16 个 slots；序列开头仍运行相同控制结构，但无效 slots 被 mask。NSA 提供的是“固定容量、可 padding 的 workload”，而不是没有边界条件的绝对固定工作量。

### 3.4 一条公开的 Backward 路径：FLA 的稀疏图转置

> **Takeaway.** Forward 的 `query → selected blocks` 关系只给 output 找到了
> owner。若继续使用 query-centric ownership，
> $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 需要浮点 atomics
> 或 partial-gradient reduction；本文审计的 FLA 社区实现选择把它转置为
> `KV block → selecting queries`，再让每个 KV-block program 完成自己的
> many-to-one reduction。

Forward 的自然 owner 是 query，因为每个 program 能独立完成一个
$O^{\mathrm{slc}}_{t,\mathcal H_g,:}$。Backward 则有两种归约方向。

以下只讨论 selection branch 的 backward。完整 NSA 还包括 gates、compression
branch、sliding branch，以及三个分支对共享 Q/input 的梯度合并。

**保存与重算。** Selection backward 不需要保存完整 probability matrix。当前
FLA 首先由 forward output 与上游梯度计算每个 softmax row 的

$$
\Delta_{a,t,h}
=
\sum_{f=0}^{d_v-1}
O^{\mathrm{slc}}_{a,t,h,f}
\,dO^{\mathrm{slc}}_{a,t,h,f}.
$$

随后根据 $Q,K^{\mathrm{slc}},I^{\mathrm{slc}}$ 与
$\mathrm{LSE}^{\mathrm{slc}}$ 分块重算 score/probability。对
batch/sequence $a$、query $t$、group $g$ 与 selected block $b$，
有

$$
dP^{\mathrm{slc}}_{a,t,g,b}
=
dO^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
\left(V^{\mathrm{slc}}_{a,g,b}\right)^\top
\in\mathbb{R}^{G\times B_{\mathrm{kv}}},
$$

$$
dS^{\mathrm{slc}}_{a,t,g,b}
=
P^{\mathrm{slc}}_{a,t,g,b}
\odot
\left(
dP^{\mathrm{slc}}_{a,t,g,b}
-
\Delta_{a,t,\mathcal H_g}[:,\mathrm{None}]
\right)
\in\mathbb{R}^{G\times B_{\mathrm{kv}}}.
$$

这里 $V^{\mathrm{slc}}_{a,g,b}$ 表示 block $b$ 内
$B_{\mathrm{kv}}$ 行 value vectors，shape 为
$[B_{\mathrm{kv}},d_v]$。相应的局部量 shapes 为

$$
S^{\mathrm{slc}}_{a,t,g,b},\qquad
P^{\mathrm{slc}}_{a,t,g,b},\qquad
dS^{\mathrm{slc}}_{a,t,g,b}
\in\mathbb{R}^{G\times B_{\mathrm{kv}}}.
$$

若不使用 activation checkpoint，FLA selection autograd state 至少包括

$$
Q,\quad K^{\mathrm{slc}},\quad V^{\mathrm{slc}},\quad
O^{\mathrm{slc}},\quad \mathrm{LSE}^{\mathrm{slc}},\quad
I^{\mathrm{slc}},
$$

以及 variable-length 场景所需的 block counts 和 sequence metadata。当前 FLA
的 autograd 路径实际保存 `q, k, v, o, lse`，并在 context 中保留 block
indices 等 metadata。准确结论是“不保存 $P^{\mathrm{slc}}$”，而不是
“forward 只需保存
$O^{\mathrm{slc}},\mathrm{LSE}^{\mathrm{slc}},I^{\mathrm{slc}}$”。

**$dQ$：继续由 query 拥有。** 对 query $t$，只有它自己选择的 blocks 参与

$$
dQ^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
=
\alpha
\sum_{b\in I^{\mathrm{slc}}_{a,t,g}}
dS^{\mathrm{slc}}_{a,t,g,b}
K^{\mathrm{slc}}_{a,g,b}.
$$

每个 block 的局部 shape 为

$$
[G,B_{\mathrm{kv}}][B_{\mathrm{kv}},d_k]\rightarrow[G,d_k].
$$

同一 program 可以循环全部 slots。逻辑上的最终结果为

$$
dQ^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{16\times192},
$$

而当前 Triton kernel 取

$$
\texttt{BK}=\operatorname{nextPow2}(d_k)=256,
$$

所以物理 accumulator `b_dq` 的 shape 是 $[16,256]$；末尾 64 个 padded
feature lanes 由 mask 处理，最终只写回前 192 维。因此 dQ 仍适合
query-centric ownership。

源码中 fixed-length 的 dQ launch grid 为

$$
\mathrm{grid}_{dQ}
=
\left(T,N_V,BH_{kv}\right).
$$

Backward 取

$$
B_V=\max\!\left(\operatorname{nextPow2}(d_v),16\right),
\qquad
N_V=\left\lceil\frac{d_v}{B_V}\right\rceil=1,
$$

使 softmax backward 中的 $\Delta$ 始终对应完整 value 维。因此当前路径中
program $(t,0,i_{bh})$，等价地说 $(a,t,g)$，独占一个
$[G,d_k]$ 的 dQ slice。variable-length 路径只把第一条 grid 轴换成 packed
token 位置，ownership 不变。

**$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$：query-centric 会产生写冲突。**
一个 KV block $b$ 可能被许多 queries 选中。令

$$
\mathcal Q_{a,g,b}
=
\left\{t\mid b\in I^{\mathrm{slc}}_{a,t,g}\right\}.
$$

它的梯度是跨 queries 的 many-to-one reduction：

$$
dK^{\mathrm{slc}}_{a,g,b}
=
\alpha
\sum_{t\in\mathcal Q_{a,g,b}}
\left(dS^{\mathrm{slc}}_{a,t,g,b}\right)^\top
Q_{a,t,\mathcal H_g,:},
$$

$$
[B_{\mathrm{kv}},G][G,d_k]
\rightarrow
[B_{\mathrm{kv}},d_k],
$$

$$
dV^{\mathrm{slc}}_{a,g,b}
=
\sum_{t\in\mathcal Q_{a,g,b}}
\left(P^{\mathrm{slc}}_{a,t,g,b}\right)^\top
dO^{\mathrm{slc}}_{a,t,\mathcal H_g,:},
$$

$$
[B_{\mathrm{kv}},G][G,d_v]
\rightarrow
[B_{\mathrm{kv}},d_v].
$$

若沿用 query owner，不同 programs 会同时更新相同的
$dK^{\mathrm{slc}}_b,dV^{\mathrm{slc}}_b$，只能使用大量 floating-point
atomics，或写 partial buffers 再归约。更自然的办法是先转置稀疏关系：

```text
forward adjacency:   query → selected KV blocks
backward adjacency:  KV block → all selecting queries
```

然后让一个 program 拥有一个 $(a,g,b)$ KV block。

**用 CSR 表示转置后的关系。** 这里先看 fixed-length 路径。令完整序列的
block 数为

$$
M=\left\lceil T/\ell'\right\rceil.
$$

将一个 KV block $(a,g,b)$ 压平为 row

$$
r(a,g,b)=\bigl(aH_{kv}+g\bigr)M+b.
$$

CSR 用两个一维 arrays 保存“每个 block 被哪些 queries 选中”：

$$
\texttt{csr\_offsets}
\in\mathbb{Z}^{B H_{kv}M+1},
$$

$$
\texttt{csr\_indices}
\in\mathbb{Z}^{E_{\mathrm{alloc}}},
\qquad
E_{\mathrm{alloc}}=B T H_{kv}n.
$$

其中 `csr_indices` 分配到 padded edge capacity；只有
`csr_offsets[-1]` 之前的 prefix 有效。源码存入的是压平后的 absolute query
position

$$
q_{\mathrm{abs}}=aT+t,
$$

因此 row $r=r(a,g,b)$ 满足

$$
\left\{aT+t\mid t\in\mathcal Q_{a,g,b}\right\}
=
\texttt{csr\_indices}
\!\left[
\texttt{csr\_offsets}[r]:
\texttt{csr\_offsets}[r+1]
\right].
$$

当前 FLA 通过 counting-scatter 构造它：

```text
遍历有效 selection edges
  → atomic count 每个 (batch,group,block) 的 fan-in
  → prefix sum 得到 csr_offsets
  → 再遍历 edges，用 atomic cursor scatter query IDs
```

atomic 只作用于紧凑整数 metadata；昂贵的
$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 浮点归约则获得唯一 owner。variable-length
路径改用 packed block/query IDs，但“一个 CSR row 对应一个 KV-block owner”的
关系不变。这里真正重要的不是 CSR 格式本身，而是它完成了从 query-centric
到 KV-centric 的稀疏图转置。

**KV-centric dKV tile。** fixed-length 的 dKV launch grid 为

$$
\mathrm{grid}_{dKV}
=
\left(N_V,BH_{kv}M\right).
$$

由于当前 backward 路径中 $N_V=1$，program
$(0,r(a,g,b))$ 是 KV block $(a,g,b)$ 的唯一 writer。variable-length
路径将第二条 grid 轴替换为“packed KV blocks 数 $\times H_{kv}$”，但仍然
由一个 program 独占一个 KV block。该 program 持有

$$
K_b\in\mathbb{R}^{B_{\mathrm{kv}}\times d_k},\qquad V_b\in\mathbb{R}^{B_{\mathrm{kv}}\times d_v}
$$

和同 shape 的 gradient accumulators。它从对应 CSR row 每次读取 $B_{\mathrm{qry}}$ 个 query positions，并展开每个 query 的 $G$ 个 heads：

$$
R=B_{\mathrm{qry}}G.
$$

批内 query 与 output-gradient shapes 为

$$
Q_{\mathrm{batch}}\in\mathbb{R}^{R\times d_k},\qquad dO_{\mathrm{batch}}\in\mathbb{R}^{R\times d_v}.
$$

局部重算与归约重新形成规则 matmul：

$$
[B_{\mathrm{kv}},d_k][d_k,R]\rightarrow[B_{\mathrm{kv}},R],
$$

$$
[B_{\mathrm{kv}},R][R,d_v]\rightarrow[B_{\mathrm{kv}},d_v],\qquad [B_{\mathrm{kv}},R][R,d_k]\rightarrow[B_{\mathrm{kv}},d_k].
$$

例如 $B_{\mathrm{kv}}=64,B_{\mathrm{qry}}=4,G=16$ 时，$R=64$。逻辑上的
score-recompute contraction 为

$$
[64,192][192,64]\rightarrow[64,64].
$$

当前 Triton kernel 的 $\texttt{BK}=256$，所以实际 `tl.dot` tile 是

$$
[64,256][256,64]\rightarrow[64,64],
$$

其中末尾 64 个 feature lanes 由 mask 补零；`b_dk` 也相应使用
$[64,256]$ 的物理 accumulator，再只写回前 192 维。

CSR 的作用不只是压缩 metadata。它把“多个 queries 向同一 KV gradient 地址写入”的冲突，改写为“一个 KV owner 顺序读取自己的 queries”，再在 owner 内部恢复规则 Tensor Core tile。

**剩余的不规则性：fan-in。** 定义

$$
NQ_{a,g,b}=|\mathcal Q_{a,g,b}|.
$$

当 `block_counts=16` 是固定标量 capacity 时，
$dQ^{\mathrm{slc}}$ program 扫描固定的 16 个 padded slots；FLA 也接受逐
query 的 tensor `block_counts`，此时运行时 $N_S$ 可以缩短。dKV program
的循环长度则由 $NQ_{a,g,b}$ 决定。fixed initial/sink block，以及由动态 ranking 形成的热门
blocks，可能拥有远高于平均值的 fan-in，形成长尾 CTA；移动的 local slots
本身只产生有界的局部 fan-in。拆分热门 row 可以改善负载均衡，却会重新引入
partial reduction 或 atomic。Backward 因而把问题从“跳过多少 FLOPs”转成了
“如何调度 many-to-one reduction”。

本文没有继续把 fan-in 长尾做成性能实验。一方面，论文作者使用的 backward
kernel 没有公开；另一方面，社区实现中改变 routing topology 时，fan-in skew、
query gather locality 与 CSR row 顺序会一起变化，很难把 latency 差异归因给
单一变量。因此，FLA backward 将不规则性从浮点写冲突转移成了 CSR row
length、query-gather locality 与 CTA load balance。本文没有对这些因素报告
性能测量；这里只把它作为源码可确认的执行后果，而不声称它解释了论文作者
kernel 的 backward latency。`[DERIVATION] [CODE/community] [GAP]`

### 3.5 NSA 的规则性来自算法契约，不来自 Kernel 魔法

在论文默认配置下，NSA selection 能映射为高效 kernel，是因为算法主动提供
三项执行契约：

$$
\boxed{\text{连续的 64-token blocks}+\text{16 个 heads 共享 IDs}+\text{固定 16 个可 padding slots}}
$$

Routing 输出的每个动态 ID 已经对应一段连续的 64-token block。因此，kernel
在 load 之前就知道：不规则性只存在于 block base，block 内 64 个 token 可以
通过规则的 affine offsets 展开。Program 加载规则的 $[64,d]$ K/V tile，
再与共享该组 IDs 的 16 个 query heads 形成 $[16,64]$ score tile。本文分析
的 FLA community backward 则进一步转置稀疏图，让 KV block 成为
$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 的唯一 owner。

这也给出三个不能越过的边界。

第一，NSA 不是“任意 block sparsity 都会自动加速”的证据。真正产生硬件效率的是连续 block、head sharing、固定 slot capacity 与明确 ownership 同时成立。

第二，selection 与 sliding 的 token budget 在成熟位置近似固定，但 compression 仍访问约 $L_t/d$ 行 compressed K/V。固定 $d$ 时，全层 compression 仍为 $\Theta(T^2/d)$。

第三，FLA 展示了 online Top-k、group-centric forward 与 CSR backward 的一套完整可行机制；它不能证明论文作者的未公开实现采用相同 routing materialization、tile 参数或 inverse-adjacency construction。

论文报告，在其 $G=16,\ell'=64,n=16$ 配置与 A100 测试环境中，作者的 Triton NSA 相比 Triton FlashAttention-2 在 64K 达到 9.0× forward 与 6.0× backward speedup。这个结果可以归因于论文报告的完整系统，不能直接归因于本文从社区代码拆出的某一个 kernel 选择。`[PAPER] [GAP]`

换句话说，NSA 最值得复用的不是某一条稀疏公式，而是一种共同设计方法：

> 先让算法输出 GPU 能消费的稀疏单位，再分别为 forward output 与 backward gradient 选择唯一 owner。

**本节证据来源**

- [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
  Attention*](https://arxiv.org/abs/2502.11089)，Sec. 3.2–3.4、Sec. 4–5。
  `[PAPER]`
- [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
  commit
  [`0a9b9f2`](https://github.com/fla-org/flash-linear-attention/tree/0a9b9f222e86b9a895c2447767e9b4cce6c8d530)：
  [`fla/ops/nsa/parallel.py`](https://github.com/fla-org/flash-linear-attention/blob/0a9b9f222e86b9a895c2447767e9b4cce6c8d530/fla/ops/nsa/parallel.py)
  与
  [`fla/ops/utils/csr.py`](https://github.com/fla-org/flash-linear-attention/blob/0a9b9f222e86b9a895c2447767e9b4cce6c8d530/fla/ops/utils/csr.py)。
  `[CODE/community]`
- 本文的实现一致性检查见
  [`NSA_IMPLEMENTATION_AUDIT.md`](../../../paper/notes/NSA_IMPLEMENTATION_AUDIT.md)
  与
  [`NSA_INFRA_DOSSIER.md`](../../../paper/notes/NSA_INFRA_DOSSIER.md)。

## 4. DSA：把任意 Token Selection 重新变成规则矩阵

> **Takeaway.** DSA 保留任意 token-level selection，却先让 128 个主 heads
> 共享同一组 IDs，再由 FlashMLA 将随机 row address 隔离在 HBM→SMEM 的
> producer 边界；算法给出复用轴，kernel 才能把离散 rows 恢复成规则矩阵。

NSA 用连续 block 把稀疏性限制在规则 tile 之间。DeepSeek Sparse Attention（DSA）选择得更细：每个 query 可以从完整历史中挑出任意 token。

这提高了 selection 的自由度，也破坏了 block attention 最依赖的局部连续性。如果按字面实现，系统会先收集一批离散 KV row，写出一个 selected-KV tensor，再把它交给普通 attention kernel。这样虽然少算了大部分 QK 与 PV，却把随机 gather 和一次额外的 HBM 往返留在了 kernel 边界。

DSA 的关键不只是 Top-K，而是三个层次共同改变了执行形状：

$$
\text{低维 Indexer 扫描历史} \rightarrow \text{所有主 heads 共享 token IDs} \rightarrow \text{FlashMLA 在 SMEM 中重建规则 tile}.
$$

下面只保留理解这条链所需的记号。

| 符号 | 含义 | DeepSeek-V3.2 |
|---|---|---:|
| $T_Q,T_K$ | 本次调用的 query 数与可见 KV 数 | 运行时决定 |
| $H_I,D_I$ | Indexer head 数与维度 | $64,128$ |
| $H$ | 主 MLA query heads | $128$ |
| $D_C,D_R$ | latent KV 与 RoPE 维度 | $512,64$ |
| $D_N$ | 原始主 MHA NoPE 维度 | $128$ |
| $D_V^{\mathrm{MHA}}$ | 原始 MHA value 维度 | $128$ |
| $\kappa$ | 每个 query 最多选择的 tokens | $2048$ |

这里的 $D_V^{\mathrm{MHA}}=128$ 不能和 FlashMLA 内部的 latent value 维度 $D_C=512$ 混写。主 Attention 也有两套代数等价、执行 shape 不同的表示：

$$
D_{QK}^{\mathrm{MHA}} =D_N+D_R =192,
$$

$$
D_{QK}^{\mathrm{MQA}} =D_C+D_R =576.
$$

### 4.1 Indexer：便宜的全历史扫描

**先分清两种 score。** DSA 一层里同时存在：

$$
I:[T_Q,T_K],
$$

以及

$$
A^{\mathrm{slc}}:[T_Q,H,\kappa].
$$

$I$ 是 Lightning Indexer 的 routing score，只负责产生 token address；$A^{\mathrm{slc}}$ 是主 MLA 在 selected tokens 上重新计算的 attention logit，只有它会进入主 Attention 的 softmax。

因此 Indexer 决定“去哪里看”，并不直接决定“看见以后怎样加权”。

**共享 query latent。** 归一化 hidden states

$$
X:[T_Q,7168]
$$

先生成主 MLA 与 Indexer 共用的低秩 query latent：

$$
C^Q = \operatorname{RMSNorm} \left( X(W^{Q,A})^{\mathsf T} \right) : [T_Q,1536].
$$

Indexer 不再从 7168-D hidden 独立做一次大投影，而是从 $C^Q$ 得到：

$$
\widehat Q^I = C^Q(W^{IQ})^{\mathsf T} : [T_Q,1536] \rightarrow [T_Q,64\cdot128].
$$

reshape 后：

$$
Q^I:[T_Q,64,128].
$$

发布版公开 DeepGEMM 接口以物化的 FP8 Q 为输入；生产系统是否把这段 projection、旋转和量化进一步融合，公开代码没有给出答案。

历史 token 的 Indexer key 则只有一条共享向量：

$$
k^I_s = \operatorname{LayerNorm} \left( W^{IK}x_s \right) \in\mathbb R^{128}.
$$

持久 cache 的逻辑 shape 为：

$$
K^I_{\mathrm{cache}}:[T_K,128].
$$

这意味着 64 个 Indexer query heads 共享同一个历史 key，而不是为每个 head 维护一份 K cache。

Indexer 还从当前 hidden state 直接生成 query-dependent head weights：

$$
U^I
=
X(W^{IW})^{\mathsf T}
:
[T_Q,7168]
\rightarrow
[T_Q,64].
$$

其中 $u^I_{t,j}$ 是 query $t$ 对第 $j$ 个 Indexer head 的基础权重；
这条分支不经过 softmax。把官方 reference 的两个固定 normalization 吸收后，
定义

$$
w^I_{t,j}
=
u^I_{t,j}H_I^{-1/2}D_I^{-1/2}.
$$

需要注意，只有 Indexer Q 复用 $C^Q$；Indexer K 与这里的 weights 都直接
由 $X$ 生成。

**RoPE、Hadamard 与 FP8 是一个执行契约。** 每条 128-D Indexer Q/K 被拆成：

$$
128=64_{\mathrm{RoPE}}+64_{\mathrm{NoPE}}.
$$

前 64 维应用 non-interleaved RoPE，与主 MLA 使用的 interleaved RoPE 不同。这不是排版差异：若两边使用不同的 pair layout，对应位置的 dot product 就会改变。

旋转后再应用归一化 Hadamard 变换 $\mathcal H_{128}$：

$$
\bar q=\mathcal H_{128}q, \qquad \bar k=\mathcal H_{128}k,
$$

$$
\mathcal H_{128}^{\mathsf T}\mathcal H_{128}=I.
$$

所以量化前有：

$$
\bar q^{\mathsf T}\bar k = q^{\mathsf T}k.
$$

Hadamard 不改变精确实数内积；它的作用是扩散 outlier，使后续 E4M3 量化更稳定。“内积不变”不能外推成“量化后逐 bit 不变”。

> **关于两个 $H$ 的记号。** $H_I=64$ 是一个标量，表示 Indexer head
> 数；$\mathcal H_{128}\in\mathbb R^{128\times128}$ 才是 Hadamard 变换
> 矩阵。它的元素为 $\pm1/\sqrt{128}$，是固定、正交且不参与学习的线性
> 变换。实现无需在内存中物化这张 $128\times128$ 矩阵，而是用 fast
> Hadamard transform 通过分层加减在 $O(128\log128)$ 时间内完成。Q/K 使用
> 同一个 $\mathcal H_{128}$，所以精确算术下内积不变；它位于 partial RoPE
> 之后、FP8 量化之前，量化后的内积则只近似保持。

令 $q^{(8)},k^{(8)}$ 表示 E4M3 数据，$s^Q_{t,j},s^K_s$ 表示相应
scale。官方 reference 将 query scale 与两个 normalization 一起折入
query-dependent weight：

$$
\widetilde w_{t,j}
=
w^I_{t,j}s^Q_{t,j}
=
u^I_{t,j}H_I^{-1/2}D_I^{-1/2}s^Q_{t,j}.
$$

于是 scorer 的实现语义可写成：

$$
I_{t,s} \approx s^K_s \sum_{j=0}^{63} \widetilde w_{t,j} \operatorname{ReLU} \left( \left\langle q^{(8)}_{t,j}, k^{(8)}_s \right\rangle \right).
$$

这里的 $q^{(8)},k^{(8)}$ 明确指经过 partial RoPE、Hadamard 与 FP8 量化后的向量，不再和变换前的 $q^I,k^I$ 混用。

Indexer K cache 的逻辑 payload 是：

$$
128\ {\rm B\ of\ E4M3} + 4\ {\rm B\ scale} = 132\ {\rm B/token}.
$$

与完整主 KV row 相比，这是全历史扫描带宽能够显著下降的基础。

**不能物化 per-head routing logits。** 按数学定义，固定 query $t$ 时先有：

$$
R^I_t = Q^I_t \left( K^I_{\mathrm{cache}} \right)^{\mathsf T} : [64,128][128,T_K] \rightarrow [64,T_K].
$$

全局 logical shape 是：

$$
R^I:[T_Q,64,T_K].
$$

再经过 ReLU、weight 和 head reduction：

$$
I_{t,s} = \sum_{j=0}^{63} w^I_{t,j} \operatorname{ReLU} \left( R^I_{t,j,s} \right),
$$

$$
I:[T_Q,T_K]_{\mathrm{FP32}}.
$$

如果把 $[T_Q,64,T_K]$ 写入 HBM，低维 Indexer 会先制造一个比最终 score 大 64 倍的中间量。发布版 DeepGEMM scorer 因而把矩阵乘、ReLU、query-dependent weighting 与 64-head reduction 融合起来，只写最终的 FP32 $I$。

它的一种核心 tile 按代码中的 GEMM 方向可以概括为：

$$
Q_{\mathrm{tile}}:[2,64,128]
\rightarrow
Q_{\mathrm{flat}}:[128,128],
\qquad
K_{\mathrm{tile}}:[256,128],
$$

$$
\underbrace{K_{\mathrm{tile}}}_{[256,128]}
\underbrace{Q_{\mathrm{flat}}^{\mathsf T}}_{[128,128]}
\rightarrow
Z:[256,128]
\equiv
[256,2,64]
\rightarrow
I_{\mathrm{tile}}:[2,256].
$$

最后一步对 64 个 Indexer heads 做 ReLU、query-dependent weighting 与
reduction，再将 candidate-major accumulator fragments 直接映射到逻辑
$[2,256]$ 的输出地址。这里没有单独的 transpose 阶段。中间的 64-head
结果 $Z$ 停留在片上；HBM 只看到每个 query 对 256 个 candidates 的最终
score tile。

但这个发布版 scorer **不包含 Top-K**。公开 DeepGEMM pipeline 仍然为

$$
I:[T_Q,T_K]_{\mathrm{FP32}}.
$$

分配并物化完整的矩形 carrier。主 scorer 写入每行的合法 candidate 区域；
启用 `clean_logits` 时，invalid/future 区域由独立 cleanup kernel 填充。
随后 exact Top-K 仍需重新扫描这张完整矩形结果。

因此 DeepGEMM 已经完成的是 **scorer 内部融合**：FP8 matrix multiply、ReLU、
query-dependent weighting 与 64-head reduction 在同一 kernel 中完成，
$[T_Q,64,T_K]$ 的 per-head scores 不写入 HBM。但这不是 Indexer–TopK
fusion。发布版公开 pipeline 仍然物化完整的
$I:[T_Q,T_K]_{\mathrm{FP32}}$，后续 exact Top-K 仍由独立实现重新扫描。

**数学 Top-K 与 kernel tensor 是两个层次。** 对 query 位置 $t$，令 $p_t$ 表示最后一个 causal 可见位置。数学上：

$$
\kappa_t = \min(\kappa,p_t+1),
$$

$$
J_t
=
\operatorname{TopKIndex}
\left(
I_{t,0:p_t+1},
\kappa_t
\right)
:
[\kappa_t].
$$

$J_{t,r}$ 是 token address，不是 routing score。同一个 query 的 128 个主 heads 共享 $J_t$，不同 query 的集合则可以完全不同。

公开接口还需要两步 shape/dtype 适配。

第一，官方 inference reference 的 `torch.topk` 返回 INT64 indices，并以一次调用的 `min(index_topk, end_pos)` 作为固定末维；causal mask 负责让未来位置保持无效。

第二，在本文的 DSA 配置下，FlashMLA sparse-prefill 接收固定宽度为
$\kappa=2048$ 的：

$$
J^{\mathrm{kernel}} : [T_Q,1,\kappa]_{\mathrm{INT32}}.
$$

若某行只有 $\kappa_t<\kappa$ 个合法 tokens，其余 slots 必须改写为 invalid
sentinel，例如 `-1` 或不小于 $T_K$ 的地址。不能重复某个合法 token 来填满，
否则 softmax 会把该 token 计算多次。

从 reference 的 INT64 Top-K 到 kernel 的固定宽度 INT32 tensor，因而是一项真实的数据准备工作，不是可以在 shape 图里省略的类型标注。

**Hard Top-K 的训练边界。** Top-K IDs 是离散控制流；除排序边界外，
language-modeling loss 不能通过“某个 token 是否被选中”这件事提供普通连续
梯度。Dense warm-up 阶段冻结主模型，以跨主 attention heads 求和、再沿序列
维 L1 归一化的 dense-attention target，通过 KL loss 单独训练 Indexer。Sparse
stage 中，主模型以 language-modeling loss 训练，Indexer 继续以当前 selected
set 上的 KL loss 训练；论文显式 detach Indexer input，使 Indexer 只接收
$\mathcal L_I$ 的训练信号，而主模型只接收 language-modeling loss。因此
主模型适应的是由当前 Indexer 动态产生的稀疏选择，而不是一组固定 token IDs。
这解释了 routing 如何学习，但不改变推理时 exact Top-K 的执行成本。

### 4.2 MLA bridge：为什么主 Kernel 是 576-D MQA

Indexer 只产生地址。真正的 attention 权重仍由主 MLA 计算。要读懂 FlashMLA 的输入 shape，需要先把 MLA 从原始 MHA 表示重排成 MQA 表示。

**原始 score 表示。** 共享 query latent 经过主 MLA 的 query up-projection：

$$
Q^{\mathrm{raw}}
=
C^Q(W^{QB})^{\mathsf T}
:
[T_Q,1536][1536,128\cdot192]
\rightarrow
[T_Q,128,192].
$$

每个 head 拆成：

$$
q^C_{t,h}\in\mathbb R^{128}, \qquad q^R_{t,h}\in\mathbb R^{64}.
$$

$q^R$ 使用主 MLA 的 interleaved RoPE。另一侧，hidden states 先投影出
512-D KV latent 与 64-D RoPE key：

$$
[C^{KV}_{\mathrm{pre}};K^R_{\mathrm{pre}}]
=
X(W^{DKV})^{\mathsf T}
:
[T_K,7168][7168,576]
\rightarrow
[T_K,576].
$$

前 512 维经过 RMSNorm 得到 $C^{KV}$，后 64 维经过 RoPE 得到
$K^R$。因此 KV cache 保存：

$$
C^{KV}:[T_K,512], \qquad K^R:[T_K,64].
$$

若直接恢复成 MHA，每个 head 的 content key 与 value 分别为：

$$
k^C_{s,h} = W_h^{UK}c^{KV}_s \in\mathbb R^{128},
$$

$$
v_{s,h} = W_h^{UV}c^{KV}_s \in\mathbb R^{128},
$$

其中：

$$
W_h^{UK}, W_h^{UV} \in \mathbb R^{128\times512}.
$$

这会得到：

$$
K^{\mathrm{MHA}}:[T_K,128,192],
$$

$$
V^{\mathrm{MHA}}:[T_K,128,128].
$$

如果真的为 128 个 heads 展开这些 tensors，MLA 压缩 cache 的意义就会在 attention 前被抵消。

**把 key up-projection 吸收到 query。** NoPE score 中的 content 项满足：

$$
(q^C_{t,h})^{\mathsf T} W_h^{UK} c^{KV}_s = \left( (W_h^{UK})^{\mathsf T}q^C_{t,h} \right)^{\mathsf T} c^{KV}_s.
$$

定义：

$$
q^A_{t,h} = (W_h^{UK})^{\mathsf T}q^C_{t,h} \in\mathbb R^{512}.
$$

再把 RoPE 部分拼回去：

$$
\widetilde q_{t,h} = [q^A_{t,h};q^R_{t,h}] \in\mathbb R^{576},
$$

$$
\widetilde k_s = [c^{KV}_s;k^R_s] \in\mathbb R^{576}.
$$

于是主 QK 的执行 shape 变成：

$$
\widetilde Q : [T_Q,128,576],
$$

$$
\widetilde K : [T_K,1,576].
$$

这里的 “1” 表示所有主 query heads 共享同一条 MQA key row。

执行 dot product 的向量虽然是 576-D，softmax scale 仍继承原始 192-D MHA
score。公开 reference 在长上下文配置下使用

$$
m_{\mathrm{YaRN}}
=
1+0.1\,m_{\mathrm{cfg}}\ln r_{\mathrm{RoPE}},
$$

$$
\alpha
=
\frac{1}{\sqrt{192}}
\begin{cases}
m_{\mathrm{YaRN}}^2,&L_{\max}>L_{\mathrm{original}},\\
1,&\text{otherwise},
\end{cases}
$$

其中 $m_{\mathrm{cfg}}$ 与 $r_{\mathrm{RoPE}}$ 分别是配置中的 YaRN 系数与
RoPE 扩展因子。$\alpha$ 不能机械地替换成 $576^{-1/2}$，因为 576-D
只是代数吸收后的执行表示，不是新的模型 score 定义。

**把 value up-projection 移到 Attention 之后。** 原始 MHA 的 head output 为：

$$
\sum_s P_{t,h,s} W_h^{UV} c^{KV}_s.
$$

利用线性性：

$$
\sum_s P_{t,h,s} W_h^{UV} c^{KV}_s = W_h^{UV} \left( \sum_s P_{t,h,s} c^{KV}_s \right).
$$

因此 MQA core 直接使用：

$$
\widetilde V_s = c^{KV}_s \in\mathbb R^{512},
$$

先得到 latent output：

$$
O^C : [T_Q,128,512],
$$

再通过每个 head 的 $W_h^{UV}$ 恢复：

$$
O^{\mathrm{MHA}} : [T_Q,128,128].
$$

最后将 heads 展平并应用
$W^O\in\mathbb R^{7168\times16384}$：

$$
Y
=
\operatorname{ConcatHeads}(O^{\mathrm{MHA}})(W^O)^{\mathsf T}
:
[T_Q,16384][16384,7168]
\rightarrow
[T_Q,7168].
$$

FlashMLA sparse core 只返回 $O^C:[T_Q,128,512]$；$W^{UV}$ 恢复与
$W^O$ 都在该 kernel 之外。

在精确算术下，MQA mode 不是新的近似，而是 MLA 的代数重排；实际 BF16/FP8
量化与不同运算顺序不保证逐 bit 相同。论文 Appendix A 给出 MHA/MQA 的代数
关系，官方 Python inference reference 的 prefill 分支仍显式 up-project K/V
并执行 MHA，decode 分支才直接体现这条 MQA 吸收路径。本文这里推导的是
FlashMLA sparse-prefill caller 需要准备的 576-D MQA 执行表示，而不是声称
reference prefill 已经调用该 kernel。

这条重排为稀疏 kernel 暴露了最重要的复用关系：同一个 selected KV row 在
算法上可以服务全部 128 个主 heads。

这里还要区分算法共享与物理共享。SM90 release kernel 以 64 heads 为一个 CTA，所以 CTA 内可严格确认 64-way row reuse；另一个 CTA 读取同一组 IDs，但跨 CTA 没有共享 SMEM，也不能把 L2 hint 写成必然命中。

### 4.3 FlashMLA：把随机 Rows 收进 SMEM 边界

> **Takeaway.** FlashMLA 不在 global memory 中物化
> $K^{\mathrm{slc}}:[T_Q,\kappa,576]$ 与
> $V^{\mathrm{slc}}:[T_Q,\kappa,512]$；producer 直接按 IDs gather 64 条 rows
> 到 SMEM，consumer 因而只看到规则的 $[64,576]$ QK tile 与
> $[64,512]$ latent PV tile。

有了 $J$、$\widetilde Q$ 和共享 KV，主 Attention 的数学程序很短：

$$
K^{\mathrm{slc}}_{t,r,:} = \widetilde K[J_{t,r},:],
$$

$$
V^{\mathrm{slc}}_{t,r,:} = C^{KV}[J_{t,r},:],
$$

$$
S_{t,h,r} = \widetilde q_{t,h}^{\mathsf T} K^{\mathrm{slc}}_{t,r,:},
$$

$$
P_{t,h,:} = \operatorname{softmax} \left( \mathrm{scale}\cdot S_{t,h,:} \right),
$$

$$
O^C_{t,h,:} = \sum_r P_{t,h,r} V^{\mathrm{slc}}_{t,r,:}.
$$

对一次固定 selection width $\kappa$ 的 kernel 调用，把所有 query positions 重新写成
batched matrix program，完整 logical shapes 是：

$$
K^{\mathrm{slc}}:[T_Q,\kappa,576],\qquad
V^{\mathrm{slc}}:[T_Q,\kappa,512],
$$

$$
\underbrace{\widetilde Q}_{[T_Q,128,576]}
\underbrace{(K^{\mathrm{slc}})^{\mathsf T}}_{[T_Q,576,\kappa]}
\rightarrow
S:[T_Q,128,\kappa]
\xrightarrow{\mathrm{softmax}_{\kappa}}
P:[T_Q,128,\kappa],
$$

$$
\underbrace{P}_{[T_Q,128,\kappa]}
\underbrace{V^{\mathrm{slc}}}_{[T_Q,\kappa,512]}
\rightarrow
O^C:[T_Q,128,512].
$$

这两次 batched matrix multiplication 分别沿 576-D QK feature 与 selected-token
轴 $\kappa$ 收缩。它们定义算法语义，但不要求两个 selected tensors 真正在 HBM
中存在。

问题不在公式，而在 $K^{\mathrm{slc}}$ 应该在哪里存在。

**公开 sparse-prefill API 的边界。** 本文审计的 Hopper 路径是：

```python
flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
)
```

其 release contract 是：

```text
q       : [TQ, HQ, 576]  BF16
kv      : [TK, 1,  576]  BF16
indices : [TQ, 1,  κ]     INT32
output  : [TQ, HQ, 512]   BF16
max     : [TQ, HQ]        FP32
lse     : [TQ, HQ]        FP32, log2-based
```

同时要求：

```text
architecture = SM90
H_KV         = 1
HQ % 64      = 0
κ > 0
κ % 128      = 0
D_QK         = 576
D_V          = 512
```

Q、KV 和 indices 的最后一维必须连续；invalid index 可以是 `-1` 或任何不小于 $T_K$ 的值。这些约束是当前 release kernel 的能力边界，不能泛化成任意 head、任意维度、任意 Top-K。

这里还有一个重要的调用约束：kernel 只检查 index 是否落在
$[0,T_K)$，并不知道每个 query 的绝对位置，也不会在内部重新施加 causal
mask。因此所有有效的 $J_{t,r}$ 都必须已经满足该 query 的因果可见性；这由
上游 Indexer、causal cleanup 与 Top-K pipeline 保证，而不是 FlashMLA 补救。

**为什么外部 gather 是错误的 materialization boundary。** 若先在单独 kernel 中收集：

$$
K^{\mathrm{slc}} : [T_Q,2048,576]_{\mathrm{BF16}},
$$

单个 query 就需要写出：

$$
2048\cdot576\cdot2 = 2.25\ {\rm MiB}.
$$

后续 Attention 还要再读一遍，仅这个中间 tensor 就增加约：

$$
4.5\ {\rm MiB/query}
$$

的 logical global-memory round trip。这个数不是必然发生的物理 HBM bytes：
中间结果的重新读取可能由 L2 服务。它对应的还是最节省的 external-gather
方案，因为

$$
\widetilde K=[C^{KV};K^R],\qquad \widetilde V=C^{KV},
$$

所以只需物化一条 576-D fused row，再让 $V^{\mathrm{slc}}$ 使用其前 512
维；若分别物化 K/V，额外流量只会更高。原始 KV-cache read 是 external 与
fused 两种方案共有的成本，因而没有计入这里的“额外流量”。

FlashMLA 不在 global memory 中创建该 tensor。它只把每轮需要的 64 条离散 row 搬入一个规则的：

$$
KV^{\mathrm{SMEM}}_{\mathrm{tile}} : [64,576]
$$

tile，然后让 Tensor Core consumer 把它当普通二维矩阵使用。

**CTA 与 warpgroup ownership。** SM90 sparse-prefill release 使用：

$$
B_H=64, \qquad B_{\mathrm{kv}}=64, \qquad N_{\mathrm{threads}}=384.
$$

其中 $B_H$ 是一个 CTA 覆盖的 query-head tile 长度，
$B_{\mathrm{kv}}$ 是一次处理的 selected-token tile 长度。

launch grid 为：

$$
\mathrm{grid.x} = T_Q(H_q/64).
$$

一个 CTA 固定拥有：

$$
Q_t^{\mathrm{CTA}} : [B_H,576]=[64,576],
$$

$$
J_t:[\kappa],
$$

$$
O_t^{\mathrm{CTA}} : [B_H,512]=[64,512].
$$

128 个主 heads 因而使用两个 CTA，两者读取同一个 $J_t$。

384 threads 分成三个 128-thread warpgroups：

| warpgroup | 角色 |
|---|---|
| WG0 | consumer：even 64-token chunk；持有输出 $0{:}256$ |
| WG1 | consumer：odd 64-token chunk；持有输出 $256{:}512$ |
| WG2 | producer：读取 indices，将间接 KV rows 搬入 SMEM |

当论文默认配置 $\kappa=2048$ 时：

$$
2048/64=32
$$

个 selected-token chunks 被组织成 16 轮 even/odd paired iterations。

**随机的是 row base，不是 row 内的 576 个元素。** 一条 BF16 KV row 占：

$$
576\cdot2 = 1152\ {\rm B}.
$$

$J_t[r]$ 决定下一条 row 的 base address，相邻 selected rows 可以落在完全不同的 cache line、page 或 HBM partition。但 base 确定后，该 row 的 576 个 BF16 元素在逻辑地址上连续。SM90 producer 实际将它拆成九个连续的 128 B feature slabs 协作搬运；这不是一条 1152 B bulk transaction，也不保证底层 DRAM transactions 全部合并。

所以 producer 面对的不是 576 次彼此无关的 scalar random load，而是：

```text
随机确定 row base
→ 多线程合作搬运连续 row
→ 写入 swizzled SMEM
→ consumer 读取规则 [64,576] tile
```

不规则性被限制在 global-memory hierarchy 到 SMEM 的边界；QK 和 PV 的内部 shape 重新变成规则矩阵。

**两个 consumers 沿两条轴分工。** WG0 与 WG1 都计算完整 QK feature：

$$
[64,576][576,64] \rightarrow [64,64].
$$

它们的 QK 区别在 token 轴：WG0 处理 even chunk，WG1 处理 odd chunk。

PV ownership 则沿 value feature 轴拆开：

$$
V = V_L\oplus V_R,
$$

$$
V_L,V_R : [64,256].
$$

| WG | 本地 probability | 常驻 FP32 accumulator | 对本地/远端 $P$ 的贡献 |
|---|---|---|---|
| WG0 | $P_0$ | $O_L:[64,256]$ | $P_0V_{0L}+P_1V_{1L}$ |
| WG1 | $P_1$ | $O_R:[64,256]$ | $P_1V_{1R}+P_0V_{0R}$ |

两个 warpgroups 各向对方提供一块 probability tile：

$$
P_0,P_1:[64,64]_{\mathrm{BF16}},\qquad
|P_0|=|P_1|=8\ {\rm KiB}.
$$

因此一个 even/odd pair 的双向 useful payload 是 16 KiB，而不是搬动各自约
64 KiB 的 FP32 output accumulator。

online softmax 仍维护 $m,z,O_{\mathrm{acc}}$ 三类状态。新增的同步只在于：WG0
先把历史状态与 even chunk 合并并发布更新后的 max，WG1 再把 odd chunk 纳入，
得到这一 pair 的统一 max 基准；两边据此重标度 probability 与各自的 output
half。两份 partial normalizer $z_0,z_1$ 仍分别累积，直到 epilogue 才跨
warpgroup 相加。因此 even/odd pipeline 改变的是 ownership，没有改变
selected-token softmax 的数学定义。

**Feature-segment reuse pipeline。** 这里不是两块完整 $[64,576]$ tile 做简单
ping-pong。`plan.k[0]` 与 `plan.k[1]` 分别保存当前 pair 的 even 与 odd
64-token chunk；每个 chunk 又沿 feature 轴拆成 $[0:256]$ 与
$[256:576]$ 两段，并各自拥有 ready/free barrier。Producer 对一对 chunks
的搬运顺序是：

```text
even[0:256]
→ odd[256:576]
→ even[256:576]
→ odd[0:256]
→ next pair
```

Consumer 用完某一段便发布对应 free barrier，producer 随即可以用下一 pair 的
数据就地覆盖该段。因此在稳态中仍然形成

```text
compute pair i
        ||
gather pair i+1
```

的重叠，只是流水发生在四个可独立复用的 feature segments 上，而不是两个完整
tile 交换角色。它减少的是 Tensor Core 等待下一批离散 rows 到达的可见
latency；不减少必须读取的 KV bytes，也不改变 selected row addresses 的
随机性。若 L2/TLB/HBM 延迟超过当前 pair 计算可覆盖的窗口，consumer 仍会
停顿。

**共享 selection 如何转化为 locality。** 在一个 CTA 内，每条 576-D KV row 被 64 个 query heads 复用。只计该 row 参与的 QK 与 latent PV，局部算术量约为：

$$
2\cdot64\cdot(576+512) = 139264\ {\rm FLOPs}.
$$

除以 1152 B useful row payload：

$$
\frac{139264}{1152} \approx 121\ {\rm FLOP/B_{useful}}.
$$

这个数字解释了为什么 “任意 token selection” 不必然意味着 consumer 只能做低强度随机访存。算法让 64 个 heads 共享地址，kernel 再把同一 row 变成片上高复用 tile。

但 $121\ \mathrm{FLOP/B_{useful}}$ 只是以逻辑有效载荷为分母的 CTA-local
intensity，不是端到端 roofline，也不是实测的 L2 或 DRAM bytes。它没有计入：

- Indexer 与 Top-K；
- index tensor 的读取；
- Q、输出与 softmax 状态；
- cache-line transaction、L2/TLB miss 与 `L2::256B` prefetch hint；
- 两个 CTA 之间不保证发生的 cache reuse；
- launch、调度与服务固定开销。

**实验：随机 KV rows 还剩多少代价？** 前面的 kernel 拆解给出一个可以直接
检验的现象预测：FlashMLA 没有消灭随机 row address，但在完整 kernel 中，
row order 与 row concentration 的变化可能只表现为较小的 latency 扰动。
下面的 KV-row locality experiment 测量这个响应有多大；它本身不识别是哪一种
infra 机制造成该响应。

我们在一张 NVIDIA H100 PCIe 上冻结未修改的官方 FlashMLA SM90
sparse-prefill kernel（commit `3969f20`），使用：

$$
Q:[128,128,576]_{\mathrm{BF16}},\qquad
KV:[16{,}777{,}216,1,576]_{\mathrm{BF16}},\qquad
J:[128,1,2048]_{\mathrm{INT32}}.
$$

每个 query 拥有一块互不重叠的 128K-row KV arena，因此跨 query 的
selected-row IDs 不会重叠。这排除了某个条件偶然获得更多跨 query 同-row
cache reuse 的混杂因素；但同一 query 的两个 64-head CTAs 仍然请求同一组
selected rows，二者之间仍可能发生 L2 reuse。
每种条件包含 10 个 interleaved campaigns、每个 campaign 20 个 matched
rounds，共 200 次逐调用 CUDA-event 测量；16 个不重复条件共保存 3200 条
observation，并统一使用 eight-slot rotating schedule。官方 34 个
sparse-prefill tests、自建的 16 条件 correctness gate 和 formal validation
均通过。`[MEASURED]`

![FlashMLA sparse-prefill 的 KV-row 顺序与聚集范围](/images/sparse-attention-kv-row-locality.png)

*图 1：纵轴使用 focused scale。左图固定完全相同的 selected-row set，只改变
ascending、descending 与八种 shuffle-seed 顺序；右图改变采样窗口 $W$，
不同 $W$ 会重新采样 window origin 与 exact selected IDs，因此折线只作视觉
引导，不表示同一 selected set 的连续轨迹。误差棒均为 paired campaign-block
bootstrap（5,000 resamples）得到的 pointwise 95% CI，不是所有可能
permutations 或多重比较校正后的总体区间。*

**先只改变访问顺序。** Panel A 对每个 query 和每个 ring slot 固定完全相同的
2048-row selected set，只将它排列成地址升序、地址降序和八种独立随机顺序。

Ascending 的中位 latency 为 $291.680\,\mu s$；descending 为
$290.304\,\mu s$，归一化 ratio 为 $0.9953$，paired 95% CI 为
$[0.9906,1.0046]$。点估计相差 $0.47\%$，但区间包含 1。

八个 shuffle seed 的 ratio 落在 $0.9953$–$0.9990$，seed-level median
为 $0.9976$，IQR 为 $0.0030$。对于这八个预先冻结的排列，每个 shuffle
相对 ascending 的 pointwise timing CI 都跨过 1。因此在这个固定 kernel
point 上，我们没有检测到随机排列相对于地址升序的 latency 惩罚。这些区间
量化的是每个 fixed-permutation condition 的 timing uncertainty，不是所有
可能 permutations 的总体区间；关于 permutation-to-permutation variation，
独立单位只有八个 seeds，每个 seed 下的 200 次 timing 仍是重复测量。

这个 null result 不是因为两组 indices 实际上仍然很接近。地址有序时，相邻 row
距离的中位数为 44，每个 64-index tile 触及的 64-row regions 中位数为 41；
在 `W=128K, shuffle-00` 下，这两个数字分别变成 38,435 和 63，同一
64-row region 内的相邻访问比例也从 $36.49\%$ 降至 $0.046\%$。软件可见
的 locality 已经发生数量级变化，而 latency 仍几乎不动。

**再改变 selected-set 的聚集范围。** Panel B 固定 $\kappa=2048$、
uniform-without-replacement 采样规则和 `shuffle-00` 的 rank-permutation
rule，将 selected rows 分别从宽度为

$$
W\in\{2K,4K,8K,16K,32K,64K,128K\}
$$

的 64-row-aligned window 中采样。这里改变的是 selection distribution 的聚集
程度：不同 $W$ 的 window origin 与 exact selected row IDs 并不相同，因此
它不是“同一组 tokens 只改变 span”的 paired ablation。只有
`W=128K, shuffle-00` 与 Panel A 共享完全相同的 indices corpus，并作为
Panel B baseline。

128K baseline 的中位 latency 是 $291.360\,\mu s$。相对它，2K 和 4K
窗口的 latency 分别低 $0.906\%$ 和 $0.818\%$，即绝对差约
$2.64\,\mu s$ 与 $2.38\,\mu s$；对应 pointwise 95% CI 为

$$
[0.9852,0.9973],\qquad [0.9869,0.9992].
$$

在预先报告的 pointwise 95% intervals 下，只有 2K 与 4K 的区间低于 1；
其余窗口均包含 1。这里同时进行了六个 W-vs-128K 比较，却没有作 family-wise
multiple-testing correction，尤其 4K 的上界 $0.9992$ 已非常接近统计边界。
一个未预注册的 Bonferroni bootstrap sensitivity check 中，2K 的区间上界约
为 $0.9993$，4K 则约为 $1.0018$。因此本文把 2K 视为较清晰的
tight-window 小收益信号，把 4K 降为更弱的 exploratory signal。

与此同时，每个 tile 触及的 region 数随 $W$ 从
28、41、51、57、60、62 增至 63，latency point estimates 却不单调：64K
快于 8K–32K，32K 又略慢于 128K。这些跨窗口关系本身不是额外的 direct
pairwise tests。实验没有建立 locality–latency 单调律、可信阈值或普遍的
显著性结论。

**启动阶段的敏感性。** 原始计时中还存在一个与每轮实验起点严格对齐的
短暂慢启动：10/10 轮实验的第一次计时都达到所属条件中位数的
$1.57$–$1.64\times$，总计 12/3,200 个观测超过各自条件中位数的
$1.2\times$。正式结果遵守预先冻结的纳入规则，没有在看到结果后删除这些
观测；中位数估计与配对随机化降低了少数长尾点的影响。

作为补充的事后敏感性检查，删除每轮实验的
`matched_round=0` 后，2K/4K ratios 从 $0.9909/0.9918$ 变为
$0.9939/0.9943$，对应收益从约 $0.91\%/0.82\%$ 缩小到
$0.61\%/0.57\%$；重新计算的 pointwise 95% intervals 仍低于 1。方向没有
翻转，但精确效应量对实验所处阶段有一定敏感性，因此这里只把它解释为
亚百分比的小信号，而不是稳定的硬件常数。

Profiler 只用于解释这个 clean-timing 结果。相对于
`W=128K, shuffle-00`，单次 $W=2K$ capture 的 NCU DRAM read 少
$2.17\%$，device-read sectors 少 $1.64\%$，NCU/Nsys duration 分别短约
$0.96\%$ 和 $1.31\%$。但 L2 hit rate、long-scoreboard stall 与
Tensor-pipe activity 都没有随 $W$ 形成一致的单调链条。四个代表点的
GMMA 指令数完全相同，achieved occupancy 也都在
$18.37\%$–$18.38\%$。这确认了 dominant Tensor Core work 与静态执行
形状受到控制，但不表示地址生成、cache transaction 或 stall behavior 相同。
每个条件只有一个 NCU/Nsys kernel instance，而且 NCU 使用 replay、不清 cache、
不锁频；这些 counters 只能作为补充诊断，不能给出带方差的因果解释。

因此，这组 locality experiment 最稳妥的结论不是“随机访问已经免费”，而是：

> **在冻结的 H100/SM90 sparse-prefill operating point 上，对于八个预先测试的
> 随机排列，我们没有检测到相对地址升序的 latency penalty，所有 point
> estimates 与 ascending 相差不超过 $0.47\%$。W-sweep 的 latency 不随
> locality 单调变化；2K 给出较清晰的亚百分比小收益信号，4K 则是更接近统计
> 边界的 pointwise signal。**

这个结果与前面的 co-design 解释相符：一个 CTA 将每条 selected row 复用于
64 个 heads，producer 把离散 global-memory rows 重排成规则 SMEM tile，
feature-segment pipeline 再尝试用 QK、softmax 与 PV 覆盖下一批 row 的到达
时间。但本实验没有关闭 64-head reuse 或 pipeline，因此只能说结果“与这些机制
隐藏了大部分 locality 差异相一致”，不能声称实验已经证明是哪一种机制导致了
本实验观察到的小响应。

实验边界同样重要：indices 是 uniform-without-replacement 的 synthetic
stimulus，private arenas 还刻意消除了真实 workload 可能具有的热门 token 和
跨 query reuse。结果只覆盖 $T_Q=128$、$\kappa=2048$、BF16 576-D rows、
$D_V=512$ 和一张 H100 PCIe；它不是完整 DeepSeek-V3.2、production DSA、
真实 Indexer trace，也不是 sparse decode 测量。Panel A 的统计结论限于八个
冻结的 permutation seeds；Panel B 比较的则是不同 $W$ 下重新采样的
synthetic corpora。实验还使用
$1/\sqrt{576}$ 作为冻结的 synthetic kernel stimulus，而不是模型的
192-D-plus-YaRN score scale。

因此这个受控实验只能证明 FlashMLA 在当前 operating point 上对 row locality
表现出较低敏感性；它不能保证 production DSA 一定得到相同结果。验证外部
有效性还需要保存真实 Indexer 输出的 indices trace，原样 replay 到同一 kernel，
并比较其 adjacent distance、regions-per-tile、跨 query overlap 与 latency
是否落在本实验覆盖的范围内。

<!-- Evidence:
FlashMLA-LocalityExp run e3-h100-20260805-formal2;
reports/e3_h100_technical_report.md;
results/summaries/e3-h100-20260805-formal2.formal-validation.json.
-->

**prefill 与 decode 分为两个 kernel。** 上面的微架构与 locality
experiment 都对应 BF16、非分页的 SM90 sparse-prefill。作为补充测量的 H100
single-query decode 路径使用的是另一条公开实现：paged FP8 sparse-decode。

在固定$B=1$ 实验形状下，它使用：

```text
q                : [1, 1, 128, 576] BF16
main KV cache    : [num_blocks, 64, 1, 656] mixed-byte packing
physical indices : [1, 1, 2048] INT32
page size        : 64 tokens
latent D_V       : 512
```

其中每个 token 的 656-byte cache row 由

$$
512\ \mathrm{B\ FP8\ NoPE}
+16\ \mathrm{B\ FP32\ scales}
+128\ \mathrm{B\ BF16\ RoPE}
=656\ \mathrm{B}
$$

组成，并非 656 bytes 全部使用 FP8。`physical indices` 也不是 Indexer 输出的
逻辑 token IDs，而是经 page table 改写后的 paged-cache token offsets。该路径
的 Top-K multiple-of-64 约束、split-KV scheduler 与 combine kernel 都不同于
本节的 sparse-prefill contract。二者实现同一 selected-token Attention 语义的
不同 operating point，不能共用 cache shape、metadata 成本或单 kernel 结论。

**在 Blackwell 上，FlashMLA 进一步用 TMA 处理离散行。** 下述结构对应所核对的 FlashMLA commit
[`9241ae3`](https://github.com/deepseek-ai/FlashMLA/tree/9241ae3ef9bac614dd25e45e507e089f888280e0)
；在这个版本中，$H_q=128,\kappa=2048,D_{QK}=576$ 的 public dispatch 精确落到
SM100 regular `head128_k576`，而不是 `head64`：`head64` prefill 要求完整
$H_q=64$，small-topk head128 又只支持 $D_{QK}=512$。

这与前文形成一条直接的架构演进。Hopper/SM90 将一个 128-head query 映射成
两个拥有独立 SMEM 的 64-head CTAs；Blackwell/SM100 则使用

$$
B_H=128,\qquad B_{\mathrm{kv}}=128,
$$

并以

$$
\mathrm{grid.x}=2T_Q,\qquad \mathrm{clusterDim.x}=2
$$

为每个 query 组织一个 two-CTA cluster。每个 CTA 仍处理 64 个 query heads，
但 K producer 改用 `TMA tile::gather4` 的 `cta_group::2` 形式。

一条 `gather4` 指令描述四条任意 row 的 64-D BF16 feature slab，因此 useful
payload 为

$$
4\times64\times2=512\ {\rm B}.
$$

在每个 128-token chunk 中，两个 CTA 各负责 64 条 selected K rows。于是每个
CTA 的一个 $[64,64]$ K slab 需要 $64/4=16$ 次 `gather4`；完整
576-D K 路径共有九个 64-D slabs，最后一个正是 RoPE 部分。与 standalone
`head64_k576` 不同，这条实际 head128 路径不会把 RoPE tail 改走专用
`cp.async`。

V 使用另一组 producer warps：两个 CTA 都覆盖 128 条 selected rows，但分别
gather 256 个 value dimensions，合起来形成 $[128,512]$ latent V tile。
因此 cluster 在 token 轴协作准备 K、在 value-feature 轴拆分 V，正好对应后续
two-CTA QK/PV ownership。

`gather4` 将四行间接寻址、bulk-copy issue 与 completion tracking 下沉给 TMA，
并能直接落入 swizzled SMEM；源码可以直接确认软件逐行地址生成与细粒度 copy
指令减少，但总同步开销和实际 latency 的变化仍需 benchmark。四条 row 仍可能
对应完全不同的 global-memory addresses，指令不保证把它们合成连续的物理
DRAM transactions。这不是本文 H100/SM90 实验所能验证的路径。

### 4.4 隐藏的二次项：稀疏主核不等于线性系统

> **Takeaway.** FlashMLA 将昂贵的主 Attention 限制在固定 Top-$\kappa$，但
> Lightning Indexer 仍为每个 query 计算全部历史 candidates，随后执行 exact
> Top-$\kappa$；因此 sparse core 是 $\Theta(N\kappa)$，完整 causal prefill
> 仍保留低常数的 $\Theta(N^2)$ Discovery term。

FlashMLA 只访问 selected 2048 tokens，但 DSA 必须先发现它们。只要 routing score 仍依赖每个 query–candidate pair，Indexer 就保留了一次全历史扫描。

下面的 $\mathcal W$ 只计 dominant dot/QK/PV，并省略所有式子共同的 FMA
factor 2。它是 shape-derived useful-MAC count，不是不同 dtype 与 kernel 的
等价时间单位。

**单个成熟 query。** 面对长度 $L$ 的历史，Indexer 成本为：
$$
\mathcal W_{\mathrm{Indexer}}(L) = H_I D_I L = 64\cdot128\cdot L = 8192L.
$$

selected main MQA core 为：

$$
\mathcal W_{\mathrm{sparse}}(L) = H(D_{QK}^{\mathrm{MQA}}+D_C) \min(L,\kappa),
$$

$$
\mathcal W_{\mathrm{sparse}}(L) = 128(576+512) \min(L,2048),
$$

$$
\mathcal W_{\mathrm{sparse}}(L) = 139264 \min(L,2048).
$$

所以：

$$
\mathcal W_{\mathrm{DSA}}(L) = 8192L + 139264\min(L,2048).
$$

第一项随历史持续增长；第二项在 $L\ge2048$ 后饱和。

**完整 causal prefill。** 现在用 $N$ 表示完整 causal sequence 的长度，对
query positions $t=1,\ldots,N$ 求和：
$$
\mathcal W_{\mathrm{Indexer,full}}(N)
= \sum_{t=1}^{N}8192t
= 8192 \frac{N(N+1)}{2}
= \Theta(N^2).
$$

sparse core 的精确求和是：

$$
\mathcal W_{\mathrm{sparse,full}}(N)
= 139264 \sum_{t=1}^{N} \min(t,\kappa).
$$

当 $N\ge\kappa$：

$$
\sum_{t=1}^{N} \min(t,\kappa)
= \frac{\kappa(\kappa+1)}{2} + (N-\kappa)\kappa.
$$

因此：

$$
\mathcal W_{\mathrm{sparse,full}}(N)
= \Theta(N\kappa) \qquad (N\ge\kappa).
$$

固定$\kappa=2048$ 时：

$$
\boxed{
\mathcal W_{\mathrm{DSA,full}}(N)
= \Theta(N^2)+\Theta(N\kappa)
= \Theta(N^2)
}.
$$

DSA 没有把 quadratic attention 变成 asymptotically linear attention。它把：

$$
\text{高维、128-head 的 quadratic 主 Attention}
$$

替换为：

$$
\text{低维 FP8 quadratic routing} + \text{高维但固定 Top-K 的主 Attention}.
$$

这是很有价值的常数重构，不是渐近阶数改变。

**它替换了多大的 quadratic 系数。** 用原始 192-D MHA score 和 128-D MHA value 作 prefill-style baseline：

$$
\mathcal W_{\mathrm{dense,MHA}}(L) = H(192+128)L = 40960L.
$$

Indexer 的二次项系数是其：

$$
\frac{8192}{40960} = \frac15.
$$

若比较 single-query dense MQA decode，完整扫描 576-D key 与 512-D latent value：

$$
\mathcal W_{\mathrm{dense,MQA}}(L) = H(576+512)L = 139264L.
$$

Indexer 的线性扫描系数比它小：

$$
\frac{139264}{8192} = 17.
$$

这些比例还隐含一个带宽变化：全历史扫描中占主导的 per-candidate KV payload
只需读取共享的 128-D FP8 Indexer key 与 scale；Indexer query、head weights 与
score carrier 写入仍然存在。完整 576-D 主 KV 则只为 selected tokens 读取。

**必须分开 endpoint 与 full-causal 口径。** endpoint model 只看一个面对完整历史 $L$ 的 query；full-causal model 对序列中全部 query positions 求和。

令 $\kappa=2048$ 且 $L\ge\kappa$。endpoint 的 prefill-style baseline 与 DSA 分别是：

$$
\mathcal W_{\mathrm{dense,endpoint}}(L) = 40960L,
$$

$$
\mathcal W_{\mathrm{DSA,endpoint}}(L) = 8192L + 139264\kappa.
$$

full-causal baseline 为：

$$
\mathcal W_{\mathrm{dense,full}}(N) = 40960 \frac{N(N+1)}{2}.
$$

DSA full-causal 为：

$$
\mathcal W_{\mathrm{DSA,full}}(N)
= 8192 \frac{N(N+1)}{2}
+ 139264 \left[
\frac{\kappa(\kappa+1)}{2}+(N-\kappa)\kappa
\right].
$$

由这些式子得到的几个解析交叉点是：

| 口径 | operation-count 关系 | tokens |
|---|---|---:|
| endpoint，prefill-style MHA baseline | DSA 与 dense 相等 | 8,704 |
| full causal prefill，MHA baseline | DSA 与 dense 相等 | 16,315 |
| single-query dense MQA decode | DSA 与 dense 相等 | 2,176 |
| endpoint，DSA 内部 | Indexer 与 sparse core 相等 | 34,816 |
| full causal，DSA 内部 | cumulative Indexer 与 cumulative sparse core 相等 | $\approx68{,}592$ |

这些值只回答：按给定 shape 计数，两项算术在哪个长度交叉。它们没有包含：

- projection、RoPE、Hadamard 与量化；
- FP8、BF16 与 Tensor Core 吞吐差异；
- FP32 logits materialization；
- exact Top-K 与 index rewrite；
- FlashMLA scheduler、split-KV 与 combine；
- 通信、launch、cache 和服务固定开销。

所以 8,704、16,315、2,176、34,816 或 68,592 都不能直接写成 latency
crossover，更不能写成 speedup 保证。后面的 Discovery scaling experiment
分别采用相同的 endpoint 与
full-causal operating-point 口径，但测量对象是包含 scorer、Top-K 与 index
transform 的公开 discovery chain；因此 34,816 与约 68,592 只提供算术量级
参考，不能预测 $[32K,64K]$ 的实测 bracket。前三种 dense baseline 关系
则只说明 DSA 替换了多大的 useful-MAC 系数。

**公开实现仍会物化完整的 routing-score 矩阵。** 如果把 full prefill 一次性写成

$$
I:[L,L]_{\mathrm{FP32}},
$$

那么在 $L=128\mathrm{Ki}$ 时，仅最终 routing logits 的逻辑 payload 就是

$$
(131072)^2\cdot4 = 64\ {\rm GiB}.
$$

这是 one-shot materialization 的 shape-derived logical payload，不是后面
Discovery scaling experiment 的实际峰值，也不是该路径的完整显存需求。
公开 prefill scorer 支持 query chunk。令 chunk size 为 $C=4096$，最终处理
长度 $N=mC$，第 $j$ 个 chunk 的右端点为 $e_j=jC$。忽略
`BLOCK_Q=2`、`BLOCK_KV=256` 带来的有界 tile rounding，真正有 causal
语义的 scorer pairs 为

$$
P_{\mathrm{causal}}(N)
=\sum_{t=1}^{N}t
=\frac{N(N+1)}2.
$$

但交给独立 Top-K 的是每个 chunk 的矩形 carrier：

$$
E_{\mathrm{carrier}}(N)
=
\sum_{j=1}^{m}C(jC)
=
\frac{N(N+C)}2.
$$

二者之差是每个 chunk 内必须清理为 $-\infty$ 的 future triangles：

$$
E_{\mathrm{future}}(N)
=E_{\mathrm{carrier}}(N)-P_{\mathrm{causal}}(N)
=\frac{N(C-1)}2.
$$

这里的 $E_{\mathrm{future}}$ 只计算真正写成 $-\infty$ 的 future elements，
不能直接代表整个 cleanup kernel 的 issued work。公开 cleanup kernel 仍按
$B_{\mathrm{clean}}=8192$ 遍历每一行的完整 carrier。记所有 query rows
累计访问的 cleanup blocks 数为 $\mathcal B_{\mathrm{cleanup}}(N)$，则

$$
\mathcal B_{\mathrm{cleanup}}(N)
=\sum_{j=1}^{m}C
\left\lceil\frac{jC}{B_{\mathrm{clean}}}\right\rceil
=\Theta\!\left(\frac{N^2}{B_{\mathrm{clean}}}\right).
$$

因此固定 $C$ 时，无效值写入量是
$\Theta(NC)=\Theta(N)$，但 cleanup kernel 的 traversal 并不是线性的。
公开 API 将 scorer 与 cleanup 合并计时，后面的实验也保持这一边界。

因此：

$$
\begin{aligned}
\text{causal scorer useful pairs/MACs} &:\Theta(N^2),\\
\text{Top-K carrier scan} &:\Theta(N^2),\\
\text{future cleanup stores} &:\Theta(NC)=\Theta(N),\\
\text{cleanup block traversal} &:\Theta(N^2/B_{\mathrm{clean}}),\\
\text{fixed-}\kappa\text{ FlashMLA} &:\Theta(N\kappa)=\Theta(N).
\end{aligned}
$$

在 2M 的最后一个 chunk 中，公开链真正物化的 logical view 是

$$
I_j:[4096,2097152]_{\mathrm{FP32}}=32\ {\rm GiB},
$$

其 DeepGEMM backing 因 256-token stride padding 再多 4 MiB。从 4K replay
到 2M，所有矩形 carriers 的累计 logical write volume 约为 8.016 TiB，但它们
不需要同时保持为 live tensors；allocator 仍可能保留已经申请的 reservation。
Chunking 把单次 live carrier 从 square tensor 降到矩形 tile，却没有改变 scorer
与 Top-K carrier 的二次累计增长。

**exact Top-K 的条件性下界。** 若把 query-dependent routing score 视为没有额外结构的 oracle，也没有 candidate-specific bound 能够安全排除未检查候选，那么 exact global Top-$\kappa$ 在最坏情况下必须检查全部 $L$ 个 candidates。

否则，对任何被跳过的候选 $s$，都可以构造一个保持已检查 scores 不变、但令 $I_{t,s}$ 大于当前第 $\kappa$ 大值的输入。selector 因而无法证明结果 exact。

在这些明确前提下：

$$
\text{single query} : \Omega(L) \ \text{score evaluations},
$$

$$
\text{full causal prefill} : \Omega(L^2) \ \text{score evaluations}.
$$

Chunking、streaming Top-K 和 kernel fusion 可以减少 peak memory、HBM 往返与
launch，但不会自动减少必须判断的 candidate 数量。要改变渐近项，需要允许
approximate retrieval、引入具有可证明安全排除保证的层次化候选结构，或让
相邻 queries 在同样的 exactness 保证下复用已经排除的候选。

**实验：Main Attention 已经稀疏，Discovery 的二次项何时显形？** 上面的
operation count 只描述算法结构；GPU latency 还取决于 dtype、kernel efficiency、
中间张量以及内存流量。下面的实验因而只问一个更窄、也可直接测量的问题：

> 在冻结的公开实现链中，随着完整 causal prefill 长度 $N$ 增长，Discovery
> 何时超过 fixed-$\kappa$ 的 FlashMLA sparse main attention？

我们没有实现或测量 Indexer–TopK fused kernel。被测对象是下面这条 released
**unfused** chain：

```text
DeepGEMM FP8 scorer
→ causal cleanup
→ FP32 score carrier
→ PyTorch exact Top-K
→ invalid-index padding / INT32 cast
→ FlashMLA sparse prefill
```

记

$$
T_{\mathrm{Discovery}}
=T_{\mathrm{scorer+cleanup}}
+T_{\mathrm{TopK}}
+T_{\mathrm{pad/cast}},
\qquad
T_{\mathrm{SparseMain}}=T_{\mathrm{FlashMLA}}.
$$

对同一个 query chunk，FlashMLA 必须等 Top-K IDs 生成后才能启动，因此这里的
Discovery 不是可以和该 chunk 的 main attention 任意重叠掉的旁路成本。

实验在单张 H100 PCIe（SM90）上固定
$B=1,C=4096,\kappa=2048$。2 次 warmup replay 后，Direct full-chain total
使用 20 次**重复** clean replay 的 start-to-checkpoint CUDA event；阶段归因
来自另外 5 次 instrumented replay，并在每个 trial 内累加所有 chunk events。
二者是不同的测量轨迹，下面不会用阶段中位数相加冒充 direct total。

这个 workload 是 shape-controlled synthetic chain：同一份 4K Indexer query、
weights 和 main query 被重复 replay，直到 2M 时一共处理 512 个 chunks；
selection 也不来自真实模型 trace。因此它控制了 shape 和执行路径，却没有复现
真实语义输入的 score 分布。下面的 crossover 只适用于这条冻结的公开组件链，
不是 production DeepSeek DSA 的性能预测。

Extended campaign 的 CUDA-event 中位数为：

| 已处理长度 $N$ | Discovery cumulative | FlashMLA sparse prefill | Direct full-chain total | Discovery / FlashMLA |
|---:|---:|---:|---:|---:|
| 32K | 0.0307 s | 0.0422 s | 0.0728 s | $0.73\times$ |
| 64K | 0.1135 s | 0.0896 s | 0.2026 s | $1.27\times$ |
| 128K | 0.4255 s | 0.1861 s | 0.6135 s | $2.29\times$ |
| 256K | 1.670 s | 0.394 s | 2.066 s | $4.23\times$ |
| 512K | 6.607 s | 0.813 s | 7.421 s | $8.13\times$ |
| 1M | 27.454 s | 1.655 s | 29.098 s | $16.59\times$ |
| 2M | 113.149 s | 3.367 s | 116.490 s | $33.60\times$ |

其中 Discovery 与 FlashMLA 两列来自 5 次 instrumented replay 的同-trial
chunk-event sums；Direct total 来自另行采集的 20 次重复 clean replay。
由于测量轨迹不同，不应把前两列的阶段中位数相加，或要求其和精确等于
Direct total。

64K 是第一个预先指定的 discovery-dominant checkpoint。严格结论只能写成

$$
\boxed{\text{prespecified checkpoint bracket}=[32K,64K]},
$$

而不能把 64K 插值成连续、精确的 crossover。`[MEASURED]`

为了区分有限区间上的缩放形状与单点波动，我们还在 5 条 paired attribution
trajectories 内计算相邻 doubling 的局部有效指数：

$$
\alpha_s(N)=
\operatorname{median}_{r=1,\ldots,5}
\log_2\frac{T_{s,r}(2N)}{T_{s,r}(N)}.
$$

| 区间 | $\alpha_{\mathrm{Discovery}}$ | $\alpha_{\mathrm{SparseMain}}$ |
|---:|---:|---:|
| 128K→256K | 1.972 | 1.083 |
| 256K→512K | 1.984 | 1.045 |
| 512K→1M | 2.055 | 1.025 |
| 1M→2M | 2.043 | 1.025 |

从 256K 到 2M，Discovery 增长 $67.75\times$，对应两端点有效指数
$2.027$；FlashMLA sparse prefill 增长 $8.54\times$，对应
$1.031$。这些指数只描述本次五条 fixed-input timing trajectories。结果与
“all-candidate discovery 近二次、fixed-$\kappa$ sparse main 近线性”的
有限范围预测一致，但它们不是 OLS 拟合、模型输入分布上的总体估计，更不是对
渐近复杂度的实验证明。

![Causal-prefill cumulative scaling 与 Discovery dominance](/images/sparse-attention-discovery-scaling.png)

*图 2：左图的 Discovery 与 FlashMLA 曲线来自 5 次 instrumented replay 的
同-trial cumulative chunk-event sums，不是 20 次 clean replay 的 direct total；
右图绘制二者之比。所有 checkpoint 都是预先指定的离散测量点，连线只作视觉
引导；阴影表示 $[32K,64K]$ 的离散 crossover bracket，不是连续插值或
置信区间。*

**4M 是容量删失点，而不是一个 latency 数据点。** 在冻结的
$C=4096,\kappa=2048$ 和当前 unfused PyTorch Top-K 路径下，preflight 给出的
严格下界包括 64.004 GiB DeepGEMM carrier backing、64 GiB Top-K contiguous
copy，以及 5.626 GiB resident tensors，总计至少

$$
133.630\ {\rm GiB}>79.180\ {\rm GiB}.
$$

因此实验在实际分配前将 4M 标记为 capacity-censored，没有伪造 latency，也
没有把它称作实测 OOM。这个结论不等于“DSA 的最大上下文是 2M”：减小 $C$、
改变 Top-K 实现或做 scorer–selector fusion，都可能改变容量边界。

**长序列上的 Nsys 数据用于解释 kernel 结构，而不是替代正式计时。** 在一个 target
chunk 上，PyTorch Top-K 的一次语义 contiguous copy 会被 TensorIterator
拆成约 1 GiB 的 shards：512K 为 8 个 launches，2M 为 32 个 launches。除去
这些 copy shards，Top-K 的 kernel-family sequence 与 launch count 都保持为
15，但 radix、scan 和 gather 等 kernel 的 grid 会随 $N$ 增长，不能笼统说
“Top-K topology 完全不变”。

同一诊断中，target-chunk attention kernel-active time 从 512K 的
$6.465\ {\rm ms}$ 变为 2M 的 $6.746\ {\rm ms}$，而
discovery/attention 从 $16.50\times$ 增至 $65.32\times$。这些 Nsys
duration 只佐证同一机制的增长方向，不能和 cumulative CUDA-event latency
混成一条曲线。

**128K 的 NCU 数据进一步解释了这些流量来自哪里。** 对 128K 最后
一个 query chunk 的 2 GiB FP32 carrier，NCU 观察到 scorer 写出约
1.958 GiB DRAM；随后 Top-K 读取 12.064 GiB，即 carrier 的
$6.03\times$，并写出 2.180 GiB。这些流量来自一次 contiguous copy、四轮
radix threshold，以及 count、scan、gather 等步骤，不是“一次读 logits、一颗
Top-K kernel”。该 target chunk 的 Top-K stage 包含 17 个 CUDA kernels；
完整 4K→128K replay 中共有 513 个 Top-K kernels。

但 launch 数量本身不是主要损失。三次 Nsys full replay 中，kernel span 内
未被 kernel 覆盖的 GPU gap 只有 $0.283\%$–$0.289\%$。因此 fusion 真正
可能消除的是

$$
\boxed{
\text{FP32 logits write}
+
\text{Top-K repeated global-memory scans}
},
$$

而不是把 CPU launch 数直接换算成 GPU speedup。

**已测量的事实到此为止。** 下面是源码与资源约束给出的设计推论，而不是
fused performance result。当前 DeepGEMM mapping 让一个 persistent CTA 拥有
两个 query rows，并在 CTA 内遍历它们的全部 KV tiles。若保持这种 ownership，
exact Top-2048 可以被组织为 CTA 内跨 tile 的在线状态，不天然需要跨 CTA
global merge。仅 score 与 ID payload 就是

$$
2\ \text{queries}\times2048\times
(\text{FP32 score}+\text{INT32 id})
=32\ {\rm KiB}.
$$

但这 32 KiB 不是免费的。当前 scorer 已使用 640 threads、96 registers/thread
与 152,228 B dynamic shared memory，并按一 CTA/SM 运行。H100 opt-in SMEM
上限为 232,448 B；扣除 candidate payload 与约 4 KiB 的双缓冲 score tile 后，
名义空间只剩约 42 KiB 给 merge scratch、barriers 与对齐。把 selector consumer
塞进主循环还可能产生 register spill、SMEM competition 或 producer
backpressure，反过来伤害原有 WGMMA/TMA scorer。

所以这组测量支持的是一个设计机会，而不是一个已经测得的加速：

> **scorer–TopK fusion 有机会消除完整 logits 的 HBM round trip，但不会减少
> exact selector 必须判断的 candidate 数，也尚未证明其片上维护成本低于被
> 消除的流量。**

我们还单独测量了 paged decode 在生成一个新 token 时的端点开销，并将历史
长度扩展到 4M：256K–1M 的 discovery/decode ratio 为
$5.21\times$–$5.23\times$，随后在 2M 与 4M 变成
$12.79\times$ 与 $17.88\times$；4M 的 direct full-chain total 为
$0.675808\ {\rm ms}$。但这组计时并不平稳：trial 0–4 与 5–99
之间存在明显的 event-latency regime shift，而 sparse-decode event median
在 1M→2M 又从约 $0.073\ {\rm ms}$ 降到 $0.036\ {\rm ms}$。

4M 与旧 128K profile 中观察到的 attention launch signature——kernel identity、
grid、block 与 launch count——相同，所以已有证据没有把这个台阶解释为一次
可见的 kernel dispatch 切换；它仍不能排除 kernel 内部分支、cache、clock 或
其他 runtime state。因此，这组 decode 数据只用于判断各阶段开销的方向，
不把 $17.88\times$ 或任一绝对 latency 当成稳定硬件常数，也不把 paged FP8
decode 与上面的 BF16 causal-prefill 曲线拼接。

<!-- Evidence:
Primary long-scale campaign: DSA-KernelExp run e2-h100-20260809-extended1;
reports/e2_h100_extended_report.md;
results/summaries/e2-h100-20260809-extended1.formal-validation.json;
profiles/capacity/e2-h100-20260809-extended1.panel-b-capacity-preflight.json;
profiles/unfused/e2-h100-20260809-extended1/extended-targets/summary/extended_nsight_summary.json;
profiles/unfused/e2-h100-20260809-extended1/extended-targets/summary/extended-nsight-validation.json.
128K NCU mechanism anchor only: run e2-h100-20260803-formal1;
reports/prefill_indexer_topk_fusion_nsight_report.md;
results/summaries/e2-h100-20260803-formal1.formal-validation.json.
-->

### 4.5 DSA 把主 Attention 变稀疏，也把瓶颈推向 Discovery

DSA 不是“先 Top-K，再调用普通 Attention”。完整的 co-design 链是：

$$
\boxed{ \begin{aligned} &\text{共享、低维、FP8 Indexer 扫描全部历史}\\ &\rightarrow \text{每个 query 的 exact Top-K token IDs}\\ &\rightarrow \text{128 个主 heads 共享 selection set}\\ &\rightarrow \text{CTA 内融合离散 row gather}\\ &\rightarrow \text{SMEM 中恢复规则 }[64,576]\text{ tile}\\ &\rightarrow \text{Tensor Core QK、online softmax 与 latent PV}. \end{aligned} }
$$

算法负责暴露跨 head 的 KV row reuse；kernel 负责把地址不规则性限制在 HBM→SMEM；pipeline 负责在这个边界内重叠 load、QK、softmax 与 PV。

这条链解释了 DSA 为什么能在长上下文服务中显著降低主 Attention 的算术和主 KV 带宽。同样重要的是它的反面结论：

$$
\boxed{
\text{DSA 降低了 quadratic term 的维度与常数，}
\quad
\text{但 all-candidate Indexer + exact Top-K 仍使 full prefill 保持 }
\Theta(N^2).
}
$$

这两点并不矛盾。前者说明 algorithm–infra co-design 怎样把任意 token sparsity 变成 GPU 可执行的规则 tile；后者说明 Top-K discovery 为什么仍是下一阶段长上下文系统必须面对的边界。

**这里需要区分推导、源码观察与实测结果。** 语义与训练描述来自 DeepSeek-V3.2 论文及公开 inference
reference；FP8 scorer 的 materialization 边界来自 DeepGEMM release；CTA、
warpgroup 和 sparse-prefill API 来自 FlashMLA SM90 release。34,816 与约
68,592 tokens 是 useful-MAC analytical references；`[32K,64K]`、2M 长点
以及 4M 的容量限制来自 causal-prefill Discovery 规模实验；
128K NCU 测量用于解释 carrier/Top-K 的具体流量。近零的局部性响应
来自另一组 H100 BF16 sparse-prefill 微基准。Indexer–TopK fusion
与 Blackwell `gather4`
都没有本文的性能测量，前者是受源码和 profiler 约束的设计机会，后者只绑定
所注明的审计 commit。

### 4.6 回到开头：两条 co-design 路线共同揭示了什么

> **Takeaway.** 算法决定“哪些数据值得访问”以及“谁可以共享一次访问”，infra
> 决定这些访问能否被一个 program 拥有、重排和归约。二者的接口是 shape、
> ownership 与 reuse，而不是一个抽象的 sparsity ratio。

把 NSA 与 DSA 放回同一张表，差别就不再只是 block sparse 与 token sparse：

| 问题 | NSA | DSA |
|---|---|---|
| 全局 control plane | $T/d$ 分辨率的 compression Attention | 全 token 分辨率的 128-D FP8 Indexer |
| 最终选择单位 | 连续 64-token block | 任意单 token |
| 算法共享轴 | 一个 KV group 的 16 个 query heads | 128 个 main heads |
| 规则性出现位置 | HBM load 之前 | 离散 rows 搬入 SMEM 之后 |
| 一个 program 内的 load reuse | 16 heads | SM90 head64 CTA 内 64 heads |
| 主要 hidden cost | compression path 与 $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ 的 many-to-one reduction | exact all-history scan 与 Top-K |

二者都有“较便宜的全局扫描 + 较昂贵的 sparse data plane”。若只保留 token
与 feature 量级，NSA 的 compressed sequence 长度约为

$$
T_c\approx \frac{T}{d},
$$

所以 compression Attention 的 score 数仍约为

$$
T\cdot T_c
=
\Theta\!\left(\frac{T^2}{d}\right).
$$

它一方面产生 coarse global-context output，另一方面复用 probability 做
selection routing。DSA 的 Indexer 是 routing-only，但保持 full token
resolution，因此完整 causal sequence 的候选分数仍为

$$
\sum_{t=1}^{T}t
=
\Theta(T^2).
$$

在论文默认把压缩步长 $d$ 与 selection width $\kappa$ 视为固定配置时，
两种方法都没有把“发现全局相关位置”变成渐近线性；它们只是把发现过程搬到了
更便宜的表示上：NSA 降低 token resolution，DSA 降低 feature dimension
并使用 FP8。真正不同的是规则性在什么时候出现：

$$
\boxed{
\begin{aligned}
\text{NSA:}\quad&
\text{低分辨率扫描}
\rightarrow\text{连续 block IDs}
\rightarrow\text{load 前已有规则性};\\
\text{DSA:}\quad&
\text{低维全分辨率扫描}
\rightarrow\text{离散 token IDs}
\rightarrow\text{load 后重建规则性}.
\end{aligned}}
$$

这里还要区分两个容易混淆的 reuse：

- **semantic fanout**：一个 selection ID 在算法语义上代表多少 logit pairs；
- **physical load reuse**：一条 KV row 进入某个 CTA/program 后，实际服务多少
  个 heads 或 query rows。

对于一个 query，NSA 的一个 block ID 代表

$$
64\ \text{tokens}\times16\ \text{heads}=1024
$$

个 logit pairs，而同一 KV row 在一个 program 内被 16 heads 使用。DSA 的一个
token ID 在算法上由 128 main heads 共享，但公开的 SM90 sparse-prefill 路径
以两个 CTA 分别处理 64 heads；shared memory 不能跨 CTA 共享，所以单 CTA
内可保证的 physical load reuse 是 64，而不是 128。算法共享是 kernel 复用的
必要条件，却不自动等于一次 HBM transaction 的最终复用次数。

同样，pipeline 也有清晰边界。它能把

```text
metadata / indirect address
        ↓
producer or load stage
        ↓
regular on-chip tile
        ↓
dense MMA consumer
```

重叠起来，却不能消除 L2 miss、TLB/page locality、HBM transaction 或流水线
启动与排空。把 irregularity 隔离在 producer，是为了不让间接地址继续污染
QK、softmax 和 PV；它不是把随机 HBM rows 变成了连续 rows。

因此，评估 sparse Attention 时至少要把以下项目放进同一本账：

$$
\boxed{
\text{selector/scorer}
+\text{routing or Top-K}
+\text{metadata}
+\text{gather}
+\text{sparse core}
+\text{backward reduction}.
}
$$

只 benchmark 最后的 sparse core，会恰好把最难扩展的部分移出图表。只解释
forward skip 了多少 pairs，却不给 $dQ,dK,dV$ 找到稳定 owner，也还不是一个
训练可执行的 primitive。

NSA 与 DSA 最终代表两条互补而不等价的路线：

$$
\boxed{
\begin{aligned}
\text{NSA:}\quad&
\text{先让算法产生规则 block，再让 kernel 直接消费它；}\\
\text{DSA:}\quad&
\text{保留 token-level 自由，再由共享 selection 与专用 kernel 恢复规则 tile。}
\end{aligned}}
$$

这条研究链留下的真正开放问题不是“下一种 sparse mask 应该长什么样”，而是：

> **能否避免全量 $\Theta(L^2)$ Discovery，同时仍向 kernel 暴露足够共享、规则且可归约的
> sparse workload？**

层次化 routing、近似检索、跨 query 复用候选集，都可能打破 exact
all-candidate scan 的前提；但它们也会同时改变模型质量、动态性与 execution
contract。下一步真正值得研究的，是这三者之间可验证的边界，而不是孤立地再降
一次 nominal sparsity。

### 资料与证据

- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) 提供 NSA
  算法、program mapping 与论文性能结果；本文的 CSR backward 机制核对自
  [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
  社区实现，不能归因给论文作者的未公开 kernel。
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 提供 DSA/Lightning
  Indexer 的算法定义；公开参考逻辑位于
  [`DeepSeek-V3.2-Exp`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)。
- DSA scorer 与 sparse Attention 的代码级结论分别核对自
  [`DeepGEMM`](https://github.com/deepseek-ai/DeepGEMM) 与
  [`FlashMLA`](https://github.com/deepseek-ai/FlashMLA)。
- causal-prefill Discovery 规模实验的长尺度数据来自单张 H100；正文分别说明了
  重复 clean timing、分阶段 timing、4M 容量预检与长序列 Nsys 的计时口径。
  另一组 128K NCU 测量用于解释 carrier/Top-K 的具体流量。
- KV-row 局部性实验的数据同样来自单张 H100；完整条件、配对调度、统计方法、
  correctness checks、profiler 限制，以及 synthetic indices 与 production
  trace 的区别均在实验段落中说明。
- Blackwell `tile::gather4` 的指令语义以
  [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  为准。它不是本文 H100/SM90 实验所能验证的路径。
- 本文的组织方式参考
  [*MLA, dim by dim*](https://nonlinear1.com/en/posts/mla-dim-by-dim)
  及其[中文版](https://nonlinear1.com/zh/posts/mla-dim-by-dim)：
  先给结论，再用一个统一的 tensor lens 逐维推导；本文把同一方法进一步追到
  program ownership、memory movement 与 backward reduction。
