---
title: "Sparse Attention: Algorithm–Infra Co-design"
description: "From block-level selection in NSA to token-level selection in DSA: a shape-by-shape derivation through GPU program ownership, memory movement, and backward reduction."
date: 2026-08-10
draft: false
lang: en
translationKey: sparse-attention-co-design
category: technical
---


## 1. Opening

The article follows one analytical lens:

> **First rewrite the algorithm as tensor dataflow in which every dimension and
> dependency is explicit. Then trace how those logical tensors become tiles that one
> GPU program can own, move, and compute.**

Following that chain, we derive the algorithmic dataflow and kernel mapping of NSA and
DSA, then use two focused H100 experiments to test questions that arise naturally from
the implementation analysis. Five connected conclusions emerge.

1. **A sparsity ratio is not an execution plan.** It says how many query–KV
   interactions remain in the logical algorithm, but not how much work is required to
   discover, represent, and execute them. Selector/Top-K, metadata, indirect KV loads,
   load reuse, backward many-to-one reduction, and whether these operations can be
   reorganized into regular MMA tiles all enter the real cost.
2. **NSA creates regularity before the load.** One selection ID names a logically
   contiguous 64-token interval. The bases of different selected blocks may still be
   scattered, but the 64 rows inside a block can be expanded from one base with affine
   offsets instead of carrying one indirect index per token. The 16 query heads in a KV
   group also share the same IDs. The FLA community implementation audited here maps
   these two axes directly to a $[\text{query heads},\text{KV tokens}]=[16,64]$
   interaction tile.
3. **Forward and backward need different sparse graphs and different infrastructure.**
   Query-centric ownership gives $O^{\mathrm{slc}}$ and
   $dQ^{\mathrm{slc}}$ a unique owner, but
   $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ must reduce contributions from every query
   that selected the same KV block. The FLA community path transposes the forward
   adjacency `query → selected blocks` into CSR-form
   `KV block → selecting queries`, then makes one KV-block program the unique
   writer of the corresponding gradients. This avoids floating-point atomics and
   partial-gradient buffers, at the cost of CSR construction, discrete query gathers,
   variable fan-in, and CTA load imbalance.
4. **DSA reconstructs regularity after the load.** Final selection is fully dynamic at
   token granularity, but the 128 main heads do not maintain 128 unrelated address
   sets: for one query, they share the same Top-2048 IDs. The public H100
   sparse-prefill kernel uses two CTAs for those heads. Each CTA handles 64 heads and
   reuses every selected row 64 times after moving it into SMEM, so only the producer
   faces scattered global-memory row bases while Tensor Core consumers still see a
   regular $[64,576]$ tile. In our frozen
   $T_Q=128,\kappa=2048$ microbenchmark, eight prespecified random permutations of an
   otherwise identical selected set differ from ascending order by only
   $0.10\%$–$0.47\%$ in point estimate, and every paired 95% CI includes 1. The
   concentration sweep is non-monotonic; only 2K/4K tight windows show sub-percent
   pointwise signals. This says the kernel is robust to the tested patterns at this
   operating point—not that random memory access is free.
5. **DSA does not eliminate $O(L^2)$.**

   **Algorithmically,** for a causal sequence of length $L$, the Lightning Indexer
   still scores all causal query–history pairs:

   $$
   \frac{L(L+1)}{2}=\Theta(L^2).
   $$

   Exact Top-K in the public unfused path must also scan an FP32 score carrier that
   grows with context. Fixing $\kappa=2048$ limits the high-dimensional main
   Attention to $\Theta(L\kappa)$ pairs; it does not make the complete DSA Attention
   module linear.

   **Operationally,** in our frozen public-component chain on one H100 with
   $B=1,C=4096,\kappa=2048$, cumulative causal-prefill Discovery is
   $0.727\times$, $1.267\times$, and $33.60\times$ the FlashMLA time at 32K,
   64K, and 2M. The first prespecified Discovery-dominant checkpoint is 64K, so
   the measured discrete bracket is $[32\mathrm K,64\mathrm K]$ without
   interpolation. From 256K to 2M, the endpoint effective exponents are 2.027 for
   Discovery and 1.031 for sparse prefill. This operating point exposes the hidden
   quadratic term; it does not imply a universal 64K crossover in production DSA.

The central thesis can therefore be compressed into two sentences:

> **Sparsity becomes executable when the algorithm exposes a reuse axis that one
> GPU program can own. Whether it becomes speed depends on the cost of discovering,
> regularizing, and reducing that sparsity.**

This article does not re-explain naive dense Attention or FlashAttention. We begin
farther downstream: when the algorithm hands a kernel block IDs or token IDs rather
than regular matrices, how does the GPU actually execute them?

To keep evidence boundaries explicit, the article uses six labels only where needed:
`[PAPER]` for statements from papers, `[CODE]` for official public code,
`[CODE/community]` for feasible paths shown by community implementations,
`[DERIVATION]` for consequences of declared shapes, dtypes, or execution order,
`[MEASURED]` for formal measurements in frozen environments, and `[GAP]`
for information that remains unavailable. The original high-performance NSA kernels
are not fully public, so code-level backward analysis is explicitly attributed to the
community implementation. DSA's Indexer scorer and FlashMLA sparse-forward kernel are
public, but production Indexer–Top-K fusion, complete runtime orchestration, and
training backward remain unavailable. Nsight results below are diagnostic evidence;
they never replace formal CUDA-event latency.

## 2. A Unified Tensor Lens: From Logical Matrices to Executable Tiles

> **Takeaway.** What matters most below is not simply “how large a matrix is,” but the distinction among tensors that exist logically in the algorithm, tensors that are actually written to HBM, and tiles consumed by one CTA/program.

Unless stated otherwise, we omit the batch axis and use conceptual token-major shapes:

| Symbol | Meaning |
|---|---|
| $T_Q,T_K$ | numbers of query and KV tokens in the current invocation |
| $T$ | full sequence length in causal self-attention |
| $H_q,H_{kv}$ | numbers of query heads and KV heads/groups |
| $G=H_q/H_{kv}$ | number of query heads served by one KV head/group |
| $t,h$ | query-token position and query-head ID |
| $r=(t,h)$ | one independent Attention softmax row |
| $g(h)=\lfloor h/G\rfloor$ | KV head/group used by query head $h$ in GQA |
| $d_k,d_v$ | Q/K and V/O dimensions per head |
| $B_R,B_{\mathrm{kv}}$ | number of softmax rows and KV-token tile length owned by one program |
| $\alpha$ | Attention-score scale |

We count one FMA as 2 FLOPs. Unless otherwise noted, byte counts assume BF16, or 2 bytes per element. Analytical byte counts exclude cache-line overfetch, allocator overhead, pages/TLB, and inter-device communication.

**Logical shape.** Fix one query head $h\in[0,H_q)$. Under GQA, it uses KV
head/group $g(h)$:
$$
Q^{(h)}:[T_Q,d_k],\qquad
K^{(g(h))}:[T_K,d_k],\qquad
V^{(g(h))}:[T_K,d_v].
$$

The mathematical semantics are

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

We omit causal and sparse-selection masks here; every invisible position is treated as
$-\infty$ before softmax.

$S/P:[T_Q,T_K]$ is first and foremost a **logical shape**: it defines which inputs each output depends on. It does not imply that the kernel must write the complete score or probability tensor to HBM. Throughout the article, we distinguish three layers:

1. **logical shape**: the complete relation in the algorithm's semantics;
2. **materialized shape**: a tensor actually written to HBM for another kernel to read;
3. **tile shape**: the local working set that one CTA/program brings on-chip and feeds to MMA.

**Tile shape.** A kernel may tile along query tokens, or it may let several query heads
share one KV tile. We therefore use $B_R$ for the number of independent softmax rows
owned by the program instead of assuming that the row axis always means query tokens.
Let $i$ denote this row tile and $j$ the current KV tile:
$$
Q_i:[B_R,d_k],\qquad
K_j:[B_{\mathrm{kv}},d_k],\qquad
V_j:[B_{\mathrm{kv}},d_v],
$$

where all $B_R$ rows must be able to share the current $K_j,V_j$. This is the reuse
axis exposed by NSA and DSA. The two local contractions are:

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

At the algorithmic level, NSA and DSA differ in how $K_j,V_j$ are selected. At the
infrastructure level, they differ in how the tile is moved on-chip and who reuses that
load. When rows mean query tokens, $B_R$ is the query-token tile length. In NSA
selection forward, $B_R=G=16$; in a FlashMLA head64 CTA, $B_R=B_H=64$.

**Online softmax.** As long as one program exclusively owns these $B_R$ softmax rows,
it can retain three states:
$$
m:[B_R],\qquad
z:[B_R],\qquad
O_{\rm acc}:[B_R,d_v].
$$

Initialize

$$
m^{(0)}=-\infty,\qquad z^{(0)}=0,\qquad O_{\rm acc}^{(0)}=0.
$$

$m$ is the row-wise maximum over visited logits, $z$ is the exponential sum relative
to $m$, and $O_{\rm acc}$ is the value numerator under the same reference point.
When a new score tile $S_{ij}:[B_R,B_{\mathrm{kv}}]$ arrives, compute

$$
\widehat m_j
=\max_{b=0,\ldots,B_{\mathrm{kv}}-1}S_{ij}[:,b]
\in\mathbb R^{B_R},
$$

$$
m_{\rm new}=\max(m_{\rm old},\widehat m_j)
\in\mathbb R^{B_R},
$$

and rescale the old state to this new row-wise maximum:

$$
\rho=\exp(m_{\rm old}-m_{\rm new})
\in\mathbb R^{B_R}.
$$

The unnormalized weights of the current tile are

$$
\widetilde P_{ij}
=\exp\!\left(S_{ij}-m_{\rm new}[:,\mathrm{None}]\right)
\in\mathbb R^{B_R\times B_{\mathrm{kv}}}.
$$

The state update is then

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

After all KV tiles:

$$
O_i
=
\frac{O_{\rm acc}}{z[:,\mathrm{None}]}
\in\mathbb R^{B_R\times d_v},\qquad
\operatorname{LSE}_i=m+\log z
\in\mathbb R^{B_R}.
$$

Selected blocks or rows may therefore be visited in batches without materializing the
complete $P$. In exact arithmetic, changing the order of KV tiles does not change the
mathematical result; in finite precision, reduction order can introduce small numerical
differences. NSA and DSA change which KV tiles enter this stream, not softmax semantics.

From this point onward, we repeatedly ask five questions about each sparse path:

1. **Stage/materialization:** Which tensors exist only logically, and which cross a
   kernel boundary through HBM?
2. **Grid/ownership:** How many programs are in the grid? Which output rows does one
   program own, and who holds $m,z,O_{\rm acc}$?
3. **Tile/reuse:** What tile does the program consume? How many query rows or heads
   reuse each KV row or tile moved on-chip?
4. **Irregularity boundary:** Where does the dynamic index appear, and where is it
   rearranged into a regular tile?
5. **Full cost:** What remains in the selector, Top-K, metadata, gather, load balance,
   and backward reduction?

The two algorithms provide complementary answers:

$$
\boxed{
\begin{aligned}
\text{NSA: }&
\text{Before the HBM load, constrain the sparse relation to contiguous token blocks;}\\
\text{DSA: }&
\text{Retain token-level indirect addressing, then reconstruct regular tiles in SMEM after the load.}
\end{aligned}}
$$

## 3. NSA: Make the Algorithm Produce Executable Block Workloads

> **Takeaway.** Native Sparse Attention (NSA) makes each selection ID point to a contiguous 64-token interval, allowing the semantic block to be decomposed into an integer number of regular kernel tiles. The FLA path analyzed here further chooses
> $B_{\mathrm{kv}}=\ell'=64$, so one tile covers exactly one block and the 16 query heads sharing the same KV group reuse that K/V region.

> **A note on the implementation used here.** The NSA paper publishes the algorithm,
> the high-level program mapping of its Triton kernels, and performance results on
> 8×A100, but not the original kernels used in those experiments. The formulas below
> come from the paper; the code-level analysis of online Top-k, group-centric forward,
> and CSR backward comes from the current FLA community implementation. FLA shows one
> feasible implementation path, but simplifies the paper's learned overlapping
> compression and is not an official DeepSeek implementation.

NSA decomposes long-context attention into three access patterns:

$$
\text{compressed global context}
\;+\;
\text{dynamically selected salient blocks}
\;+\;
\text{fixed local window}.
$$

The crucial chain of derivation is not merely “sum three branches,” but

$$
\text{information demand}
\longrightarrow
\text{block IDs}
\longrightarrow
\text{GQA workload with shared block selection}
\longrightarrow
\text{regular }16\times64\text{ matmul tile}.
$$

### 3.1 Algorithmic Contract: What Each Sparse Branch Contributes

**Notation.** We omit the batch axis and use a sequence-major layout. For a sequence of length $T$,
$$
Q\in\mathbb{R}^{T\times H_q\times d_k}.
$$

NSA targets GQA. The paper's efficiency experiments use

$$
H_q=64,\qquad H_{kv}=4,\qquad G=H_q/H_{kv}=16,\qquad d_k=192,\qquad d_v=128.
$$

The query-head group corresponding to KV head $g$ is

$$
\mathcal H_g=\{gG,\ldots,(g+1)G-1\}.
$$

The algorithmic parameters of the three branches are:

| Branch | Sparse unit | Paper configuration | Visible extent per query |
|---|---:|---:|---:|
| compression | overlapping window | $\ell=32,d=16$ | approximately $L_t/d$ |
| selection | contiguous token block | $\ell'=64,n=16$ | at most $n\ell'=1024$ |
| sliding | local token window | $w=512$ | at most 512 |

Here $L_t=t+1$ is the length of the causal prefix at position $t$. It contains

$$
C_t=\max\left(0,\left\lfloor\frac{L_t-\ell}{d}\right\rfloor+1\right)
$$

complete compression windows, and

$$
M_t=\left\lceil\frac{L_t}{\ell'}\right\rceil
$$

causally visible selection blocks.

**Independent representations for the three branches.** The paper gives each branch its own K/V projection:

$$
K^c\in\mathbb{R}^{T\times H_{kv}\times d_k},\qquad V^c\in\mathbb{R}^{T\times H_{kv}\times d_v},\qquad c\in\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}.
$$

Therefore, even if multiple branches access the same token, we cannot first take a union of token IDs and treat the result as one attention operation. The branches use different representations, normalize independently, and only then combine their outputs.

**Compression.** Each KV head maps an overlapping window of length $\ell=32$ through a learned compressor into one compressed K/V row. Adjacent windows have stride $d=16$, so a full sequence produces approximately $C_T\approx T/d$ compressed K/V rows.

The local shapes for position $t$ and group $g$ are

$$
Q_{t,\mathcal H_g,:}\in\mathbb{R}^{G\times d_k},\qquad
\widetilde K^{\mathrm{cmp}}_{0:C_t,g,:}\in\mathbb{R}^{C_t\times d_k},\qquad
\widetilde V^{\mathrm{cmp}}_{0:C_t,g,:}\in\mathbb{R}^{C_t\times d_v}.
$$

$$
[G,d_k][d_k,C_t]\rightarrow[G,C_t],\qquad [G,C_t][C_t,d_v]\rightarrow[G,d_v].
$$

With a fixed stride $d$, the total compression workload across all queries remains

$$
\sum_{t=0}^{T-1}\Theta(C_t)=\Theta(T^2/d).
$$

It is smaller than dense attention by a constant factor of roughly $d$, but it is not linear in sequence length.

**Sliding window.** The local branch accesses only

$$
\mathcal W_t=\{\max(0,t-w+1),\ldots,t\},\qquad w=512,
$$

providing fixed, contiguous local context that tiled attention can handle naturally.

**Combining the three branches.** Each branch performs its own softmax and produces

$$
O^{\mathrm{cmp}}_{t,h,:},\ O^{\mathrm{slc}}_{t,h,:},\ O^{\mathrm{win}}_{t,h,:}\in\mathbb{R}^{d_v},
$$

which are then combined by independent sigmoid gates:

$$
O_{t,h,:}=\gamma^{\mathrm{cmp}}_{t,h}O^{\mathrm{cmp}}_{t,h,:}+\gamma^{\mathrm{slc}}_{t,h}O^{\mathrm{slc}}_{t,h,:}+\gamma^{\mathrm{win}}_{t,h}O^{\mathrm{win}}_{t,h,:}.
$$

These gates need not sum to 1. In the execution DAG, compression attention and sliding attention can run in parallel; selection attention must wait for the routing IDs.

### 3.2 Routing: From Compression Probabilities to Top-16 Block IDs

Selection does not train an entirely independent token router. It reuses the probabilities already learned by compression attention and remaps compressed-window importance onto 64-token selection blocks.

**Spatial remapping.** Let

$$
P^{\mathrm{cmp}}_{t,h}\in\mathbb{R}^{C_t}
$$

be the compression probability for query head $h$. Equation 9 of the paper converts it into a selection-block score:

$$
R^{\mathrm{slc}}_{t,h}[j]
=
\sum_{m=0}^{\ell'/d-1}
\sum_{u=0}^{\ell/d-1}
P^{\mathrm{cmp}}_{t,h}
\!\left[\frac{\ell'}{d}j-m-u\right],
\qquad j\in\{0,\ldots,M_t-1\}.
$$

An out-of-range compressed-window index is treated as zero. The paper's configuration satisfies

$$
\ell'/d=4,\qquad \ell/d=2.
$$

This double sum handles the fact that overlapping compression windows and selection blocks do not correspond one-to-one.

**GQA group reduction and Top-$n$.** The $G=16$ query heads sharing a KV head must use the same selection IDs, so

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

The concrete configuration in Section 4 of the paper further specifies that, among the 16 selection slots, one initial block and the two most recent local blocks are always active; only the remaining slots are filled by the dynamic ranking above.
`[PAPER]`

The fixed initial block deterministically creates a hot row with $\Theta(T)$ fan-in. The two local slots move forward with the query: under the current-plus-previous rule alone, any particular block serves at most about $2\ell'=128$ query positions. Dynamic ranking can create additional hot spots.
`[DERIVATION]`

At a mature position,

$$
I^{\mathrm{slc}}_{t,g}
=(j_0,\ldots,j_{n-1})\in\mathbb{Z}^{16}.
$$

This is an index tensor with an explicit slot order, not a mathematical set. Near the beginning of a sequence, fewer than 16 causally valid blocks may exist. The kernel may still maintain 16 selection slots, but padding must be marked by $-1$ or by a block count. “A fixed 16 slots” does not mean “16 valid blocks always exist.”

The complete shape chain is

```text
compression probabilities       [T,Hq,≈T/16]
  → Eq. 9 spatial remap          [T,Hq,≈T/64]
  → reduce G=16 query heads      [T,Hkv,≈T/64]
  → Top-16                       [T,Hkv,16]
```

**Hidden materialization cost.** Standard FlashAttention does not write the complete probability matrix back to HBM; it usually returns only the output and LSE. Therefore, “reuse the compression probability” does not by itself explain how that probability reaches Top-k.

At $T=65536$, the actual number of complete windows is $C_T=4095$. To illustrate the upper bound for a fixed-shape allocation, let the padded capacity be

$$
\bar C=\left\lceil T/d\right\rceil=4096.
$$

Materializing per-head FP32 compressed scores at that capacity,

$$
[T,H_q,\bar C],
$$

requires

$$
65536\times64\times4096\times4=64\ \mathrm{GiB}.
$$

per layer.

If we first sum over each GQA group **before** the Equation 9 remap, materializing

$$
[T,H_{kv},\bar C],
$$

still requires

$$
65536\times4\times4096\times4=4\ \mathrm{GiB}.
$$

If instead we are discussing group-reduced block scores **after** the remap, let
$M_T=\lceil T/\ell'\rceil=1024$. Their shape should be
$$
[T,H_{kv},M_T],
$$

corresponding to

$$
65536\times4\times1024\times4=1\ \mathrm{GiB}.
$$

These three numbers describe different locations in the dataflow and must not be conflated.

The current FLA implementation demonstrates one feasible mechanism: save the compression LSE, recompute probabilities in blocks from Q and compressed K, and maintain Top-$n$ online using bitonic merges, thereby avoiding materialization of the quadratic score tensors above. This path trades recomputation for memory capacity: the Top-k kernel must scan compressed K with Q once more, so routing still contains a $\Theta(T^2/d)$ QK scan and candidate merging. It eliminates HBM materialization of the quadratic score tensor, not quadratic pair scoring itself.

The accompanying path also simplifies compression to non-overlapping mean pooling, shares K/V, and imposes implementation constraints that the GQA group size be at least 16 and a power of two. None of these is part of the paper algorithm's general definition. FLA therefore proves only that one infrastructure path is feasible; it does not prove that the paper authors used the same implementation. `[CODE/community] [GAP]`

### 3.3 Selection Forward: Logical Gather, Physical Block Tile

Routing produces discrete block IDs:

$$
I^{\mathrm{slc}}_{t,g}
=(j_0,\ldots,j_{n-1})\in\mathbb{Z}^{n}.
$$

Each $j_s$ points to a contiguous 64-token block in selection K/V:

$$
\mathcal B_{j_s}=\{j_s\ell',\ldots,(j_s+1)\ell'-1\},\qquad \ell'=64.
$$

**Logical view.** For a mature query, if all 16 slots are valid, we can write

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

The logical attention shapes are

$$
[16,192][192,1024]\rightarrow[16,1024],\qquad [16,1024][1024,128]\rightarrow[16,128].
$$

Here 1024 is the padded capacity, not the number of valid tokens at every position. Some slots may be invalid near the beginning of the sequence, and a causal block containing the current position may expose only part of its tokens. An implementation must apply the ID mask, range mask, and causal mask together.

The $[1024,d]$ above is only the **logical shape** after concatenation. In the original KV storage, it corresponds to 16 mutually disjoint 64-token blocks, each internally contiguous in token index; it is not one contiguous 1,024-token source range. If the physical layout is token-major
$[T,H_{kv},D]$, adjacent token rows for a fixed head have a fixed stride and need not form a byte-contiguous $64D$ interval. What the algorithm truly improves is the addressing structure: one block requires only one irregular base address, from which 64 rows are expanded by regular affine offsets, rather than 64 separate indirect token lookups.

If we actually constructed these two tensors for every query/group, the external gather would need both to handle 16 irregular block bases and to assemble them into a compact selected-KV tensor written back to HBM. The K/V payload alone would be

$$
1024(192+128)\times2=655{,}360\ \mathrm{bytes}=640\ \mathrm{KiB}
$$

in BF16. This creates an extra write, after which attention reads the same data again. The correct approach is to retain only the IDs and load the original K/V by block inside the attention kernel.

**Program ownership.** The most natural owner for selection forward is

$$
\boxed{\text{one query position}\times\text{one KV group}}.
$$

We first expand all input shapes and grid symbols used by this source path. For fixed-length inputs, FLA uses

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

Here:

- $B$ is the batch size;
- $T_Q,T_K$ are the numbers of query and KV tokens per sequence in the current invocation;
- $H_q,H_{kv}$ are the numbers of query and KV heads;
- $G=H_q/H_{kv}$ is the number of query heads served by one KV head;
- $d_k,d_v$ are the Q/K feature and V/output feature dimensions per head;
- $N_{\mathrm{slc}}$ is the slot capacity along the final dimension of the selection-ID tensor, written as
  `S` in the source; the configuration here uses 16;
- $I^{\mathrm{slc}}$ stores block IDs. When `block_counts` is a tensor, it records the number of valid slots for each query/group. When it is a scalar capacity, the kernel scans all $N_{\mathrm{slc}}$ slots;
- $B_V$ is the tile width that one program computes along the value-feature axis;
- $N_V=\lceil d_v/B_V\rceil$ is the number of tiles required to cover the complete value dimension.

The NSA numerical configuration used here is

$$
H_q=64,\qquad H_{kv}=4,\qquad G=16,\qquad d_k=192,\qquad d_v=128.
$$

FLA further assumes that $Q$ corresponds to the last $T_Q$ tokens of each KV sequence. Thus, on the fixed-length path, the absolute position of query slot $t$ within the KV sequence is not simply $t$, but

$$
p_t=T_K-T_Q+t.
$$

The subsequent causal mask and selected-block start positions are both evaluated relative to $p_t$.

The launch grid in the source is a three-dimensional **program-count tuple**:

$$
\boxed{
\mathrm{grid}
=
\left(T_Q,\ N_V,\ B\times H_{kv}\right)
}.
$$

The source names the three program IDs `i_t`, `i_v`, and `i_bh`. To avoid confusing
`i_v` with the value tensor $V$, we rename the second ID $\nu$:

$$
t\in[0,T_Q),\qquad
\nu\in[0,N_V),\qquad
i_{bh}\in[0,B\times H_{kv}).
$$

They represent, in order, the query position, the value-feature tile, and a flattened batch/KV-head position. The third ID is unflattened as

$$
a=\left\lfloor\frac{i_{bh}}{H_{kv}}\right\rfloor,\qquad
g=i_{bh}\bmod H_{kv}
$$

to recover batch index $a$ and KV-head/group index $g$. We intentionally use $a$ for the batch here to avoid confusion with the KV block ID $b$ below. The query heads served by group $g$ remain

$$
\mathcal H_g=\{gG,\ldots,(g+1)G-1\}.
$$

Next define the value-feature interval owned by program $\nu$:

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

Program $(t,\nu,i_{bh})$, equivalently $(a,t,g,\nu)$, is then the unique writer of the following output slice:

$$
O^{\mathrm{slc}}
\left[a,t,\mathcal H_g,\mathcal D_\nu\right]
\in
\mathbb{R}^{G\times|\mathcal D_\nu|}.
$$

The physical accumulator is allocated as $[G,B_V]$; a final tile narrower than $B_V$ is handled by a feature mask. This also makes the axis order explicit: after removing batch flattening, the grid is
$(T_Q,N_V,H_{kv})$, not $(T_Q,H_{kv},N_V)$.

If $N_V>1$, the same $(a,t,g)$ expands into $N_V$ programs. Each program recomputes the same QK/softmax for its own value-feature slice, but the
$\mathcal D_\nu$ intervals are disjoint, so there is no output write conflict. LSE is not tiled along the value dimension; the source permits only the $\nu=0$ program to write
$\mathrm{LSE}^{\mathrm{slc}}[a,t,\mathcal H_g]$.

The paper uses $d_v=128$, and the current FLA path also takes $B_V=128$, so

$$
N_V=1,\qquad \mathcal D_0=\{0,\ldots,127\}.
$$

Each $(a,t,g)$ then corresponds to exactly one program, which exclusively owns the complete selection output
$[G,d_v]=[16,128]$. The program subsequently loads and keeps resident the logical query tile

$$
Q_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{G\times d_k}
=
\mathbb{R}^{16\times192}.
$$

The public variable-length path uses a packed $B=1$ representation. If sequence $i$ contains $T_Q^{(i)}$ query tokens and $T_K^{(i)}$ KV tokens, respectively, then

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

The grid then becomes

$$
\left(T_Q^{\mathrm{total}},N_V,H_{kv}\right).
$$

The first program ID $u\in[0,T_Q^{\mathrm{total}})$ is a packed query slot;
`token_indices_q[u]` maps it to
$(i_{\mathrm{seq}},t_{\mathrm{local}})$. Let

$$
\mathrm{bos}_Q(i)
=
\texttt{cu\_seqlens\_q}[i]
$$

denote the starting offset of sequence $i$ in packed Q. The physical output owner is then

$$
O^{\mathrm{slc}}
\left[
0,\,
\mathrm{bos}_Q(i_{\mathrm{seq}})+t_{\mathrm{local}},\,
\mathcal H_g,\,
\mathcal D_\nu
\right].
$$

The corresponding causal absolute position of this query within its sequence's KV storage is

$$
p_{i_{\mathrm{seq}},t_{\mathrm{local}}}
=
T_K^{(i_{\mathrm{seq}})}
-
T_Q^{(i_{\mathrm{seq}})}
+
t_{\mathrm{local}}.
$$

Thus, varlen changes only how fixed-length $(a,t)$ coordinates are located. The $\nu$ and $g$ grid axes, and the ownership rule that “one program uniquely writes one output slice,” remain unchanged.

The program then loops over the 16 selection slots. We use $B_{\mathrm{kv}}$ for the number of KV-token rows loaded and computed in one inner-loop iteration. In the current FLA path, it corresponds to the source parameter `BS` and exactly equals NSA's selection block length $\ell'$:

$$
B_{\mathrm{kv}}=\texttt{BS}=\ell'=64.
$$

For a valid block ID $j_s$, the program computes $\mathrm{base}=j_s\ell'$ and loads

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

This is the logical tile at the algorithmic level. In the current Hopper branch of FLA,

$$
\texttt{BK}
=
\min\!\left(256,\operatorname{nextPow2}(d_k)\right)
=
256,
$$

so matching Triton's dot-product feature tile requires the physical construction

$$
b_Q:[16,256],\qquad
b_K:[256,64],\qquad
b_V:[64,128],
$$

where the $192\rightarrow256$ tail of the feature dimension is handled by masking/padding. The logical contractions are therefore

$$
[16,192][192,64]\rightarrow[16,64],\qquad [16,64][64,128]\rightarrow[16,128].
$$

The physical padded execution shape for QK is

$$
[16,256][256,64]\rightarrow[16,64].
$$

This $192\rightarrow256$ execution shape is a conclusion about the **current FLA Hopper path**. The non-Hopper branch caps `BK` at 128; with $d_k=192$, it produces two feature tiles and triggers the source's `NK == 1` assertion. The
$[16,256][256,64]$ shape therefore cannot be extrapolated directly to that path.

The only dynamic component is the block base computed before each load. Once the data reaches the on-chip tile, computation returns to a fixed shape.

We deliberately do not use $B_K$ for the token tile here. The NSA paper's token-tile notation $B_K$ corresponds to `BS` in the current FLA source, whereas FLA's `BK` denotes the padded feature width—for example,

$$
d_k=192\longrightarrow \texttt{BK}=256.
$$

Mixing the two notation systems would mistake “64 tokens” for “256 feature elements.”

**Online normalization.** Across the 16 blocks, the program uses the exact online-softmax merge introduced above. It maintains only the running maximum, normalizer, and output accumulator for each query head; it does not write
$P^{\mathrm{slc}}\in\mathbb{R}^{16\times1024}$. After all selected blocks have been processed, it directly writes

$$
O^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{16\times128}.
$$

The output therefore has a unique writer, and the forward pass needs no atomic reduction on the output.

**Why head sharing is an execution contract.** For one 64-token block, the QK and PV multiply-add count is approximately

$$
2\,G\,B_{\mathrm{kv}}(d_k+d_v)
=
2\times16\times64\times(192+128)
=
655{,}360\ \mathrm{FLOPs}.
$$

With BF16 K/V, the primary K/V payload for this iteration is

$$
2\,B_{\mathrm{kv}}(d_k+d_v)
=
2\times64\times(192+128)
=
40{,}960\ \mathrm{bytes}.
$$

Considering only the K/V payload shared by 16 heads, the local arithmetic intensity is

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

This is not a roofline figure for the entire kernel: it excludes Q, output, indices, LSE, masks, and launch overhead. It only shows that the more query heads reuse the same K/V block, the more easily the cost of one irregular address lookup is amortized by regular matrix multiplication.

For a mature query, the main selection loop has 16 slots. Near the beginning of a sequence, the same control structure still runs, but invalid slots are masked. NSA provides a “fixed-capacity, padding-compatible workload,” not an absolutely fixed amount of work without boundary conditions.

### 3.4 One Public Backward Path: Transposing the Sparse Graph in FLA

> **Takeaway.** The forward relation `query → selected blocks` finds an owner only for the output.
> If we retain query-centric ownership,
> $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ requires floating-point atomics
> or a partial-gradient reduction. The FLA community implementation audited here instead transposes the relation into
> `KV block → selecting queries`, then lets each KV-block program perform its own
> many-to-one reduction.

The natural owner in the forward pass is the query, because each program can independently complete one
$O^{\mathrm{slc}}_{t,\mathcal H_g,:}$. The backward pass, however, has two reduction directions.

The discussion below covers only the backward pass of the selection branch. Complete NSA also includes gates, a compression branch, a sliding branch, and the merging of gradients from all three branches into shared Q/input tensors.

**Saved state and recomputation.** Selection backward does not need to save the complete probability matrix. It recomputes score/probability tiles from
$Q,K^{\mathrm{slc}},I^{\mathrm{slc}}$, and
$\mathrm{LSE}^{\mathrm{slc}}$, then combines them with
$dO^{\mathrm{slc}}$ to recover local gradients. For batch/sequence $a$, query $t$,
group $g$, and selected block $b$, the local shapes are

$$
S^{\mathrm{slc}}_{a,t,g,b},\qquad
P^{\mathrm{slc}}_{a,t,g,b},\qquad
dS^{\mathrm{slc}}_{a,t,g,b}
\in\mathbb{R}^{G\times B_{\mathrm{kv}}}.
$$

Without activation checkpointing, the FLA selection autograd state includes at least

$$
Q,\quad K^{\mathrm{slc}},\quad V^{\mathrm{slc}},\quad
O^{\mathrm{slc}},\quad \mathrm{LSE}^{\mathrm{slc}},\quad
I^{\mathrm{slc}},
$$

as well as block counts and sequence metadata required in variable-length cases. The current FLA autograd path actually saves `q, k, v, o, lse` and retains metadata such as block indices in the context. The accurate conclusion is “$P^{\mathrm{slc}}$ is not saved,” not “the forward pass only needs to save
$O^{\mathrm{slc}},\mathrm{LSE}^{\mathrm{slc}},I^{\mathrm{slc}}$.”

**$dQ$: the query remains the owner.** For query $t$, only the blocks it selected contribute:

$$
dQ^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
=
\alpha
\sum_{b\in I^{\mathrm{slc}}_{a,t,g}}
dS^{\mathrm{slc}}_{a,t,g,b}
K^{\mathrm{slc}}_{a,g,b}.
$$

The local shape for each block is

$$
[G,B_{\mathrm{kv}}][B_{\mathrm{kv}},d_k]\rightarrow[G,d_k].
$$

The same program can loop over all slots and accumulate

$$
dQ^{\mathrm{slc}}_{a,t,\mathcal H_g,:}
\in\mathbb{R}^{16\times192},
$$

in registers before writing it back exactly once. Thus, dQ remains well suited to query-centric ownership.

**$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$: query-centric ownership creates write conflicts.**
One KV block $b$ may be selected by many queries. Define

$$
\mathcal Q_{a,g,b}
=
\left\{t\mid b\in I^{\mathrm{slc}}_{a,t,g}\right\}.
$$

Its gradients are many-to-one reductions across queries:

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

If we keep the query as owner, different programs update the same
$dK^{\mathrm{slc}}_b,dV^{\mathrm{slc}}_b$ concurrently. The implementation must then use many floating-point atomics or write partial buffers and reduce them later. A more natural approach is to transpose the sparse relation first:

```text
forward adjacency:   query → selected KV blocks
backward adjacency:  KV block → all selecting queries
```

and then assign one program to each $(a,g,b)$ KV block.

**CSR representation.** Let the batch size be $B$, and let the number of blocks in a complete sequence be

$$
M=\left\lceil T/\ell'\right\rceil.
$$

The padded shape of the input IDs is $[B,T,H_{kv},n]$. The upper bound on edge capacity allocated for `csr_indices` by the current FLA implementation is

$$
E_{\mathrm{alloc}}=B\,T\,H_{kv}\,n.
$$

The actual number of valid edges is determined by the ID range, block count, and causal condition:

$$
E_{\mathrm{valid}}=\texttt{csr\_offsets}[-1]\le E_{\mathrm{alloc}}.
$$

Flatten $(a,g,b)$ into a one-dimensional row:

$$
r(a,g,b)=\bigl(aH_{kv}+g\bigr)M+b.
$$

The CSR metadata shapes are

$$
\texttt{csr\_offsets}
\in\mathbb{Z}^{B\,H_{kv}\,M+1},
\qquad
\texttt{csr\_indices}
\in\mathbb{Z}^{E_{\mathrm{alloc}}},
$$

where only $\texttt{csr\_indices}[0:E_{\mathrm{valid}}]$ is a valid prefix. Row
$r=r(a,g,b)$ corresponds to

$$
\mathcal Q_{a,g,b}
=
\texttt{csr\_indices}
\!\left[
\texttt{csr\_offsets}[r]:
\texttt{csr\_offsets}[r+1]
\right].
$$

This is one-dimensional flattened CSR. It must not be written as a nonexistent two-dimensional index $\texttt{offsets}[g,b]$.

CSR is constructed with two counting/scatter passes:

```text
traverse valid selection edges
  → atomically count the fan-in of each (batch,group,block)
  → prefix sum to obtain csr_offsets
  → traverse the edges again and scatter query IDs using an atomic cursor
```

Atomics operate only on compact integer metadata; the expensive floating-point reductions for
$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ obtain a unique owner.

Take

$$
B=1,\quad T=65536,\quad H_{kv}=4,\quad n=16,\quad \ell'=64,\quad M=1024.
$$

Then

$$
E_{\mathrm{alloc}}=65536\times4\times16=4{,}194{,}304,
$$

so the int32 `csr_indices` buffer allocated by the current FLA implementation is 16 MiB.

For a complete causal self-attention sequence, suppose early positions use all distinct visible blocks. The first 15 query blocks of 64 tokens then have $1,\ldots,15$ valid slots respectively, after which the count reaches 16. In this case,

$$
E_{\mathrm{valid}}
=
H_{kv}\,\ell'
\left(\sum_{k=1}^{15}k+16(M-15)\right)
=
4{,}163{,}584.
$$

This number is smaller than the allocation upper bound; the two should not both be denoted by the same $E$.

Meanwhile,

$$
B\,H_{kv}\,M+1=4097
$$

int32 offsets require only about 16 KiB. For comparison, materializing a uint8 inverse mask of shape $[T,H_{kv},M]$ requires

$$
65536\times4\times1024=256\ \mathrm{MiB}.
$$

CSR reduces inverse metadata from $\Theta(T\,H_{kv}\,M)$ to
$O(E_{\mathrm{alloc}}+B\,H_{kv}\,M)$, at the cost of two edge traversals, integer
atomics, a prefix sum, and subsequent irregular query gathers.

**KV-centric dKV tile.** A dKV program owns

$$
K_b\in\mathbb{R}^{B_{\mathrm{kv}}\times d_k},\qquad V_b\in\mathbb{R}^{B_{\mathrm{kv}}\times d_v}
$$

and gradient accumulators of the same shapes. It reads $B_{\mathrm{qry}}$ query positions at a time from the corresponding CSR row and expands the $G$ heads of each query:

$$
R=B_{\mathrm{qry}}G.
$$

The batched query and output-gradient shapes are

$$
Q_{\mathrm{batch}}\in\mathbb{R}^{R\times d_k},\qquad dO_{\mathrm{batch}}\in\mathbb{R}^{R\times d_v}.
$$

Local recomputation and reduction once again form regular matrix multiplications:

$$
[B_{\mathrm{kv}},d_k][d_k,R]\rightarrow[B_{\mathrm{kv}},R],
$$

$$
[B_{\mathrm{kv}},R][R,d_v]\rightarrow[B_{\mathrm{kv}},d_v],\qquad [B_{\mathrm{kv}},R][R,d_k]\rightarrow[B_{\mathrm{kv}},d_k].
$$

For example, when $B_{\mathrm{kv}}=64,B_{\mathrm{qry}}=4,G=16$, we have $R=64$, and the central score-recomputation tile is

$$
[64,192][192,64]\rightarrow[64,64].
$$

CSR does more than compress metadata. It rewrites the conflict “many queries write the same KV-gradient address” into “one KV owner sequentially reads its own queries,” and then restores a regular Tensor Core tile within that owner.

**Remaining irregularity: fan-in.** Define

$$
NQ_{a,g,b}=|\mathcal Q_{a,g,b}|.
$$

When `block_counts=16` is a fixed scalar capacity, each
$dQ^{\mathrm{slc}}$ program scans 16 padded slots. FLA also accepts a per-query tensor `block_counts`, in which case the runtime $N_S$ may be shorter. The loop length of a dKV program, by contrast, is determined by $NQ_{a,g,b}$. A fixed initial/sink block, as well as popular blocks created by dynamic ranking, may have fan-in far above the mean and form long-tail CTAs; the moving local slots themselves create only bounded local fan-in. Splitting a hot row can improve load balance but reintroduces a partial reduction or atomic operations. The backward problem therefore changes from “how many FLOPs can be skipped?” to “how should a many-to-one reduction be scheduled?”

We do not turn this long tail in fan-in into a performance experiment here. On the one hand, the backward kernel used by the paper's authors is not public. On the other hand, changing the routing topology in the community implementation changes fan-in skew, query-gather locality, and CSR row order together, making it difficult to attribute latency differences to a single variable. FLA backward therefore moves irregularity from floating-point write conflicts into CSR row length, query-gather locality, and CTA load balance. We report no performance measurement for these factors; we treat them only as execution consequences confirmed by the source and do not claim that they explain the backward latency of the paper authors' kernel. `[DERIVATION] [CODE/community] [GAP]`

### 3.5 NSA's Regularity Comes from the Algorithmic Contract, Not Kernel Magic

NSA selection maps efficiently to a kernel because the algorithm deliberately provides three execution contracts:

$$
\boxed{\text{contiguous 64-token blocks}+\text{16 heads sharing IDs}+\text{a fixed 16-slot capacity that supports padding}}
$$

The forward pass brings dynamic block bases into the program's inner loop and restores regular
$[16,64]$ computation after the load. The FLA community backward path analyzed here then transposes the sparse graph so that a KV block becomes the unique owner of
$dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$.

This also reveals three boundaries that must not be crossed.

First, NSA is not evidence that “arbitrary block sparsity automatically accelerates.” Hardware efficiency arises only when contiguous blocks, head sharing, a fixed slot capacity, and explicit ownership all hold simultaneously.

Second, the token budgets of the selection and sliding branches are approximately fixed at mature positions, but compression still accesses roughly $L_t/d$ compressed K/V rows. For fixed $d$, full-layer compression remains $\Theta(T^2/d)$.

Third, FLA demonstrates a complete feasible mechanism comprising online Top-k, group-centric forward, and CSR backward. It cannot prove that the paper authors' unreleased implementation uses the same routing materialization, tile parameters, or inverse-adjacency construction.

The paper reports that, with its $G=16,\ell'=64,n=16$ configuration and on its A100 test environment, the authors' Triton NSA reaches 9.0× forward and 6.0× backward speedup over Triton FlashAttention-2 at 64K. This result can be attributed to the complete system reported in the paper; it cannot be attributed directly to any single kernel choice isolated from the community code in this article. `[PAPER] [GAP]`

In other words, the most reusable part of NSA is not one particular sparsity formula, but a co-design method:

> First make the algorithm emit sparse units that a GPU can consume, then choose a unique owner separately for the forward output and the backward gradient.

**Sources for this section**

- [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
  Attention*](https://arxiv.org/abs/2502.11089), Sections 3.2–3.4 and 4–5.
  `[PAPER]`
- [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
  commit
  [`0a9b9f2`](https://github.com/fla-org/flash-linear-attention/tree/0a9b9f222e86b9a895c2447767e9b4cce6c8d530):
  [`fla/ops/nsa/parallel.py`](https://github.com/fla-org/flash-linear-attention/blob/0a9b9f222e86b9a895c2447767e9b4cce6c8d530/fla/ops/nsa/parallel.py)
  and
  [`fla/ops/utils/csr.py`](https://github.com/fla-org/flash-linear-attention/blob/0a9b9f222e86b9a895c2447767e9b4cce6c8d530/fla/ops/utils/csr.py).
  `[CODE/community]`
- The conclusions about FLA's grid, padded tiles, and CSR ownership were checked
  line by line against `parallel.py` and `csr.py` at the pinned commit above.
  `[CODE/community]`

## 4. DSA: Turning Arbitrary Token Selection Back into Regular Matrices

> **Takeaway.** DSA retains arbitrary token-level selection, but first makes all 128
> main heads share the same set of IDs. FlashMLA then confines random row addresses
> to the HBM→SMEM producer boundary. The algorithm provides an axis of reuse, which
> lets the kernel reconstruct discrete rows as regular matrices.

NSA confines sparsity to the boundaries between regular tiles by selecting contiguous
blocks. DeepSeek Sparse Attention (DSA) operates at a finer granularity: each query can
select arbitrary tokens from the entire history.

This gives the selector more freedom, but it also destroys the local continuity on which
block attention relies. A literal implementation would first gather a set of discrete KV
rows, write a selected-KV tensor, and then feed that tensor to an ordinary attention
kernel. This avoids most QK and PV computation, but leaves a random gather and an extra
HBM round trip at the kernel boundary.

DSA is therefore about more than Top-K. Three levels of the design jointly change the
execution shape:

$$
\text{low-dimensional Indexer scans history} \rightarrow \text{all main heads share token IDs} \rightarrow \text{FlashMLA reconstructs regular tiles in SMEM}.
$$

The table below keeps only the notation needed to understand this chain.

| Symbol | Meaning | DeepSeek-V3.2 |
|---|---|---:|
| $T_Q,T_K$ | Number of queries in the current call and number of visible KV tokens | Determined at runtime |
| $H_I,D_I$ | Number and dimension of Indexer heads | $64,128$ |
| $H$ | Number of main MLA query heads | $128$ |
| $D_C,D_R$ | Latent-KV and RoPE dimensions | $512,64$ |
| $D_N$ | NoPE dimension of the original main MHA | $128$ |
| $D_V^{\mathrm{MHA}}$ | Value dimension of the original MHA | $128$ |
| $\kappa$ | Maximum number of tokens selected per query | $2048$ |

Here, $D_V^{\mathrm{MHA}}=128$ must not be conflated with FlashMLA's internal
latent-value dimension $D_C=512$. The main Attention also admits two algebraically
equivalent representations with different execution shapes:

$$
D_{QK}^{\mathrm{MHA}} =D_N+D_R =192,
$$

$$
D_{QK}^{\mathrm{MQA}} =D_C+D_R =576.
$$

### 4.1 Indexer: An Inexpensive Scan over the Full History

**First distinguish the two kinds of scores.** One DSA layer contains both

$$
I:[T_Q,T_K],
$$

and

$$
A^{\mathrm{slc}}:[T_Q,H,\kappa].
$$

$I$ is the Lightning Indexer's routing score, used only to produce token addresses.
$A^{\mathrm{slc}}$ is the attention logit recomputed by the main MLA over the selected
tokens, and only this score enters the main Attention softmax.

The Indexer therefore decides *where to look*, not *how to weight what is found*.

**Shared query latent.** The normalized hidden states

$$
X:[T_Q,7168]
$$

first produce the low-rank query latent shared by the main MLA and the Indexer:

$$
C^Q = \operatorname{RMSNorm} \left( X(W^{Q,A})^{\mathsf T} \right) : [T_Q,1536].
$$

Rather than performing another large projection directly from the 7168-D hidden state,
the Indexer derives its query from $C^Q$:

$$
\widehat Q^I = C^Q(W^{IQ})^{\mathsf T} : [T_Q,1536] \rightarrow [T_Q,64\cdot128].
$$

After reshaping:

$$
Q^I:[T_Q,64,128].
$$

The released DeepGEMM interface takes a materialized FP8 Q as input. The public code
does not reveal whether the production system further fuses this projection, rotation,
and quantization.

Each historical token, by contrast, has only one shared Indexer key vector:

$$
k^I_s = \operatorname{LayerNorm} \left( W^{IK}x_s \right) \in\mathbb R^{128}.
$$

The persistent cache has logical shape

$$
K^I_{\mathrm{cache}}:[T_K,128].
$$

Thus, the 64 Indexer query heads share one historical key rather than maintaining a
separate K cache for every head.

The Indexer also projects query-dependent head weights directly from the current
hidden state:

$$
U^I
=
X(W^{IW})^{\mathsf T}
:
[T_Q,7168]
\rightarrow
[T_Q,64].
$$

$u^I_{t,j}$ is the base weight of Indexer head $j$ for query $t$; this branch does
not pass through a softmax. Absorbing the two fixed normalization factors used by the
official reference, define

$$
w^I_{t,j}
=
u^I_{t,j}H_I^{-1/2}D_I^{-1/2}.
$$

Only Indexer Q reuses $C^Q$. Indexer K and these weights are both projected directly
from $X$.

**RoPE, Hadamard, and FP8 form an execution contract.** Each 128-D Indexer Q/K vector
is split as

$$
128=64_{\mathrm{RoPE}}+64_{\mathrm{NoPE}}.
$$

The first 64 dimensions use non-interleaved RoPE, unlike the interleaved RoPE used by
the main MLA. This is not a notational detail: if the two sides use different pair
layouts, their dot product at a given position changes.

After rotation, a normalized Hadamard transform $\mathcal H_{128}$ is applied:

$$
\bar q=\mathcal H_{128}q, \qquad \bar k=\mathcal H_{128}k,
$$

$$
\mathcal H_{128}^{\mathsf T}\mathcal H_{128}=I.
$$

Before quantization, therefore,

$$
\bar q^{\mathsf T}\bar k = q^{\mathsf T}k.
$$

The Hadamard transform does not change the exact real-valued inner product; it spreads
outliers so that subsequent E4M3 quantization is more stable. “Inner-product
preserving” must not be extrapolated to mean “bitwise identical after quantization.”

> **A note on the two symbols named $H$.** $H_I=64$ is a scalar—the number of
> Indexer heads. $\mathcal H_{128}\in\mathbb R^{128\times128}$ is the Hadamard
> transform matrix. Its entries are $\pm1/\sqrt{128}$; it is fixed, orthogonal, and
> not learned. The implementation need not materialize this matrix. A fast Hadamard
> transform evaluates it through hierarchical additions and subtractions in
> $O(128\log128)$ time. Q and K use the same $\mathcal H_{128}$ after partial RoPE
> and before FP8 quantization, so the real-valued inner product is preserved exactly
> before quantization and only approximately afterward.

Let $q^{(8)},k^{(8)}$ denote the E4M3 data, and let
$s^Q_{t,j},s^K_s$ denote the corresponding scales. The official reference folds the
query scale and both normalization factors into the query-dependent weight:

$$
\widetilde w_{t,j}
=w^I_{t,j}s^Q_{t,j}
=u^I_{t,j}H_I^{-1/2}D_I^{-1/2}s^Q_{t,j}.
$$

The scorer's implementation semantics can then be written as

$$
I_{t,s} \approx s^K_s \sum_{j=0}^{63} \widetilde w_{t,j} \operatorname{ReLU} \left( \left\langle q^{(8)}_{t,j}, k^{(8)}_s \right\rangle \right).
$$

Here, $q^{(8)},k^{(8)}$ explicitly refer to vectors after partial RoPE, Hadamard
transformation, and FP8 quantization; they are no longer interchangeable with the
pre-transform $q^I,k^I$.

The logical payload of one Indexer K-cache entry is

$$
128\ {\rm B\ of\ E4M3} + 4\ {\rm B\ scale} = 132\ {\rm B/token}.
$$

Compared with a full main-KV row, this is what makes the bandwidth of the full-history
scan substantially smaller.

**Per-head routing logits must not be materialized.** Mathematically, for a fixed
query $t$, we first have

$$
R^I_t = Q^I_t \left( K^I_{\mathrm{cache}} \right)^{\mathsf T} : [64,128][128,T_K] \rightarrow [64,T_K].
$$

The global logical shape is

$$
R^I:[T_Q,64,T_K].
$$

After ReLU, weighting, and reduction across heads:

$$
I_{t,s} = \sum_{j=0}^{63} w^I_{t,j} \operatorname{ReLU} \left( R^I_{t,j,s} \right),
$$

$$
I:[T_Q,T_K]_{\mathrm{FP32}}.
$$

Writing $[T_Q,T_K,64]$ to HBM would make the low-dimensional Indexer create an
intermediate 64 times larger than its final score tensor. The released DeepGEMM scorer
therefore fuses the matrix multiplication, ReLU, query-dependent weighting, and
64-head reduction, writing only the final FP32 $I$.

One of its core tiles, expressed in the GEMM direction used by the code, can be
summarized as

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

The final step applies ReLU, query-dependent weighting, and reduction over the 64
Indexer heads, then maps candidate-major accumulator fragments directly to the logical
$[2,256]$ output addresses. There is no separate transpose stage. The intermediate
64-head result $Z$ remains on chip; HBM sees only the final score tile for two
queries and 256 candidates.

The released scorer, however, **does not include Top-K**. The public DeepGEMM pipeline
still allocates and materializes the complete rectangular carrier

$$
I:[T_Q,T_K]_{\mathrm{FP32}}.
$$

The main scorer writes the legal candidate range of each row. When
`clean_logits` is enabled, a separate cleanup kernel fills invalid and future regions.
Exact Top-K then scans the complete rectangular result again.

DeepGEMM has therefore implemented **fusion within the scorer**: FP8 matrix
multiplication, ReLU, query-dependent weighting, and 64-head reduction occur in one
kernel, and the per-head scores $[T_Q,64,T_K]$ are never written to HBM. But this is
not Indexer–TopK fusion. The released scorer still emits the full
$I:[T_Q,T_K]_{\mathrm{FP32}}$, which a separate exact Top-K implementation must scan
again.

**Mathematical Top-K and the kernel tensor live at two different levels.** For query
position $t$, let $p_t$ be the final causally visible position. Mathematically,

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

$J_{t,r}$ is a token address, not a routing score. All 128 main heads of the same
query share $J_t$, while the selected sets of different queries can be entirely
different.

The public interface also requires two shape/dtype adaptations.

First, `torch.topk` in the official inference reference returns INT64 indices and uses
`min(index_topk, end_pos)` for the fixed trailing dimension of one call; the causal
mask keeps future positions invalid.

Second, under the DSA configuration studied here, FlashMLA sparse-prefill accepts a
fixed width $\kappa=2048$:

$$
J^{\mathrm{kernel}} : [T_Q,1,\kappa]_{\mathrm{INT32}}.
$$

If a row has only $\kappa_t<\kappa$ legal tokens, all remaining slots must be replaced
by an invalid sentinel, such as `-1` or
an address no smaller than $T_K$. Repeating a legal token to fill the row is
incorrect, because the softmax would count that token more than once.

Converting the reference's INT64 Top-K result into a fixed-width INT32 kernel tensor
is therefore real data-preparation work, not a dtype annotation that can be omitted
from a shape diagram.

**The training boundary of hard Top-K.** Top-K IDs are discrete control flow. Except at
sorting boundaries, the language-modeling loss cannot provide an ordinary continuous
gradient through the decision of whether a token is selected. During dense warm-up,
the main model is frozen and the Indexer is trained with a KL loss against a target
obtained by summing dense attention over the main heads and L1-normalizing along the
sequence axis. During the sparse stage, the main model trains on the language-modeling
loss while the Indexer continues to train on a KL loss restricted to the current
selected set. The paper explicitly detaches the Indexer input, so the Indexer receives
only $\mathcal L_I$ while the main model receives only the language-modeling loss.
The main model therefore adapts to sparse selections produced dynamically by the
current Indexer, not to a fixed set of token IDs. This explains how routing is learned,
but it does not change the execution cost of exact Top-K at inference time.

### 4.2 MLA Bridge: Why the Main Kernel Is 576-D MQA

The Indexer produces addresses; the main MLA still computes the actual attention
weights. FlashMLA's input shapes become clear only after rewriting MLA from its
original MHA form into an equivalent MQA execution form.

**Start from the original score.** The shared query latent is projected by

$$
Q^{\mathrm{raw}}=C^Q(W^{QB})^{\mathsf T}:
[T_Q,1536][1536,128\times192]
\rightarrow[T_Q,128,192].
$$

Each head is split into

$$
q^C_{t,h}\in\mathbb R^{128},\qquad
q^R_{t,h}\in\mathbb R^{64},
$$

where $q^R$ receives the main MLA's interleaved RoPE. On the KV side,

$$
[C^{KV}_{\mathrm{pre}};K^R_{\mathrm{pre}}]
=X(W^{DKV})^{\mathsf T}:
[T_K,7168][7168,576]\rightarrow[T_K,576].
$$

RMSNorm turns the first 512 dimensions into $C^{KV}$, while RoPE turns the final
64 into $K^R$. The cache therefore stores

$$
C^{KV}:[T_K,512],\qquad K^R:[T_K,64].
$$

If expanded into ordinary MHA, each head would construct

$$
k^C_{s,h}=W_h^{UK}c^{KV}_s\in\mathbb R^{128},\qquad
v_{s,h}=W_h^{UV}c^{KV}_s\in\mathbb R^{128},
$$

with

$$
W_h^{UK},W_h^{UV}\in\mathbb R^{128\times512}.
$$

The resulting tensors would have shapes

$$
K^{\mathrm{MHA}}:[T_K,128,192],\qquad
V^{\mathrm{MHA}}:[T_K,128,128].
$$

Materializing them for all 128 heads would discard the cache compression before
attention even begins.

**Absorb the key up-projection into the query.** The content score satisfies

$$
(q^C_{t,h})^{\mathsf T}W_h^{UK}c^{KV}_s
=\left((W_h^{UK})^{\mathsf T}q^C_{t,h}\right)^{\mathsf T}c^{KV}_s.
$$

Define

$$
q^A_{t,h}=(W_h^{UK})^{\mathsf T}q^C_{t,h}\in\mathbb R^{512},
$$

then concatenate the RoPE component:

$$
\widetilde q_{t,h}=[q^A_{t,h};q^R_{t,h}]\in\mathbb R^{576},
\qquad
\widetilde k_s=[c^{KV}_s;k^R_s]\in\mathbb R^{576}.
$$

The QK execution shapes are now

$$
\widetilde Q:[T_Q,128,576],\qquad
\widetilde K:[T_K,1,576].
$$

The singleton head dimension means that every main query head uses the same MQA key
row.

The executed dot product is 576-D, but its scale is still defined by the original
192-D MHA score. In the released reference,

$$
m_{\mathrm{YaRN}}=1+0.1m_{\mathrm{cfg}}\ln r_{\mathrm{RoPE}},
$$

and

$$
\alpha=\frac{1}{\sqrt{192}}
\begin{cases}
m_{\mathrm{YaRN}}^2,&L_{\max}>L_{\mathrm{original}},\\
1,&\text{otherwise}.
\end{cases}
$$

Using $576^{-1/2}$ would change the model: 576 is the width of the absorbed execution
representation, not the dimension that originally defined the attention score.

**Move the value up-projection after Attention.** By linearity,

$$
\sum_sP_{t,h,s}W_h^{UV}c^{KV}_s
=W_h^{UV}\left(\sum_sP_{t,h,s}c^{KV}_s\right).
$$

The sparse MQA core can therefore consume

$$
\widetilde V_s=c^{KV}_s\in\mathbb R^{512}
$$

and first produce

$$
O^C:[T_Q,128,512].
$$

Applying each head's $W_h^{UV}$ recovers

$$
O^{\mathrm{MHA}}:[T_Q,128,128].
$$

After concatenating the heads, the layer output projection completes the path:

$$
Y=\operatorname{ConcatHeads}(O^{\mathrm{MHA}})(W^O)^{\mathsf T}:
[T_Q,16384][16384,7168]\rightarrow[T_Q,7168],
$$

where $W^O\in\mathbb R^{7168\times16384}$. FlashMLA's sparse core returns $O^C$;
the $W^{UV}$ recovery and $W^O$ projection happen outside that core.

This MQA form is an algebraic rearrangement, not a new approximation. In exact
arithmetic it computes the same score and output as the MHA form, although BF16/FP8
rounding and a different operation order need not be bit-identical. The public Python
reference expands K/V and uses MHA in its prefill branch; its decode branch exposes the
MQA rearrangement directly. Here we use the representation consumed by FlashMLA's
sparse-prefill interface, not the literal call path of that Python prefill branch.

The rewrite exposes the central reuse relation: one selected KV row can serve all 128
main heads. Algorithmic sharing is not the same as physical sharing, however. The
SM90 kernel places 64 heads in one CTA, so each loaded row is reused 64 ways in SMEM.
A second CTA reads the same IDs, but the CTAs do not share SMEM; any reuse between them
depends on the memory hierarchy and must not be assumed to be an L2 hit.

### 4.3 FlashMLA: Confining Random Rows to the SMEM Boundary

> **Takeaway.** FlashMLA does not materialize
> $K^{\mathrm{slc}}:[T_Q,\kappa,576]$ and
> $V^{\mathrm{slc}}:[T_Q,\kappa,512]$ in HBM. The producer gathers 64 discrete rows
> directly into SMEM, so the consumer sees only a regular $[64,576]$ QK tile and
> a regular $[64,512]$ latent-PV tile.

Given $J$, $\widetilde Q$, and the shared KV representation, the mathematical
program for the main Attention is short:

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

For one kernel call with fixed selection width $\kappa$, rewriting all query positions as a batched
matrix program gives the complete logical shapes:

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

These two batched matrix multiplications contract over the 576-D QK feature axis and
the selected-token axis $\kappa$, respectively. They define the algorithm's semantics,
but do not require either selected tensor to exist physically in HBM.

The real question is not the formula, but where $K^{\mathrm{slc}}$ should exist.

**Boundary of the public sparse-prefill API.** The Hopper path audited here is

```python
flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
)
```

Its release contract is

```text
q       : [TQ, HQ, 576]  BF16
kv      : [TK, 1,  576]  BF16
indices : [TQ, 1,  kappa] INT32
output  : [TQ, HQ, 512]   BF16
max     : [TQ, HQ]        FP32
lse     : [TQ, HQ]        FP32, log2-based
```

and it requires

```text
architecture = SM90
H_KV         = 1
HQ % 64      = 0
kappa > 0
kappa % 128  = 0
D_QK         = 576
D_V          = 512
```

The last dimensions of Q, KV, and indices must be contiguous. An invalid index can be
`-1` or any value no smaller than $T_K$. These constraints are the capability
boundary of the current release kernel; they cannot be generalized to arbitrary head
counts, arbitrary dimensions, or arbitrary Top-K sizes.

**Why an external gather is the wrong materialization boundary.** If a separate kernel
first gathers

$$
K^{\mathrm{slc}} : [T_Q,2048,576]_{\mathrm{BF16}},
$$

then a single query must write

$$
2048\cdot576\cdot2 = 2.25\ {\rm MiB}.
$$

Attention subsequently reads it again, so this intermediate tensor alone adds about

$$
4.5\ {\rm MiB/query}
$$

of logical global-memory traffic, before accounting for indices, Q, output, or cache
misses. This 4.5 MiB estimate assumes the best external layout: gather only the fused
row $[C^{KV};K^R]$ and treat $V=C^{KV}$ as a view of its first 512 dimensions. A
separate K tensor and V tensor would move even more data. The source-KV read is common
to both designs and is not included in this comparison; cache hits may also prevent
all logical bytes from reaching DRAM.

FlashMLA never creates this tensor in HBM. It moves only the 64 discrete rows needed
for the current iteration into a regular

$$
KV^{\mathrm{SMEM}}_{\mathrm{tile}} : [64,576]
$$

tile, which the Tensor Core consumer then treats as an ordinary two-dimensional
matrix.

**CTA and warpgroup ownership.** The SM90 sparse-prefill release uses

$$
B_H=64, \qquad B_{\kappa}=64, \qquad N_{\mathrm{threads}}=384.
$$

Its launch grid is

$$
\mathrm{grid.x} = T_Q(H/64).
$$

One CTA has fixed ownership of

$$
Q_t^{\mathrm{CTA}} : [64,576],
$$

$$
J_t:[\kappa],
$$

$$
O_t^{\mathrm{CTA}} : [64,512].
$$

The 128 main heads therefore use two CTAs, both reading the same $J_t$.

The 384 threads form three 128-thread warpgroups:

| Warpgroup | Role |
|---|---|
| WG0 | Consumer: even 64-token chunk; owns output $0{:}256$ |
| WG1 | Consumer: odd 64-token chunk; owns output $256{:}512$ |
| WG2 | Producer: reads indices and moves indirect KV rows into SMEM |

When $\kappa=2048$,

$$
2048/64=32
$$

selected-token chunks are organized into 16 even/odd paired iterations.

**The row base is random; the 576 elements within a row are not.** One BF16 KV row
occupies

$$
576\cdot2 = 1152\ {\rm B}.
$$

$J_t[r]$ determines the base address of the next row. Adjacent selected rows may
reside in entirely different cache lines, pages, or HBM partitions. Once that base
address is known, however, the row's 1152 bytes are contiguous.

The producer is therefore not performing 576 unrelated random scalar loads. In the
SM90 source, the logical 1152-B row is actually moved as nine 64-BF16 feature slabs,
each 128 B; eight threads issue 16-B `cp.async` copies for each slab. It is not one
1152-B bulk transaction. Conceptually, the producer performs

```text
randomly determine the row base
→ cooperatively move a contiguous row with multiple threads
→ write it into swizzled SMEM
→ let the consumer read a regular [64,576] tile
```

The irregularity is confined to the global-memory-hierarchy-to-SMEM boundary; the internal shapes of QK
and PV become regular matrices again.

**The two consumers partition two different axes.** WG0 and WG1 both compute the full
QK feature contraction:

$$
[64,576][576,64] \rightarrow [64,64].
$$

They differ along the token axis for QK: WG0 processes the even chunk, while WG1
processes the odd chunk.

PV ownership is instead split along the value-feature axis:

$$
V = V_L\oplus V_R,
$$

$$
V_L,V_R : [64,256].
$$

| WG | Local probability | Resident FP32 accumulator | Contributions from local/remote $P$ |
|---|---|---|---|
| WG0 | $P_0$ | $O_L:[64,256]$ | $P_0V_{0L}+P_1V_{1L}$ |
| WG1 | $P_1$ | $O_R:[64,256]$ | $P_1V_{1R}+P_0V_{0R}$ |

Each warpgroup publishes one probability tile to the other. One even/odd pair therefore
exchanges two tiles,

$$
2\times P:[64,64]_{\mathrm{BF16}} = 16\ {\rm KiB},
$$

rather than moving their approximately 64 KiB FP32 output accumulators.

Online softmax still maintains $m,z,O_{\mathrm{acc}}$. WG0 first publishes the max
from the even chunk, and WG1 merges the odd chunk to establish a pair-global max
baseline. The two warpgroups rescale against this common max, but each continues to
accumulate its own normalizer until the epilogue combines them in `reduce_L()`; the
two output halves are reduced there as well. The pipeline changes ownership, not the
mathematical softmax over selected tokens.

**Half-buffer pipeline.** This is not a simple ping-pong between two complete
$[64,576]$ tiles. `plan.k[0]` and `plan.k[1]` hold the even and odd 64-token chunks
of the current pair, respectively. Each chunk is further divided along the feature
axis into $[0:256]$ and $[256:576]$, with a separate ready/free barrier for each
half. The producer moves one pair of chunks in this order:

```text
even[0:256]
→ odd[256:576]
→ even[256:576]
→ odd[0:256]
→ next pair
```

As soon as a consumer finishes using one half, it publishes the corresponding free
barrier, allowing the producer to overwrite that half in place with data from the
next pair. In steady state, this still creates the overlap

```text
compute pair i
        ||
gather pair i+1
```

but the pipeline operates at the granularity of four feature half-buffers rather than
by exchanging the roles of two complete tiles. It reduces the visible latency of
waiting for the next set of discrete rows to reach the Tensor Cores; it neither
reduces the number of KV bytes that must be read nor changes the randomness of the
selected row addresses. If L2/TLB/HBM latency exceeds the window that the computation
of the current pair can cover, the consumer still stalls.

**How shared selection becomes locality.** Within one CTA, every 576-D KV row is
reused by 64 query heads. Counting only the QK and latent-PV work performed with that
row, the local arithmetic is approximately

$$
2\cdot64\cdot(576+512) = 139264\ {\rm FLOPs}.
$$

Dividing by the 1152 B useful row payload gives

$$
\frac{139264}{1152} \approx 121\ {\rm FLOP/B_{useful}}.
$$

This explains why “arbitrary token selection” does not necessarily force the consumer
into low-intensity random memory access. The algorithm makes 64 heads share the
address, and the kernel turns the same row into an on-chip tile with high reuse.

But 121 FLOP/B is only CTA-local intensity, not an end-to-end roofline number. It does
not include

- the Indexer and Top-K;
- reads of the index tensor;
- Q, output, and softmax state;
- L2/TLB misses;
- cache-line granularity and the 256-B L2 prefetch hint;
- cache reuse between the two CTAs, which is not guaranteed;
- launch, scheduling, and fixed service overheads.

**Experiment: how much does KV-row locality still matter?** The decomposition above
suggests a testable prediction. FlashMLA does not remove random row addresses, but
their effect on end-to-end kernel latency may be small after 64-head reuse, SMEM
repacking, and producer–consumer overlap. The experiment below measures that response;
it does not isolate which mechanism causes it.

We ran the unmodified official SM90 sparse-prefill kernel at FlashMLA commit
`3969f20` on one NVIDIA H100 PCIe, using

$$
Q:[128,128,576]_{\mathrm{BF16}},\qquad
KV:[16{,}777{,}216,1,576]_{\mathrm{BF16}},\qquad
J:[128,1,2048]_{\mathrm{INT32}}.
$$

Each query receives a private, non-overlapping 128K-row KV arena. This removes
cross-query overlap in selected row IDs, so one condition cannot accidentally gain
more cross-query same-row cache reuse than another. It does **not** remove possible L2
reuse between the two 64-head CTAs serving the same query, because those CTAs request
the same selected rows.

Each condition contains 10 interleaved measurement blocks with 20 matched rounds per
block, giving 200 per-call CUDA-event timings. Sixteen distinct conditions produce
3,200 observations under the same eight-slot rotating schedule. All 34 official
sparse-prefill tests, a 16-condition correctness gate, and the validation checks
passed. `[MEASURED]`

![KV-row ordering and concentration in FlashMLA sparse-prefill](/images/sparse-attention-kv-row-locality.png)

*Figure 1: The focused vertical scale makes sub-percent effects visible. The left
panel holds the selected-row set fixed and changes only its ascending, descending, or
eight shuffled orders. The right panel changes the sampling window $W$; each $W$
resamples both the window origin and the exact IDs, so the line is only a visual guide,
not a trajectory of one fixed selected set. Error bars are pointwise 95% intervals
from a paired measurement-block bootstrap with 5,000 resamples. They are neither a
population interval over all permutations nor a family-wise corrected interval.*

**First, change only the order.** For every query and rotating slot, the left panel
holds the same 2,048-row set fixed and arranges it in ascending address order,
descending order, or eight independently seeded random permutations.

Ascending order has a median latency of $291.680\,\mu s$; descending order has
$290.304\,\mu s$. Their normalized ratio is $0.9953$, with a paired 95% interval of
$[0.9906,1.0046]$. The point estimates differ by $0.47\%$, but the interval includes
1.

Across the eight shuffle seeds, ratios range from $0.9953$ to $0.9990$, with a
seed-level median of $0.9976$ and IQR of $0.0030$. Every shuffle's pointwise timing
interval crosses 1. At this fixed kernel configuration, we therefore detect no
latency penalty for those eight random orders relative to sorted addresses. The
intervals quantify timing uncertainty for each fixed permutation. They do not describe
the population of all possible permutations: only the eight seeds are independent
units for permutation-to-permutation variation, while the 200 timings per seed are
repeated measurements.

The near-null response is not caused by the shuffled indices remaining locally
similar. With sorted addresses, the median adjacent-row distance is 44 and a
64-index tile touches a median of 41 distinct 64-row regions. Under
`W=128K, shuffle-00`, these become 38,435 and 63, while the fraction of adjacent
accesses in the same region falls from $36.49\%$ to $0.046\%$. Software-visible
locality changes by orders of magnitude; latency barely changes.

**Next, change the concentration of the selected set.** The right panel fixes
$\kappa=2048$, uniform sampling without replacement, and the `shuffle-00` rank
permutation rule, then samples rows from 64-row-aligned windows

$$
W\in\{2K,4K,8K,16K,32K,64K,128K\}.
$$

This changes the selection distribution: each $W$ resamples both the window origin
and the exact selected IDs. It is therefore not a paired ablation in which one token
set merely changes span. Only `W=128K, shuffle-00` shares exactly the same index corpus
with the left panel and serves as the baseline.

The 128K baseline has a median latency of $291.360\,\mu s$. Relative to it, 2K and
4K windows are faster by $0.906\%$ and $0.818\%$, absolute differences of about
$2.64\,\mu s$ and $2.38\,\mu s$. Their pointwise 95% intervals are

$$
[0.9852,0.9973],\qquad[0.9869,0.9992].
$$

The other intervals include 1. Six window-versus-128K comparisons were reported
without a family-wise correction, and the 4K upper bound of $0.9992$ is already close
to the boundary. In an unregistered Bonferroni-bootstrap sensitivity check, the upper
bounds become about $0.9993$ for 2K and $1.0018$ for 4K. We therefore treat 2K as the
clearer tight-window signal and 4K as weaker, exploratory evidence.

Meanwhile, the number of regions touched per tile rises with $W$ from 28, 41, 51,
57, 60, and 62 to 63, but latency is not monotonic: 64K is faster than 8K–32K, while
32K is slightly slower than 128K. Those cross-window relations were not additional
direct pairwise tests. The data do not establish a monotonic locality–latency law, a
threshold, or a universal significance claim.

**Sensitivity to measurement start-up.** The raw timings contain a transient aligned
with the beginning of each measurement block. The first timed call in all 10 blocks is
$1.57$–$1.64\times$ its condition median, and 12 of 3,200 observations exceed their
condition median by more than $1.2\times$. The primary analysis follows the admission
rule fixed in advance and does not remove observations after seeing the data; paired
randomization and median estimates limit the effect of these few long-tail points.

As a post-hoc sensitivity check, removing `matched_round=0` from every block changes
the 2K/4K ratios from $0.9909/0.9918$ to $0.9939/0.9943$, shrinking the estimated
benefits from about $0.91\%/0.82\%$ to $0.61\%/0.57\%$. The recomputed pointwise
95% intervals remain below 1. The direction survives, but the exact magnitude depends
somewhat on measurement phase, so these remain sub-percent signals rather than stable
hardware constants.

Profiler captures are only supplementary diagnostics. Relative to one
`W=128K, shuffle-00` capture, a $W=2K$ capture reports $2.17\%$ fewer NCU DRAM
reads, $1.64\%$ fewer device-read sectors, and NCU/Nsys durations shorter by about
$0.96\%$ and $1.31\%$. But L2 hit rate, long-scoreboard stalls, and Tensor-pipe
activity do not form a consistent monotonic chain over $W$. Four representative
points execute the same number of GMMA instructions and achieve $18.37\%$–$18.38\%$
occupancy, controlling the dominant Tensor Core work and static execution shape but
not address generation, cache transactions, or stalls. Each condition has only one
NCU/Nsys kernel instance; NCU uses replay without cache clearing or clock locking.
These counters cannot provide a causal explanation with variance.

The strongest conclusion is therefore not that “random access is free,” but:

> **For the eight random permutations tested at this fixed H100/SM90 sparse-prefill
> configuration, we detect no latency penalty relative to ascending addresses; every
> point estimate differs by at most $0.47\%$. Latency is not monotonic over the window
> sweep. A 2K window shows the clearer sub-percent benefit, while the 4K result lies
> closer to the statistical boundary.**

This is consistent with 64-head reuse, SMEM repacking, and the feature-segment
pipeline hiding most of the locality difference. The experiment does not disable any
of those mechanisms, however, so it cannot identify which one causes the small
response.

The scope is equally important. The indices are synthetic samples without replacement;
private arenas deliberately remove popular tokens and cross-query reuse that real
workloads may contain. The result covers only $T_Q=128$, $\kappa=2048$, BF16 576-D
rows, $D_V=512$, and one H100 PCIe. It is not a measurement of the complete
DeepSeek-V3.2 model, production DSA, real Indexer traces, or sparse decode. The test
also uses $1/\sqrt{576}$ as a fixed kernel stimulus rather than the model's
192-D-plus-YaRN scale.

This controlled experiment therefore demonstrates low sensitivity to row locality in
the tested configuration, not guaranteed behavior in production. External validation
requires replaying real Indexer traces through the same kernel and checking whether
their adjacent distances, regions per tile, cross-query overlap, and latency fall
inside the range covered here.

**Prefill and decode use different kernels.** The microarchitecture and locality
experiment above concern BF16, non-paged SM90 sparse-prefill. The separately measured
single-query H100 path uses paged FP8 sparse-decode.

At the fixed $B=1$ shape, it uses

```text
q                : [1, 1, 128, 576] BF16
main KV cache    : [num_blocks, 64, 1, 656] mixed-byte packing
physical indices : [1, 1, 2048] INT32
page size        : 64 tokens
latent D_V       : 512
```

Each 656-B cache row consists of

$$
512\ \mathrm{B\ FP8\ NoPE}
+16\ \mathrm{B\ FP32\ scales}
+128\ \mathrm{B\ BF16\ RoPE}
=656\ \mathrm{B};
$$

it is not entirely FP8. `physical indices` are not the logical token IDs emitted by
the Indexer, either: a page table has rewritten them into paged-cache offsets. The
Top-K multiple-of-64 constraint, split-KV scheduler, and combine kernel also differ
from sparse-prefill. The two kernels implement the same selected-token attention
semantics under different serving conditions, but their cache shapes, metadata costs,
and one-kernel conclusions are not interchangeable.

**On Blackwell, FlashMLA moves the indirect gather into TMA.** The following structure
is specific to the audited FlashMLA commit
[`9241ae3`](https://github.com/deepseek-ai/FlashMLA/tree/9241ae3ef9bac614dd25e45e507e089f888280e0).
For $H_q=128,\kappa=2048,D_{QK}=576$, the public dispatcher selects the regular SM100
`head128_k576` path—not `head64`. Prefill `head64` requires the full model to have
$H_q=64$, while the small-Top-K head128 specialization supports only $D_{QK}=512$.

SM90 maps one 128-head query to two independent 64-head CTAs. SM100 instead uses

$$
B_H=128,\qquad B_{\mathrm{kv}}=128,
$$

and launches

$$
\mathrm{grid.x}=2T_Q,\qquad\mathrm{clusterDim.x}=2,
$$

forming one two-CTA cluster per query. Each CTA still handles 64 query heads, but the
K producer uses `TMA tile::gather4` with `cta_group::2`.

One `gather4` instruction describes a 64-D BF16 feature slab from four arbitrary rows,
for a useful payload of

$$
4\times64\times2=512\ {\rm B}.
$$

Within each 128-token chunk, each CTA owns 64 selected K rows. A $[64,64]$ K slab
therefore requires 16 `gather4` instructions per CTA, and the full 576-D K path has
nine 64-D slabs, including the RoPE slab. Unlike the standalone `head64_k576`
specialization, this actual head128 path does not move the RoPE tail through a separate
`cp.async` path.

Separate V producers cover all 128 selected rows in both CTAs while splitting the
512 value dimensions into 256 dimensions per CTA. The cluster therefore cooperates
over the token axis for K and partitions the value-feature axis for V, matching its
later two-CTA QK/PV ownership.

The source shows that `gather4` moves four-row address generation, bulk-copy issue,
and completion tracking into TMA and can place data directly in swizzled SMEM. It does
not show an end-to-end latency reduction by itself, and the four global-memory rows
may still be physically far apart. This SM100 path is also outside what the H100/SM90
measurements in this article validate.
### 4.4 The Hidden Quadratic Term: A Sparse Main Kernel Does Not Make a Linear System

> **Takeaway.** FlashMLA confines the expensive main Attention computation to a fixed Top-$K$, but
> the exact Lightning Indexer still examines the entire candidate history for every query. Therefore,
> the sparse core is $\Theta(L\kappa)$, while full causal prefill still retains a low-constant
> $\Theta(L^2)$ routing term.

FlashMLA accesses only the selected 2,048 tokens, but DSA must first discover them. As long as the routing score still depends on every query–candidate pair, the Indexer retains a full-history scan.

In what follows, $\mathcal W$ counts only the dominant dot/QK/PV operations and omits
the factor of 2 for FMA common to every expression. These are shape-derived useful-MAC
counts, not equivalent units of time across different dtypes and kernels.

**One mature query.** Given a history of length $L$, the Indexer cost is:

$$
\mathcal W_{\mathrm{Indexer}}(L) = H_I D_I L = 64\cdot128\cdot L = 8192L.
$$

The selected main MQA core costs:

$$
\mathcal W_{\mathrm{sparse}}(L) = H(D_{QK}^{\mathrm{MQA}}+D_C) \min(L,\kappa),
$$

$$
\mathcal W_{\mathrm{sparse}}(L) = 128(576+512) \min(L,2048),
$$

$$
\mathcal W_{\mathrm{sparse}}(L) = 139264 \min(L,2048).
$$

Therefore:

$$
\mathcal W_{\mathrm{DSA}}(L) = 8192L + 139264\min(L,2048).
$$

The first term keeps growing with history length; the second saturates once $L\ge2048$.

**Full causal prefill.** Now let $N$ denote the complete causal sequence length and
sum over query positions $t=1,\ldots,N$:

$$
\mathcal W_{\mathrm{Indexer,full}}(N)
=\sum_{t=1}^{N}8192t
=8192\frac{N(N+1)}2
=\Theta(N^2).
$$

The exact sum for the sparse core is:

$$
\mathcal W_{\mathrm{sparse,full}}(N)
=139264\sum_{t=1}^{N}\min(t,\kappa).
$$

When $N\ge\kappa$:

$$
\sum_{t=1}^{N}\min(t,\kappa)
=\frac{\kappa(\kappa+1)}2+(N-\kappa)\kappa.
$$

Hence:

$$
\mathcal W_{\mathrm{sparse,full}}(N)=\Theta(N\kappa)\qquad(N\ge\kappa).
$$

For fixed $\kappa=2048$:

$$
\boxed{
\mathcal W_{\mathrm{DSA,full}}(N)
=\Theta(N^2)+\Theta(N\kappa)
=\Theta(N^2)
}.
$$

DSA does not turn quadratic Attention into asymptotically linear Attention. It replaces:

$$
\text{high-dimensional, 128-head quadratic main Attention}
$$

with:

$$
\text{low-dimensional FP8 quadratic routing} + \text{high-dimensional but fixed Top-K main Attention}.
$$

This is a highly valuable reconstruction of constants, not a change in asymptotic complexity.

**How large a quadratic coefficient does it replace?** Using the original 192-D MHA score and 128-D MHA value as a prefill-style baseline:

$$
\mathcal W_{\mathrm{dense,MHA}}(L) = H(192+128)L = 40960L.
$$

The coefficient of the Indexer's quadratic term is:

$$
\frac{8192}{40960} = \frac15.
$$

If we instead compare against single-query dense MQA decode, which scans a full 576-D key and a 512-D latent value:

$$
\mathcal W_{\mathrm{dense,MQA}}(L) = H(576+512)L = 139264L.
$$

The coefficient of the Indexer's linear scan is smaller by:

$$
\frac{139264}{8192} = 17.
$$

These ratios also imply a bandwidth shift. The dominant per-candidate KV payload in
the full-history scan is only the shared 128-D FP8 Indexer key and its scale, although
the Indexer query, head weights, and score-carrier writes remain. The full 576-D main
KV representation is read only for selected tokens.

**Endpoint and full-causal accounting must be separated.** The endpoint model considers only one query facing a complete history of length $L$; the full-causal model sums over every query position in the sequence.

Let $\kappa=2048$ and $L\ge\kappa$. The prefill-style baseline and DSA endpoint costs are respectively:

$$
\mathcal W_{\mathrm{dense,endpoint}}(L)=40960L,
$$

$$
\mathcal W_{\mathrm{DSA,endpoint}}(L)=8192L+139264\kappa.
$$

The full-causal baseline is:

$$
\mathcal W_{\mathrm{dense,full}}(N)=40960\frac{N(N+1)}2.
$$

The full-causal DSA cost is:

$$
\mathcal W_{\mathrm{DSA,full}}(N)
=8192\frac{N(N+1)}2
+139264\left[\frac{\kappa(\kappa+1)}2+(N-\kappa)\kappa\right].
$$

These expressions yield the following analytical crossover points:

| Accounting regime | Operation-count relationship | $L$ |
|---|---|---:|
| Endpoint, prefill-style MHA baseline | DSA equals dense | 8,704 |
| Full causal prefill, MHA baseline | DSA equals dense | 16,315 |
| Single-query dense MQA decode | DSA equals dense | 2,176 |
| Endpoint, internal to DSA | Indexer equals sparse core | 34,816 |
| Full causal, internal to DSA | Cumulative Indexer equals cumulative sparse core | $\approx68{,}592$ |

These values answer only one question: at what length do the two arithmetic terms cross under the given shape-based accounting? They do not include:

- projection, RoPE, Hadamard transforms, or quantization;
- throughput differences among FP8, BF16, and Tensor Cores;
- FP32 logits materialization;
- exact Top-K and index rewriting;
- the FlashMLA scheduler, split-KV, and combine;
- communication, launch, cache, or serving fixed overhead.

Therefore, 8,704, 16,315, 2,176, 34,816, and 68,592 are not latency
crossovers or speedup guarantees. The scaling experiment below uses the same endpoint
and full-causal viewpoints, but measures the released discovery chain including scorer,
Top-K, and index transformation. The values 34,816 and approximately 68,592 provide
only an arithmetic scale reference; they do not predict the measured $[32K,64K]$
checkpoint bracket. The first three dense-baseline comparisons only quantify the
useful-MAC coefficient that DSA replaces.

**The released implementation still materializes a complete routing-score matrix.**
If full prefill is represented as one tensor,

$$
I:[L,L]_{\mathrm{FP32}},
$$

then at $L=128\mathrm{Ki}$, the logical payload of the final routing logits alone is:

$$
(131072)^2\cdot4 = 64\ {\rm GiB}.
$$

This is the shape-derived logical payload of one-shot materialization, not the actual
peak footprint of the experiment below and not the pipeline's full memory requirement.
The public prefill scorer supports query chunking. Let the chunk size be
$C=4096$, the final processed length be $N=mC$, and the right endpoint of chunk $j$
be $e_j=jC$. Ignoring bounded tile rounding from `BLOCK_Q=2` and
`BLOCK_KV=256`, the scorer pairs with genuine causal semantics are:

$$
P_{\mathrm{causal}}(N)=\sum_{t=1}^{N}t=\frac{N(N+1)}2.
$$

But the independent Top-K stage receives a rectangular carrier for each chunk:

$$
E_{\mathrm{carrier}}(N)
=
\sum_{j=1}^{m}C(jC)
=
\frac{N(N+C)}2.
$$

Their difference is the future triangle within each chunk that must be cleared to $-\infty$:

$$
E_{\mathrm{future}}(N)
=E_{\mathrm{carrier}}(N)-P_{\mathrm{causal}}(N)
=\frac{N(C-1)}2.
$$

Here $E_{\mathrm{future}}$ counts only elements actually written as $-\infty$; it
does not describe all work issued by the cleanup kernel. The released cleanup kernel
still traverses the full carrier for every row in blocks of
$B_{\mathrm{clean}}=8192$. If $\mathcal B_{\mathrm{cleanup}}(N)$ counts the
accumulated cleanup blocks visited by all query rows, then

$$
\mathcal B_{\mathrm{cleanup}}(N)
=\sum_{j=1}^{m}C
\left\lceil\frac{jC}{B_{\mathrm{clean}}}\right\rceil
=\Theta\!\left(\frac{N^2}{B_{\mathrm{clean}}}\right).
$$

For fixed $C$, the number of invalid values written is $\Theta(NC)=\Theta(N)$,
but the cleanup traversal is not linear. The released timing API groups scorer and
cleanup together, and the experiment preserves that boundary. Therefore:

$$
\begin{aligned}
\text{causal scorer useful pairs/MACs} &:\Theta(N^2),\\
\text{Top-K carrier scan} &:\Theta(N^2),\\
\text{future cleanup stores} &:\Theta(NC)=\Theta(N),\\
\text{cleanup block traversal} &:\Theta(N^2/B_{\mathrm{clean}}),\\
\text{fixed-}\kappa\text{ FlashMLA} &:\Theta(N\kappa)=\Theta(N).
\end{aligned}
$$

In the last chunk at 2M, the released chain materializes the logical view

$$
I_j:[4096,2097152]_{\mathrm{FP32}}=32\ {\rm GiB},
$$

with another 4 MiB in the DeepGEMM backing allocation because of 256-token stride
padding. Across the 4K-to-2M replay, the rectangular carriers account for about
8.016 TiB of cumulative logical writes. They need not remain live simultaneously,
although the allocator may retain previous reservations. Chunking turns one square
live tensor into a rectangular tile; it does not change the quadratic cumulative
growth of scorer work or Top-K carrier traffic.

**A conditional lower bound for exact Top-K.** Suppose query-dependent routing scores are
treated as an oracle with no additional structure, and there is no candidate-specific bound
that can safely eliminate an unexamined candidate. Then exact global Top-$\kappa$ must,
in the worst case, inspect all $L$ candidates.

Otherwise, for any skipped candidate $s$, one could construct an input that leaves all
inspected scores unchanged but makes $I_{t,s}$ larger than the current $\kappa$-th largest
value. The selector therefore cannot prove that its result is exact.

Under these explicit assumptions:

$$
\text{single query} : \Omega(L) \ \text{score evaluations},
$$

$$
\text{full causal prefill} : \Omega(L^2) \ \text{score evaluations}.
$$

Chunking, streaming Top-K, and kernel fusion can reduce peak memory, HBM round trips, and
launches, but they do not automatically reduce the number of candidates that must be judged.
Changing the asymptotic term requires approximate retrieval, a provably valid hierarchical
candidate structure, or reuse of a smaller candidate set across queries.

**Experiment: when does Discovery's quadratic term become visible after the main
attention is sparse?** Operation counts describe the algorithm, but GPU latency also
depends on dtype, kernel efficiency, intermediate tensors, and memory traffic. We ask
one narrow, directly measurable question:

> As full causal prefill length $N$ grows in the fixed released pipeline, when does
> Discovery overtake fixed-$\kappa$ FlashMLA sparse attention?

We did not implement or measure an Indexer–TopK fused kernel. The measured object is
this released **unfused** chain:

```text
DeepGEMM FP8 scorer
→ causal cleanup
→ FP32 score carrier
→ PyTorch exact Top-K
→ invalid-index padding / INT32 cast
→ FlashMLA sparse prefill
```

Define

$$
T_{\mathrm{Discovery}}
=T_{\mathrm{scorer+cleanup}}
+T_{\mathrm{TopK}}
+T_{\mathrm{pad/cast}},
\qquad
T_{\mathrm{SparseMain}}=T_{\mathrm{FlashMLA}}.
$$

For a given query chunk, FlashMLA cannot start until the Top-K IDs exist; Discovery
is not a side path that can be freely overlapped with that chunk's main attention.

The experiment fixes $B=1,C=4096,\kappa=2048$ on one H100 PCIe (SM90). After two
warm-up replays, the direct full-chain total uses 20 **repeated** clean replays measured
by start-to-checkpoint CUDA events. Stage attribution comes from five separate
instrumented replays, summing all chunk events within each trial. These are different
measurement trajectories; stage medians are never added to impersonate the direct
total.

This is a shape-controlled synthetic chain: the same 4K Indexer query, weights, and
main query are replayed repeatedly, reaching 512 chunks at 2M, and selections do not
come from a real model trace. The experiment controls shapes and execution paths but
does not reproduce the score distribution of semantic model inputs. Its crossover
therefore applies only to this fixed public component chain, not to production
DeepSeek DSA.

The CUDA-event medians are:

| Processed length $N$ | Discovery cumulative | FlashMLA sparse prefill | Direct full-chain total | Discovery / FlashMLA |
|---:|---:|---:|---:|---:|
| 32K | 0.0307 s | 0.0422 s | 0.0728 s | $0.73\times$ |
| 64K | 0.1135 s | 0.0896 s | 0.2026 s | $1.27\times$ |
| 128K | 0.4255 s | 0.1861 s | 0.6135 s | $2.29\times$ |
| 256K | 1.670 s | 0.394 s | 2.066 s | $4.23\times$ |
| 512K | 6.607 s | 0.813 s | 7.421 s | $8.13\times$ |
| 1M | 27.454 s | 1.655 s | 29.098 s | $16.59\times$ |
| 2M | 113.149 s | 3.367 s | 116.490 s | $33.60\times$ |

The Discovery and FlashMLA columns are same-trial chunk-event sums from five
instrumented replays. Direct total comes from the separate 20 repeated clean replays.
Because the trajectories differ, the first two medians should not be added or required
to equal the direct-total median.

Thus, 64K is the first prespecified discovery-dominant checkpoint. The strict conclusion can
only be written as:

$$
\boxed{\text{prespecified checkpoint bracket}=[32K,64K]},
$$

and 64K must not be called a continuous, exact crossover. `[MEASURED]`

To distinguish finite-range scaling from one-point noise, we compute a local effective
exponent within each of the five paired attribution trajectories:

$$
\alpha_s(N)=
\operatorname{median}_{r=1,\ldots,5}
\log_2\frac{T_{s,r}(2N)}{T_{s,r}(N)}.
$$

| Interval | $\alpha_{\mathrm{Discovery}}$ | $\alpha_{\mathrm{SparseMain}}$ |
|---:|---:|---:|
| 128K→256K | 1.972 | 1.083 |
| 256K→512K | 1.984 | 1.045 |
| 512K→1M | 2.055 | 1.025 |
| 1M→2M | 2.043 | 1.025 |

From 256K to 2M, Discovery grows $67.75\times$, an endpoint effective exponent of
$2.027$; FlashMLA sparse-prefill grows $8.54\times$, an exponent of $1.031$.
These exponents describe five fixed-input timing trajectories over a finite range.
They are not OLS fits, population estimates over model inputs, or experimental proofs
of asymptotic complexity. They are consistent with the predicted near-quadratic
all-candidate Discovery and near-linear fixed-$\kappa$ sparse main attention.

![Cumulative causal-prefill scaling and Discovery dominance](/images/sparse-attention-discovery-scaling.png)

*Figure 2: The left panel uses same-trial cumulative chunk-event sums from five
instrumented replays, not the direct totals from 20 clean replays; the right panel
plots their ratio. Checkpoints were fixed in advance and lines are visual guides.
The shaded $[32K,64K]$ region is a discrete checkpoint bracket, not interpolation or
a confidence interval.*

**4M is limited by capacity, so it has no latency measurement.** With
$C=4096,\kappa=2048$ and the current unfused PyTorch Top-K path fixed, a strict
preflight lower bound contains 64.004 GiB for the DeepGEMM carrier backing, 64 GiB
for a Top-K contiguous copy, and 5.626 GiB of resident tensors:

$$
133.630\ {\rm GiB}>79.180\ {\rm GiB}.
$$

The run was therefore marked capacity-limited before allocation. We report neither a
fabricated latency nor a measured OOM. This does not mean that DSA has a universal 2M
context limit: a smaller $C$, a different Top-K implementation, or scorer–selector
fusion could move the capacity boundary.

**Nsys at long sequence lengths explains launch structure, not the timing curve.** For
one target chunk, TensorIterator splits one semantic contiguous copy inside PyTorch
Top-K into approximately 1-GiB shards: eight launches at 512K and 32 at 2M. After
excluding these copy shards, the Top-K kernel-family sequence and launch count remain
15, but the grids of radix, scan, gather, and related kernels grow with $N$. It would
therefore be wrong to say that the whole Top-K topology is unchanged.

In the same diagnostic captures, target-chunk attention kernel-active time changes
only from $6.465\ {\rm ms}$ at 512K to $6.746\ {\rm ms}$ at 2M, while the
discovery/attention ratio rises from $16.50\times$ to $65.32\times$. These Nsys
durations support the mechanism's direction; they must not be spliced into the
cumulative CUDA-event timing curve.

**The 128K NCU data show where the traffic comes from.** The 2 GiB FP32 carrier
for the final query chunk at 128K causes the scorer to write approximately 1.958 GiB to DRAM
in NCU; Top-K then reads 12.064 GiB—$6.03\times$ the carrier size—and writes
2.180 GiB. This traffic comes from one contiguous copy, four rounds of radix thresholding,
and count, scan, gather, and related steps—not from “one read of logits and one Top-K kernel.”
The Top-K stage for the 128K target chunk contains 17 CUDA kernels; the complete 4K→128K
replay contains 513 Top-K kernels.

But the launch count itself is not the main loss. Across three full Nsys replays, GPU gaps
inside the kernel span account for only $0.283\%$–$0.289\%$. Therefore, what fusion
could truly eliminate is:

$$
\boxed{
\text{FP32 logits write}
+
\text{Top-K repeated global-memory scans}
},
$$

rather than converting the raw number of CPU launches directly into an assumed GPU speedup.

**The measured facts end here.** What follows is a design inference constrained by source code
and resource limits, not a fused-kernel performance result. The current DeepGEMM mapping
assigns two query rows to one persistent CTA, which iterates over all of their KV tiles
internally. If this ownership is preserved, exact Top-2048 can be organized as online state
across tiles within the CTA and does not inherently require a cross-CTA global merge. The score
and ID payload alone is:

$$
2\ \text{queries}\times2048\times
(\text{FP32 score}+\text{INT32 id})
=32\ {\rm KiB}.
$$

But this 32 KiB is not free. The current scorer already uses 640 threads, 96 registers/thread,
and 152,228 B of dynamic shared memory, running one CTA per SM. The H100 opt-in SMEM limit is
232,448 B; after subtracting the candidate payload and approximately 4 KiB for the
double-buffered score tile, only about 42 KiB of nominal space remains for merge scratch,
barriers, and alignment. Inserting the selector consumer into the main loop could also cause
register spills, SMEM contention, or producer backpressure, thereby degrading the existing
WGMMA/TMA scorer.

The measurements therefore identify a design opportunity, not a measured speedup:

> **Scorer–TopK fusion may eliminate the HBM round trip for complete logits, but it does not
> reduce the number of candidates that an exact selector must judge, nor has it yet been shown
> that maintaining the selector state on chip costs less than the traffic it eliminates.**

We also measured the endpoint cost of generating one new token with paged decode, up
to a 4M-token history. The observed discovery/decode ratio is $5.21\times$–$5.23\times$
from 256K through 1M, then rises to $12.79\times$ at 2M and $17.88\times$ at 4M;
the direct 4M full-chain total is $0.675808\ {\rm ms}$. These timings are not stationary:
there is a clear event-latency regime shift between trials 0–4 and 5–99, and the sparse-
decode event median itself drops from about $0.073\ {\rm ms}$ at 1M to
$0.036\ {\rm ms}$ at 2M.

The 4M and earlier 128K profiles have the same visible attention launch signature—
kernel identity, grid, block, and launch count—so the evidence does not explain the
step as a visible dispatch change. It still cannot exclude internal branches, cache,
clock, or other runtime state. We therefore use the decode data only to establish the
direction of stage dominance. Neither $17.88\times$ nor any absolute latency is treated
as a stable hardware constant, and the paged FP8 decode curve is not joined to the
BF16 causal-prefill curve above.

### 4.5 DSA Makes the Main Attention Sparse—and Pushes the Bottleneck Toward Discovery

DSA is not simply “run Top-K, then call ordinary Attention.” The complete co-design chain is:

$$
\boxed{ \begin{aligned} &\text{shared, low-dimensional FP8 Indexer scans the entire history}\\ &\rightarrow \text{exact Top-K token IDs for each query}\\ &\rightarrow \text{128 main heads share one selection set}\\ &\rightarrow \text{fused discrete-row gather within the CTA}\\ &\rightarrow \text{reconstruct a regular }[64,576]\text{ tile in SMEM}\\ &\rightarrow \text{Tensor Core QK, online softmax, and latent PV}. \end{aligned} }
$$

The algorithm exposes KV-row reuse across heads; the kernel confines address irregularity to
the HBM→SMEM boundary; the pipeline overlaps load, QK, softmax, and PV within that boundary.

This chain explains why DSA can substantially reduce the arithmetic cost and main-KV bandwidth
of long-context serving. Its converse is equally important:

$$
\boxed{ \text{DSA reduces the dimension and constant of the quadratic term,} \quad \text{but the exact Lightning Indexer keeps full prefill at }O(L^2). }
$$

These two statements are not contradictory. The former shows how algorithm–infra co-design
turns arbitrary token sparsity into regular tiles that a GPU can execute; the latter identifies
Top-K discovery as the next boundary that long-context systems must confront.

**The derivations, source observations, and measurements cover different layers.** The
DeepSeek-V3.2 paper and public inference reference define the semantics and training
procedure. DeepGEMM reveals where its FP8 scorer materializes scores; FlashMLA SM90
defines the CTA, warpgroup, and sparse-prefill interface. The values 34,816 and about
68,592 tokens are useful-MAC references. The $[32K,64K]$ bracket, 2M measurements,
and 4M capacity limit come from the causal-prefill scaling experiment; 128K NCU
measurements explain carrier and Top-K traffic. The near-null locality response comes
from a separate H100 BF16 sparse-prefill microbenchmark. This article does not measure
the performance of Indexer–TopK fusion or Blackwell `gather4`: the former remains a
source- and profiler-constrained design opportunity, while the latter is tied to the
specific audited commit above.

### 4.6 Returning to the Beginning: What Do These Two Co-design Paths Reveal Together?

> **Takeaway.** The algorithm decides “which data is worth accessing” and “who can share one
> access”; the infrastructure decides whether those accesses can be owned, rearranged, and
> reduced by one program. Their interface is shape, ownership, and reuse—not an abstract
> sparsity ratio.

Putting NSA and DSA back into one table makes their distinction more than merely block-sparse
versus token-sparse:

| Question | NSA | DSA |
|---|---|---|
| Global control plane | Compression Attention at $T/d$ resolution | Full-token-resolution 128-D FP8 Indexer |
| Final unit of selection | Contiguous 64-token block | Arbitrary single token |
| Axis shared by the algorithm | 16 query heads in one KV group | 128 main heads |
| Where regularity appears | Before the HBM load | After discrete rows enter SMEM |
| Load reuse within one program | 16 heads | 64 heads within one SM90 head64 CTA |
| Main hidden cost | Compression path and the many-to-one reduction for $dK^{\mathrm{slc}}/dV^{\mathrm{slc}}$ | Exact all-history scan and Top-K |

Both systems have a “cheaper global scan + more expensive sparse data plane.” Keeping only
the token and feature scales, NSA's compressed sequence length is approximately:

$$
T_c\approx \frac{T}{d},
$$

so the number of scores in Compression Attention remains approximately:

$$
T\cdot T_c
=
\Theta\!\left(\frac{T^2}{d}\right).
$$

It produces a coarse global-context output while also reusing its probabilities for selection
routing. DSA's Indexer is routing-only, but retains full token resolution, so the number of
candidate scores over a complete causal sequence remains:

$$
\sum_{t=1}^{T}t
=
\Theta(T^2).
$$

Neither method makes “discover globally relevant positions” asymptotically linear. Instead,
each moves discovery into a cheaper representation: NSA lowers token resolution, while DSA
lowers feature dimension and uses FP8. Their real difference is *when* regularity appears:

$$
\boxed{
\begin{aligned}
\text{NSA:}\quad&
\text{low-resolution scan}
\rightarrow\text{contiguous block IDs}
\rightarrow\text{regularity exists before the load};\\
\text{DSA:}\quad&
\text{low-dimensional, full-resolution scan}
\rightarrow\text{discrete token IDs}
\rightarrow\text{regularity is reconstructed after the load}.
\end{aligned}}
$$

We must also distinguish two forms of reuse that are easy to conflate:

- **semantic fanout**: how many logit pairs a selection ID represents in the algorithmic semantics;
- **physical load reuse**: how many heads or query rows a KV row actually serves after entering a particular CTA/program.

One NSA block ID represents:

$$
64\ \text{tokens}\times16\ \text{heads}=1024
$$

logit pairs, while the same KV row is used by 16 heads within one program. A DSA token ID is
shared by 128 main heads at the algorithmic level, but the public SM90 sparse-prefill path uses
two CTAs that each process 64 heads. Shared memory cannot be shared across CTAs, so the
physical load reuse guaranteed within one CTA is 64, not 128. Algorithmic sharing is a necessary
condition for kernel reuse, but it does not automatically equal the final reuse count of one
HBM transaction.

Pipeline behavior has a similarly clear boundary. It can overlap:

```text
metadata / indirect address
        ↓
producer or load stage
        ↓
regular on-chip tile
        ↓
dense MMA consumer
```

but it cannot eliminate L2 misses, TLB/page locality effects, HBM transactions, or pipeline
startup and drain. Irregularity is isolated in the producer so that indirect addresses do not
continue to contaminate QK, softmax, and PV; this does not turn random HBM rows into
contiguous rows.

Therefore, evaluating sparse Attention requires at least the following items in one ledger:

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

Benchmarking only the final sparse core removes precisely the least scalable component from the
chart. Likewise, explaining only how many pairs the forward pass skips—without assigning a
stable owner to $dQ,dK,dV$—does not yet produce an executable training primitive.

NSA and DSA ultimately represent two complementary but non-equivalent paths:

$$
\boxed{
\begin{aligned}
\text{NSA:}\quad&
\text{first make the algorithm produce regular blocks, then let the kernel consume them directly;}\\
\text{DSA:}\quad&
\text{preserve token-level freedom, then recover regular tiles through shared selection and a specialized kernel.}
\end{aligned}}
$$

The real open question left by this research thread is not “what should the next sparse mask
look like?” but:

> **Can we avoid a full $L^2$ selector while still exposing a sparse workload with enough
> sharing, regularity, and reducibility for the kernel?**

Hierarchical routing, approximate retrieval, and candidate-set reuse across queries may all
break the assumption of an exact all-candidate scan; but they simultaneously change model
quality, dynamism, and the execution contract. The next question truly worth studying is the
verifiable boundary among these three—not another isolated reduction in nominal sparsity.

### Sources and Evidence

- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) provides the NSA
  algorithm, program mapping, and paper-reported performance results. The CSR backward
  mechanism in this article was checked against the community
  [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
  implementation and must not be attributed to an unreleased kernel from the paper's authors.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) provides the algorithmic definition
  of DSA/Lightning Indexer; the public reference logic is available in
  [`DeepSeek-V3.2-Exp`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).
- Code-level conclusions about the DSA scorer and sparse Attention were checked against
  [`DeepGEMM`](https://github.com/deepseek-ai/DeepGEMM) and
  [`FlashMLA`](https://github.com/deepseek-ai/FlashMLA), respectively.
- The causal-prefill Discovery scaling data were collected on one H100. The body
  separates repeated clean timings, instrumented stage timings, capacity preflight,
  and profiler diagnostics; profiler durations do not replace the CUDA-event curve.
- The KV-row locality data were also collected on one H100. The experiment section
  states the conditions, paired schedule, statistical procedure, correctness checks,
  profiler limitations, and the boundary between synthetic indices and production
  traces.
- The instruction semantics of Blackwell `tile::gather4` follow the
  [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
  This is not a path that the H100/SM90 experiments in this article can validate.
- The organization of this article draws on
  [*MLA, dim by dim*](https://nonlinear1.com/en/posts/mla-dim-by-dim)
  and its [Chinese version](https://nonlinear1.com/zh/posts/mla-dim-by-dim):
  state the conclusions first, then derive them dimension by dimension through a unified
  tensor lens. This article follows the same method further into program ownership, memory
  movement, and backward reduction.
