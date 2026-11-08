---
title: "DeepSeek EP 方案的抉择：Prefill 与 Decode 之间的 Insight"
description: "EP 不改变每张卡的计算量，只改变每个专家的行数、通信份数和专家负载不均。从这三个量出发，解释 DeepInfer 中 prefill 取小 EP、decode 取大 EP 的原因。"
date: 2026-11-08
draft: false
lang: zh
translationKey: deepseek-ep-scale
category: technical
---

<link rel="stylesheet" href="/blog/fused-indextopk/article.css" />

[English version](/blog/deepseek-ep-scale-en/)

部署 MoE 模型时，专家并行（EP）的规模要在 prefill 和 decode 两个阶段分别确定。DeepSeek 在 DeepInfer 中给两个阶段选择了相反的规模：prefill 在显存放得下的前提下取最小的 EP，通常是 EP2 到 EP8；decode 则开到 EP64 到 EP128。本文从“每个专家每步分到多少 token”出发，解释这一选择背后的原因。

> 本文依据 DeepInfer 的[部署文档](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)和 DeepSeek V4.1-Flash 的模型配置，只讨论 MoE 层：384 个路由专家，每个 token 选 6 个，另有 1 个共享专家，hidden 5120，专家中间维度 2304，专家权重为 FP4。硬件取 B200 的规格（FP8 稠密算力 4.5 PFLOPS，显存带宽 8 TB/s，NVLink 单向 900 GB/s），文中的时间都是按峰值估算的每层每卡耗时。

## 概要

- **EP 规模变化时的守恒量：每张卡的计算量。** 每张卡处理的 token 数 $T$ 固定时，它平均要算的行数是 $7T$（6 个路由专家加 1 个共享专家），与 EP 无关。
- **EP 规模变化时的变化量：具体有三个量。** 一是每个专家每步分到的行数：每张卡上的专家变少，同样的 $6T$ 行集中到更少的专家上，读一次权重能分给更多行，读权重的时间随之减少。二是每个 token 的通信份数：6 个专家分散到更多张卡上，要传的份数随之增加，逐渐接近 6 份，通信时间随之变长。三是专家负载不均：每张卡上的专家变少，负载偏高的专家多出来的那部分，更难被同卡的其他专家摊平，最忙的卡比平均更慢，其他卡都要等它。
- **两个阶段的差别只在读权重的时间。** 每个 token 的通信份数和专家负载不均的程度只取决于 EP，与每卡的 token 数 $T$ 无关，所以同一个 EP 下，两个阶段的通信占计算的比例相同，最忙的卡比平均慢多少也相同。每个专家的行数却正比于 $T$：prefill 每卡约 3.2 万个 token，EP4 时每个专家已有约 2,000 行，读权重只占计算时间的十几分之一；decode 每卡只有 256 个 token，EP8 时每个专家只有 32 行，读权重的时间是计算的 4 倍。
- **Prefill 和 Decode 各自的需求决定了 EP 的规模。** 在 B200 上，每个专家的行数超过约 150 行，读权重就不再是瓶颈，再加大 EP 也省不下时间。Prefill 在 EP4 时每个专家已有约 2,000 行，加大 EP 只会增加通信、加重专家负载不均，所以在显存允许的范围内取最小值。Decode 在 EP8 时每个专家只有 32 行，时间主要花在读权重上，EP32 时也只有 128 行，要加到 EP64 才能越过 150 行，所以取 EP64 到 EP128，并依靠 NVLink 域承担增加的通信，依靠 EPLB 缓解专家负载不均。

## 一、从 DeepInfer 思考 EP 规模的选择

DeepInfer 的部署文档对并行方式的说明很简短：prefill 使用 EP2 到 EP8，在显存允许的范围内取最小值；decode 使用 EP64 到 EP128，在可用 GPU 允许的范围内取最大值。两个阶段运行的是同一个模型、同一组专家，选择却正好相反。

一种常见的解释是，decode 每步的计算量小，所以可以分到更多卡上；prefill 的通信量大，所以要少分几张。这个解释默认了一个前提：EP 越大，每张卡分到的计算越少。本文先指出这个前提并不成立，再把 EP 的影响分成两类来看：改变 EP 时保持不变的守恒量，以及随之改变的变化量。

## 二、EP 规模变化时的守恒量：每张卡的计算量

DeepInfer 的注意力采用数据并行，DP 与 EP 的大小相同，每张卡处理自己那部分请求，持有 $T$ 个 token，形状为 [T, 5120]。gate 为每个 token 从 384 个路由专家中选出 6 个。token 按目标卡发出，每张卡收下选中本卡专家的行，按专家分组做两次矩阵乘，再把结果送回来源卡，与共享专家的输出相加。

$T$ 由每张卡上的请求决定，加卡并不会让它变小，官方的性能曲线也是按每卡的请求数对齐比较的。在这个前提下，整个 EP 组每层产生 EP × T × 6 行，路由均匀时平均分到每张卡的是

$$
\frac{EP \times T \times 6}{EP} = 6T
$$

行，再加上共享专家的 $T$ 行，共 $7T$ 行。EP 在式中被约去：从 EP8 增加到 EP64，每张卡要做的有效计算保持不变。一行经过一个专家约需 70.8 MFLOP，decode 每卡 256 个 token 时每层约 28 µs，prefill 每卡 32,768 个 token 时约 3.6 ms。

这里守恒的是平均每卡的有效计算。路由不均匀时，最忙那张卡会多算一些，这正是下一节要讨论的专家负载不均；分组矩阵乘在专家行数很少时还要补零对齐，实际执行的运算会略多于有效计算，这在 decode 的小 EP 下更明显，但不改变本文的结论。

既然 EP 不能减少计算，它的作用就只能从它改变的那些量里找。

## 三、EP 规模变化时的变化量：行数、通信份数与专家负载不均

### 每个专家的行数：决定读权重的时间

EP 改变的是这 $6T$ 行如何分到专家上。每张卡持有 384 / EP 个专家，每个专家每步分到的行数为

$$
r = \frac{EP \times T \times 6}{384}
$$

以 decode 每卡 256 个 token 为例，EP8 时每张卡有 48 个专家，每个专家只有 32 行；EP64 时每张卡有 6 个专家，每个专家有 256 行。

$r$ 的意义在于，专家权重每一步都要从显存读取一次，$r$ 决定了读一次权重能被多少行复用。一个专家的参数量记为 $P$，算 $r$ 行需要 $2Pr$ 次运算，读取权重需要搬运 $Pb$ 字节，$b$ 是每个参数的字节数。令两者耗时相等，$P$ 被约去，得到权重复用阈值：

$$
r^{*} = \frac{b}{2} \times \frac{\text{算力}}{\text{显存带宽}}
$$

V4.1 的专家权重为 FP4，连同缩放因子约 0.53 字节每参数，代入 B200 的规格，$r^{*} \approx 0.265 \times 562.5 \approx 150$ 行。这里只计了权重，把每行激活的读写也算上，阈值约为 180 行，不影响下文的结论。$r$ 高于 $r^{*}$ 时，专家矩阵乘受算力限制；低于 $r^{*}$ 时，时间主要花在读权重上。$r / r^{*}$ 近似等于计算时间与读权重时间之比，加大 EP 的作用，就是把 $r$ 推过 $r^{*}$，让读权重的时间降到计算以下。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 430" role="img" aria-label="Rows per expert versus EP for prefill and decode, with the weight-reuse threshold" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><line class="grid" x1="80.0" y1="338.2" x2="640.0" y2="338.2"/><text class="s" x="72.0" y="342.2" text-anchor="end">10</text><line class="grid" x1="80.0" y1="265.7" x2="640.0" y2="265.7"/><text class="s" x="72.0" y="269.7" text-anchor="end">100</text><line class="grid" x1="80.0" y1="193.2" x2="640.0" y2="193.2"/><text class="s" x="72.0" y="197.2" text-anchor="end">1,000</text><line class="grid" x1="80.0" y1="120.7" x2="640.0" y2="120.7"/><text class="s" x="72.0" y="124.7" text-anchor="end">10,000</text><line class="axis" x1="80.0" y1="360.0" x2="80.0" y2="364.0"/><text class="s" x="80.0" y="378.0" text-anchor="middle">2</text><line class="axis" x1="173.3" y1="360.0" x2="173.3" y2="364.0"/><text class="s" x="173.3" y="378.0" text-anchor="middle">4</text><line class="axis" x1="266.7" y1="360.0" x2="266.7" y2="364.0"/><text class="s" x="266.7" y="378.0" text-anchor="middle">8</text><line class="axis" x1="360.0" y1="360.0" x2="360.0" y2="364.0"/><text class="s" x="360.0" y="378.0" text-anchor="middle">16</text><line class="axis" x1="453.3" y1="360.0" x2="453.3" y2="364.0"/><text class="s" x="453.3" y="378.0" text-anchor="middle">32</text><line class="axis" x1="546.7" y1="360.0" x2="546.7" y2="364.0"/><text class="s" x="546.7" y="378.0" text-anchor="middle">64</text><line class="axis" x1="640.0" y1="360.0" x2="640.0" y2="364.0"/><text class="s" x="640.0" y="378.0" text-anchor="middle">128</text><line class="axis" x1="80.0" y1="360.0" x2="640.0" y2="360.0"/><text class="si" x="360.0" y="400.0" text-anchor="middle">EP</text><text class="si" x="72.0" y="56.0" text-anchor="start">每个专家每步的行数 r</text><line class="ln comp" x1="80" y1="26.0" x2="106.0" y2="26.0"/><text class="t" x="112.0" y="30.0" text-anchor="start">Prefill（每卡 32,768 个 token）</text><line class="ln comp dash" x1="360.69999999999993" y1="26.0" x2="386.69999999999993" y2="26.0"/><text class="t" x="392.7" y="30.0" text-anchor="start">Decode（每卡 256 个 token）</text><line class="ref" x1="80.0" y1="253.0" x2="640.0" y2="253.0"/><line class="ref2" x1="80.0" y1="247.5" x2="640.0" y2="247.5"/><text class="si" x="86.0" y="269.0" text-anchor="start">r* ≈ 150（只计权重）</text><text class="si" x="86.0" y="241.5" text-anchor="start">≈ 180（计入激活）</text><polyline class="ln comp" points="80.0,192.4 173.3,170.6 266.7,148.8 360.0,127.0 453.3,105.1 546.7,83.3 640.0,61.5"/><circle class="dot f-comp" cx="80.0" cy="192.4" r="5.5"><title>Prefill EP2：r = 1,024 行，r / r* = 6.85</title></circle><circle class="dot f-comp" cx="173.3" cy="170.6" r="5.5"><title>Prefill EP4：r = 2,048 行，r / r* = 13.71</title></circle><circle class="dot f-comp" cx="266.7" cy="148.8" r="5.5"><title>Prefill EP8：r = 4,096 行，r / r* = 27.41</title></circle><circle class="dot f-comp" cx="360.0" cy="127.0" r="3.5"><title>Prefill EP16：r = 8,192 行，r / r* = 54.83</title></circle><circle class="dot f-comp" cx="453.3" cy="105.1" r="3.5"><title>Prefill EP32：r = 16,384 行，r / r* = 109.66</title></circle><circle class="dot f-comp" cx="546.7" cy="83.3" r="3.5"><title>Prefill EP64：r = 32,768 行，r / r* = 219.31</title></circle><circle class="dot f-comp" cx="640.0" cy="61.5" r="3.5"><title>Prefill EP128：r = 65,536 行，r / r* = 438.62</title></circle><polyline class="ln comp dash" points="80.0,345.2 173.3,323.4 266.7,301.6 360.0,279.7 453.3,257.9 546.7,236.1 640.0,214.3"/><circle class="dot f-comp" cx="80.0" cy="345.2" r="3.5"><title>Decode EP2：r = 8 行，r / r* = 0.05</title></circle><circle class="dot f-comp" cx="173.3" cy="323.4" r="3.5"><title>Decode EP4：r = 16 行，r / r* = 0.11</title></circle><circle class="dot f-comp" cx="266.7" cy="301.6" r="3.5"><title>Decode EP8：r = 32 行，r / r* = 0.21</title></circle><circle class="dot f-comp" cx="360.0" cy="279.7" r="3.5"><title>Decode EP16：r = 64 行，r / r* = 0.43</title></circle><circle class="dot f-comp" cx="453.3" cy="257.9" r="3.5"><title>Decode EP32：r = 128 行，r / r* = 0.86</title></circle><circle class="dot f-comp" cx="546.7" cy="236.1" r="5.5"><title>Decode EP64：r = 256 行，r / r* = 1.71</title></circle><circle class="dot f-comp" cx="640.0" cy="214.3" r="5.5"><title>Decode EP128：r = 512 行，r / r* = 3.43</title></circle><line class="ref2" x1="474.2" y1="253.0" x2="474.2" y2="360.0"/><text class="si" x="480.2" y="352.0" text-anchor="start">Decode 在 EP ≈ 38 越过阈值</text><text class="s" x="173.3" y="156.6" text-anchor="middle">官方 prefill：EP2–8</text><text class="s" x="592.6" y="209.3" text-anchor="middle">官方 decode：EP64–128</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 1：每个专家每步的行数随 EP 的变化。prefill 每卡 32,768 个 token，decode 每卡 256 个 token；虚线为权重复用阈值。</figcaption>
</figure>

图 1 中两条线的斜率相同，都随 EP 线性增长，但高度相差 128 倍。prefill 在 EP2 时就已远在阈值之上；decode 要到 EP 约为 38 时才越过阈值。

### 每个 token 的通信份数：决定通信时间

一个 token 的 6 个专家里，平均有 $6 \times (EP - 1) / EP$ 个在别的卡上。每个 token 要传几份，取决于实现是否按卡去重，两种做法都有实际的实现。不去重时，每个“token–专家”对各传一份，DeepEP 的 V1 低延迟模式和 Mega MoE 都是这样：

$$
c_{\text{不去重}} = 6 \times \left(1 - \frac{1}{EP}\right)
$$

按卡去重时，发往同一张卡的只传一份，到了再展开，DeepEP 的 V1 普通模式和 V2 direct 模式是这样：

$$
c_{\text{去重}} = (EP - 1) \times \left[1 - \left(1 - \frac{1}{EP}\right)^{6}\right]
$$

每份数据在 dispatch 时约 5.4 KB（FP8），在 combine 时约 10.2 KB（BF16）。通信时间正比于 $T \times c$，计算时间正比于 $7T$，两个式子里都没有 $T$，所以无论是否去重，通信与计算之比都与 $T$ 无关，只取决于 EP 和链路带宽。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 450" role="img" aria-label="Copies per token versus EP, with and without per-card deduplication" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><line class="grid" x1="80.0" y1="385.0" x2="640.0" y2="385.0"/><text class="s" x="72.0" y="389.0" text-anchor="end">0</text><line class="grid" x1="80.0" y1="340.4" x2="640.0" y2="340.4"/><text class="s" x="72.0" y="344.4" text-anchor="end">1</text><line class="grid" x1="80.0" y1="295.8" x2="640.0" y2="295.8"/><text class="s" x="72.0" y="299.8" text-anchor="end">2</text><line class="grid" x1="80.0" y1="251.2" x2="640.0" y2="251.2"/><text class="s" x="72.0" y="255.2" text-anchor="end">3</text><line class="grid" x1="80.0" y1="206.5" x2="640.0" y2="206.5"/><text class="s" x="72.0" y="210.5" text-anchor="end">4</text><line class="grid" x1="80.0" y1="161.9" x2="640.0" y2="161.9"/><text class="s" x="72.0" y="165.9" text-anchor="end">5</text><line class="grid" x1="80.0" y1="117.3" x2="640.0" y2="117.3"/><text class="s" x="72.0" y="121.3" text-anchor="end">6</text><line class="axis" x1="80.0" y1="385.0" x2="80.0" y2="389.0"/><text class="s" x="80.0" y="403.0" text-anchor="middle">2</text><line class="axis" x1="173.3" y1="385.0" x2="173.3" y2="389.0"/><text class="s" x="173.3" y="403.0" text-anchor="middle">4</text><line class="axis" x1="266.7" y1="385.0" x2="266.7" y2="389.0"/><text class="s" x="266.7" y="403.0" text-anchor="middle">8</text><line class="axis" x1="360.0" y1="385.0" x2="360.0" y2="389.0"/><text class="s" x="360.0" y="403.0" text-anchor="middle">16</text><line class="axis" x1="453.3" y1="385.0" x2="453.3" y2="389.0"/><text class="s" x="453.3" y="403.0" text-anchor="middle">32</text><line class="axis" x1="546.7" y1="385.0" x2="546.7" y2="389.0"/><text class="s" x="546.7" y="403.0" text-anchor="middle">64</text><line class="axis" x1="640.0" y1="385.0" x2="640.0" y2="389.0"/><text class="s" x="640.0" y="403.0" text-anchor="middle">128</text><line class="axis" x1="80.0" y1="385.0" x2="640.0" y2="385.0"/><text class="si" x="360.0" y="425.0" text-anchor="middle">EP</text><text class="si" x="72.0" y="81.0" text-anchor="start">每个 token 的通信份数 c</text><line class="ln comm" x1="80" y1="22.0" x2="106.0" y2="22.0"/><text class="t" x="112.0" y="26.0" text-anchor="start">不去重（Mega MoE、DeepEP V1 低延迟）</text><line class="ln comm dash" x1="80" y1="46.0" x2="106.0" y2="46.0"/><text class="t" x="112.0" y="50.0" text-anchor="start">按卡去重（DeepEP V1 普通、V2 direct）</text><line class="ref" x1="80.0" y1="117.3" x2="640.0" y2="117.3"/><text class="si" x="84.0" y="111.3" text-anchor="start">上限 6（每个 token 选 6 个专家）</text><polyline class="ln comm" points="80.0,251.2 173.3,184.2 266.7,150.8 360.0,134.0 453.3,125.7 546.7,121.5 640.0,119.4"/><circle class="dot f-comm" cx="80.0" cy="251.2" r="4.5"><title>不去重 EP2：c = 3.00</title></circle><circle class="dot f-comm" cx="173.3" cy="184.2" r="4.5"><title>不去重 EP4：c = 4.50</title></circle><circle class="dot f-comm" cx="266.7" cy="150.8" r="4.5"><title>不去重 EP8：c = 5.25</title></circle><circle class="dot f-comm" cx="360.0" cy="134.0" r="4.5"><title>不去重 EP16：c = 5.62</title></circle><circle class="dot f-comm" cx="453.3" cy="125.7" r="4.5"><title>不去重 EP32：c = 5.81</title></circle><circle class="dot f-comm" cx="546.7" cy="121.5" r="4.5"><title>不去重 EP64：c = 5.91</title></circle><circle class="dot f-comm" cx="640.0" cy="119.4" r="4.5"><title>不去重 EP128：c = 5.95</title></circle><polyline class="ln comm dash" points="80.0,341.1 173.3,275.0 266.7,212.9 360.0,170.1 453.3,145.1 546.7,131.6 640.0,124.5"/><circle class="dot f-comm" cx="80.0" cy="341.1" r="4.5"><title>按卡去重 EP2：c = 0.98</title></circle><circle class="dot f-comm" cx="173.3" cy="275.0" r="4.5"><title>按卡去重 EP4：c = 2.47</title></circle><circle class="dot f-comm" cx="266.7" cy="212.9" r="4.5"><title>按卡去重 EP8：c = 3.86</title></circle><circle class="dot f-comm" cx="360.0" cy="170.1" r="4.5"><title>按卡去重 EP16：c = 4.82</title></circle><circle class="dot f-comm" cx="453.3" cy="145.1" r="4.5"><title>按卡去重 EP32：c = 5.38</title></circle><circle class="dot f-comm" cx="546.7" cy="131.6" r="4.5"><title>按卡去重 EP64：c = 5.68</title></circle><circle class="dot f-comm" cx="640.0" cy="124.5" r="4.5"><title>按卡去重 EP128：c = 5.84</title></circle><line class="ref2" x1="173.3" y1="190.2" x2="173.3" y2="269.0"/><line class="ref2" x1="546.7" y1="127.5" x2="546.7" y2="125.6"/><text class="t" x="183.3" y="200.7" text-anchor="start">EP4：相差 1.82 倍</text><text class="t" x="546.7" y="157.6" text-anchor="middle">EP64：相差 1.04 倍</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 2：每个 token 的通信份数随 EP 的变化。</figcaption>
</figure>

两种做法的 $c$ 都随 EP 单调增加，并趋近 6；去重带来的节省却随 EP 增大而迅速消失，EP4 时能少传近一半（4.50 对 2.47），EP64 时只少 4%。所以去重在 prefill 的小 EP 下很有价值，到 decode 的大 EP 下几乎没有作用；反过来说，Mega MoE 不去重，在 decode 中几乎不吃亏，在 prefill 中则要多付一些通信。DeepInfer 在 NVLink 域内使用的是 Mega MoE，下文的数值按不去重计算。

### 专家负载不均：决定最忙的卡比平均慢多少

所有卡要在 combine 时汇合，一层的耗时由最忙的卡决定。设某个专家的负载是平均值的 $h$ 倍，它所在卡的负载与平均值之比为

$$
lb = 1 + \frac{h - 1}{384 / EP}
$$

每卡的专家越多，单个专家负载偏高的影响越容易被摊平。$lb$ 同样与 $T$ 无关，只取决于 EP。

| 阶段 | EP | $r$ | $r / r^{*}$ | 通信 / 计算 | $lb$（$h = 10$） |
|---|---:|---:|---:|---:|---:|
| Prefill | 2 | 1,024 | 6.8 | 0.47 | 1.05 |
| Prefill | 4 | 2,048 | 13.7 | 0.71 | 1.09 |
| Prefill | 8 | 4,096 | 27 | 0.83 | 1.19 |
| Decode | 8 | 32 | 0.21 | 0.83 | 1.19 |
| Decode | 32 | 128 | 0.86 | 0.92 | 1.75 |
| Decode | 64 | 256 | 1.7 | 0.93 | 2.5 |
| Decode | 128 | 512 | 3.4 | 0.94 | 4 |

*表 1：EP 改变的三个量。通信 / 计算按 Mega MoE 不去重、NVLink 计。prefill 每卡 T = 32,768，取 DeepInfer benchmark 的默认值；decode 每卡 T = 256，取官方曲线上每卡 64 个请求、每个请求每步验证 4 个 token 的配置。*

表 1 中 EP8 的两行最能说明问题：两个阶段的通信与计算之比和 $lb$ 完全相同，$r / r^{*}$ 却相差两个数量级。通信和专家负载不均只随 EP 变化，每个专家的行数随 EP × T 变化。图 3 把这一点画了出来：通信与计算之比只有一条线，两个阶段共用；读权重与计算之比是两条线，彼此平行，相差 128 倍。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 450" role="img" aria-label="Weight-read time and communication time relative to compute, versus EP" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><line class="grid" x1="80.0" y1="331.5" x2="640.0" y2="331.5"/><text class="s" x="72.0" y="335.5" text-anchor="end">0.01</text><line class="grid" x1="80.0" y1="262.0" x2="640.0" y2="262.0"/><text class="s" x="72.0" y="266.0" text-anchor="end">0.1</text><line class="grid" x1="80.0" y1="192.6" x2="640.0" y2="192.6"/><text class="s" x="72.0" y="196.6" text-anchor="end">1</text><line class="grid" x1="80.0" y1="123.1" x2="640.0" y2="123.1"/><text class="s" x="72.0" y="127.1" text-anchor="end">10</text><line class="axis" x1="80.0" y1="380.0" x2="80.0" y2="384.0"/><text class="s" x="80.0" y="398.0" text-anchor="middle">2</text><line class="axis" x1="173.3" y1="380.0" x2="173.3" y2="384.0"/><text class="s" x="173.3" y="398.0" text-anchor="middle">4</text><line class="axis" x1="266.7" y1="380.0" x2="266.7" y2="384.0"/><text class="s" x="266.7" y="398.0" text-anchor="middle">8</text><line class="axis" x1="360.0" y1="380.0" x2="360.0" y2="384.0"/><text class="s" x="360.0" y="398.0" text-anchor="middle">16</text><line class="axis" x1="453.3" y1="380.0" x2="453.3" y2="384.0"/><text class="s" x="453.3" y="398.0" text-anchor="middle">32</text><line class="axis" x1="546.7" y1="380.0" x2="546.7" y2="384.0"/><text class="s" x="546.7" y="398.0" text-anchor="middle">64</text><line class="axis" x1="640.0" y1="380.0" x2="640.0" y2="384.0"/><text class="s" x="640.0" y="398.0" text-anchor="middle">128</text><line class="axis" x1="80.0" y1="380.0" x2="640.0" y2="380.0"/><text class="si" x="360.0" y="420.0" text-anchor="middle">EP</text><text class="si" x="72.0" y="76.0" text-anchor="start">与计算时间之比</text><line class="ln wt" x1="80" y1="26.0" x2="106.0" y2="26.0"/><text class="t" x="112.0" y="30.0" text-anchor="start">读权重 ÷ 计算：prefill</text><line class="ln wt dash" x1="295.0" y1="26.0" x2="321.0" y2="26.0"/><text class="t" x="327.0" y="30.0" text-anchor="start">读权重 ÷ 计算：decode</text><line class="ln comm" x1="80" y1="50.0" x2="106.0" y2="50.0"/><text class="t" x="112.0" y="54.0" text-anchor="start">通信 ÷ 计算：两个阶段相同（不去重，NVLink）</text><line class="ref" x1="80.0" y1="192.6" x2="640.0" y2="192.6"/><text class="si" x="636.0" y="186.6" text-anchor="end">与计算相等</text><polyline class="ln wt" points="80.0,255.0 173.3,275.6 266.7,295.9 360.0,315.7 453.3,334.5 546.7,351.6 640.0,366.0"/><circle class="dot f-wt" cx="80.0" cy="255.0" r="4.5"><title>读权重 ÷ 计算（prefill） EP2：0.126</title></circle><circle class="dot f-wt" cx="173.3" cy="275.6" r="4.5"><title>读权重 ÷ 计算（prefill） EP4：0.0638</title></circle><circle class="dot f-wt" cx="266.7" cy="295.9" r="4.5"><title>读权重 ÷ 计算（prefill） EP8：0.0325</title></circle><circle class="dot f-wt" cx="360.0" cy="315.7" r="4.5"><title>读权重 ÷ 计算（prefill） EP16：0.0169</title></circle><circle class="dot f-wt" cx="453.3" cy="334.5" r="4.5"><title>读权重 ÷ 计算（prefill） EP32：0.00904</title></circle><circle class="dot f-wt" cx="546.7" cy="351.6" r="4.5"><title>读权重 ÷ 计算（prefill） EP64：0.00514</title></circle><circle class="dot f-wt" cx="640.0" cy="366.0" r="4.5"><title>读权重 ÷ 计算（prefill） EP128：0.00318</title></circle><polyline class="ln wt dash" points="80.0,108.6 173.3,129.3 266.7,149.6 360.0,169.4 453.3,188.2 546.7,205.2 640.0,219.7"/><circle class="dot f-wt" cx="80.0" cy="108.6" r="4.5"><title>读权重 ÷ 计算（decode） EP2：16.2</title></circle><circle class="dot f-wt" cx="173.3" cy="129.3" r="4.5"><title>读权重 ÷ 计算（decode） EP4：8.16</title></circle><circle class="dot f-wt" cx="266.7" cy="149.6" r="4.5"><title>读权重 ÷ 计算（decode） EP8：4.16</title></circle><circle class="dot f-wt" cx="360.0" cy="169.4" r="4.5"><title>读权重 ÷ 计算（decode） EP16：2.16</title></circle><circle class="dot f-wt" cx="453.3" cy="188.2" r="4.5"><title>读权重 ÷ 计算（decode） EP32：1.16</title></circle><circle class="dot f-wt" cx="546.7" cy="205.2" r="4.5"><title>读权重 ÷ 计算（decode） EP64：0.657</title></circle><circle class="dot f-wt" cx="640.0" cy="219.7" r="4.5"><title>读权重 ÷ 计算（decode） EP128：0.407</title></circle><polyline class="ln comm" points="80.0,215.2 173.3,202.9 266.7,198.3 360.0,196.2 453.3,195.2 546.7,194.7 640.0,194.5"/><circle class="dot f-comm" cx="80.0" cy="215.2" r="4.5"><title>通信 ÷ 计算 EP2：0.473</title></circle><circle class="dot f-comm" cx="173.3" cy="202.9" r="4.5"><title>通信 ÷ 计算 EP4：0.709</title></circle><circle class="dot f-comm" cx="266.7" cy="198.3" r="4.5"><title>通信 ÷ 计算 EP8：0.827</title></circle><circle class="dot f-comm" cx="360.0" cy="196.2" r="4.5"><title>通信 ÷ 计算 EP16：0.886</title></circle><circle class="dot f-comm" cx="453.3" cy="195.2" r="4.5"><title>通信 ÷ 计算 EP32：0.916</title></circle><circle class="dot f-comm" cx="546.7" cy="194.7" r="4.5"><title>通信 ÷ 计算 EP64：0.931</title></circle><circle class="dot f-comm" cx="640.0" cy="194.5" r="4.5"><title>通信 ÷ 计算 EP128：0.938</title></circle><text class="s" x="360.0" y="157.4" text-anchor="start">decode：EP32–64 之间降到 1 以下</text><text class="s" x="86.0" y="359.1" text-anchor="start">prefill：始终远低于 1</text><text class="s" x="86.0" y="235.2" text-anchor="start">只随 EP 变化，与 T 无关</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 3：读权重时间、通信时间与计算时间之比随 EP 的变化。</figcaption>
</figure>

## 四、两个阶段的取舍

把读权重、通信和计算三项放在一起，就能看出每种组合下谁最慢。图 4 按每卡的 $T$ 和 EP，标出计算、读权重和通信三项中耗时最长的一项。左图在 NVLink 域内：prefill 的官方配置落在计算受限的区域，decode 的官方配置刚好越过读权重受限的区域。右图把链路换成每卡 100 GB/s 的网卡，几乎所有组合都变成通信最慢，这也是两个阶段都留在 NVLink 域内的原因。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 470" role="img" aria-label="Which term is slowest for each EP and per-card token count, over NVLink and over a 100 GB/s NIC" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><rect class="f-comp" x="80" y="20.0" width="14" height="14" rx="3"/><text class="t" x="100.0" y="32.0" text-anchor="start">计算最慢</text><rect class="f-wt" x="190" y="20.0" width="14" height="14" rx="3"/><text class="t" x="210.0" y="32.0" text-anchor="start">读权重最慢</text><rect class="f-comm" x="300" y="20.0" width="14" height="14" rx="3"/><text class="t" x="320.0" y="32.0" text-anchor="start">通信最慢</text><text class="h" x="226.0" y="68.0" text-anchor="middle">NVLink 900 GB/s</text><text class="s" x="92.0" y="97.0" text-anchor="end">64K</text><rect class="f-comp" x="101.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 65,536：计算 7,215.5 µs，读权重 455.6 µs，通信 3,411.4 µs</title></rect><rect class="f-comp" x="137.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 65,536：计算 7,215.5 µs，读权重 230.0 µs，通信 5,117.1 µs</title></rect><rect class="f-comp" x="173.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 65,536：计算 7,215.5 µs，读权重 117.2 µs，通信 5,969.9 µs</title></rect><rect class="f-comp" x="209.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 65,536：计算 7,215.5 µs，读权重 60.8 µs，通信 6,396.3 µs</title></rect><rect class="f-comp" x="245.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 65,536：计算 7,215.5 µs，读权重 32.6 µs，通信 6,609.5 µs</title></rect><rect class="f-comp" x="281.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 65,536：计算 7,215.5 µs，读权重 18.5 µs，通信 6,716.1 µs</title></rect><rect class="f-comp" x="317.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 65,536：计算 7,215.5 µs，读权重 11.5 µs，通信 6,769.4 µs</title></rect><text class="s" x="92.0" y="123.0" text-anchor="end">32K</text><rect class="f-comp" x="101.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 32,768：计算 3,607.8 µs，读权重 455.6 µs，通信 1,705.7 µs</title></rect><rect class="f-comp" x="137.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 32,768：计算 3,607.8 µs，读权重 230.0 µs，通信 2,558.5 µs</title></rect><rect class="f-comp" x="173.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 32,768：计算 3,607.8 µs，读权重 117.2 µs，通信 2,984.9 µs</title></rect><rect class="f-comp" x="209.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 32,768：计算 3,607.8 µs，读权重 60.8 µs，通信 3,198.2 µs</title></rect><rect class="f-comp" x="245.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 32,768：计算 3,607.8 µs，读权重 32.6 µs，通信 3,304.8 µs</title></rect><rect class="f-comp" x="281.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 32,768：计算 3,607.8 µs，读权重 18.5 µs，通信 3,358.1 µs</title></rect><rect class="f-comp" x="317.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 32,768：计算 3,607.8 µs，读权重 11.5 µs，通信 3,384.7 µs</title></rect><text class="s" x="92.0" y="149.0" text-anchor="end">16K</text><rect class="f-comp" x="101.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 16,384：计算 1,803.9 µs，读权重 455.6 µs，通信 852.8 µs</title></rect><rect class="f-comp" x="137.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 16,384：计算 1,803.9 µs，读权重 230.0 µs，通信 1,279.3 µs</title></rect><rect class="f-comp" x="173.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 16,384：计算 1,803.9 µs，读权重 117.2 µs，通信 1,492.5 µs</title></rect><rect class="f-comp" x="209.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 16,384：计算 1,803.9 µs，读权重 60.8 µs，通信 1,599.1 µs</title></rect><rect class="f-comp" x="245.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 16,384：计算 1,803.9 µs，读权重 32.6 µs，通信 1,652.4 µs</title></rect><rect class="f-comp" x="281.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 16,384：计算 1,803.9 µs，读权重 18.5 µs，通信 1,679.0 µs</title></rect><rect class="f-comp" x="317.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 16,384：计算 1,803.9 µs，读权重 11.5 µs，通信 1,692.4 µs</title></rect><text class="s" x="92.0" y="175.0" text-anchor="end">8K</text><rect class="f-comp" x="101.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 8,192：计算 901.9 µs，读权重 455.6 µs，通信 426.4 µs</title></rect><rect class="f-comp" x="137.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 8,192：计算 901.9 µs，读权重 230.0 µs，通信 639.6 µs</title></rect><rect class="f-comp" x="173.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 8,192：计算 901.9 µs，读权重 117.2 µs，通信 746.2 µs</title></rect><rect class="f-comp" x="209.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 8,192：计算 901.9 µs，读权重 60.8 µs，通信 799.5 µs</title></rect><rect class="f-comp" x="245.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 8,192：计算 901.9 µs，读权重 32.6 µs，通信 826.2 µs</title></rect><rect class="f-comp" x="281.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 8,192：计算 901.9 µs，读权重 18.5 µs，通信 839.5 µs</title></rect><rect class="f-comp" x="317.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 8,192：计算 901.9 µs，读权重 11.5 µs，通信 846.2 µs</title></rect><text class="s" x="92.0" y="201.0" text-anchor="end">4K</text><rect class="f-wt" x="101.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 4,096：计算 451.0 µs，读权重 455.6 µs，通信 213.2 µs</title></rect><rect class="f-comp" x="137.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 4,096：计算 451.0 µs，读权重 230.0 µs，通信 319.8 µs</title></rect><rect class="f-comp" x="173.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 4,096：计算 451.0 µs，读权重 117.2 µs，通信 373.1 µs</title></rect><rect class="f-comp" x="209.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 4,096：计算 451.0 µs，读权重 60.8 µs，通信 399.8 µs</title></rect><rect class="f-comp" x="245.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 4,096：计算 451.0 µs，读权重 32.6 µs，通信 413.1 µs</title></rect><rect class="f-comp" x="281.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 4,096：计算 451.0 µs，读权重 18.5 µs，通信 419.8 µs</title></rect><rect class="f-comp" x="317.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 4,096：计算 451.0 µs，读权重 11.5 µs，通信 423.1 µs</title></rect><text class="s" x="92.0" y="227.0" text-anchor="end">2K</text><rect class="f-wt" x="101.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 2,048：计算 225.5 µs，读权重 455.6 µs，通信 106.6 µs</title></rect><rect class="f-wt" x="137.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 2,048：计算 225.5 µs，读权重 230.0 µs，通信 159.9 µs</title></rect><rect class="f-comp" x="173.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 2,048：计算 225.5 µs，读权重 117.2 µs，通信 186.6 µs</title></rect><rect class="f-comp" x="209.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 2,048：计算 225.5 µs，读权重 60.8 µs，通信 199.9 µs</title></rect><rect class="f-comp" x="245.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 2,048：计算 225.5 µs，读权重 32.6 µs，通信 206.5 µs</title></rect><rect class="f-comp" x="281.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 2,048：计算 225.5 µs，读权重 18.5 µs，通信 209.9 µs</title></rect><rect class="f-comp" x="317.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 2,048：计算 225.5 µs，读权重 11.5 µs，通信 211.5 µs</title></rect><text class="s" x="92.0" y="253.0" text-anchor="end">1K</text><rect class="f-wt" x="101.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 1,024：计算 112.7 µs，读权重 455.6 µs，通信 53.3 µs</title></rect><rect class="f-wt" x="137.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 1,024：计算 112.7 µs，读权重 230.0 µs，通信 80.0 µs</title></rect><rect class="f-wt" x="173.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 1,024：计算 112.7 µs，读权重 117.2 µs，通信 93.3 µs</title></rect><rect class="f-comp" x="209.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 1,024：计算 112.7 µs，读权重 60.8 µs，通信 99.9 µs</title></rect><rect class="f-comp" x="245.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 1,024：计算 112.7 µs，读权重 32.6 µs，通信 103.3 µs</title></rect><rect class="f-comp" x="281.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 1,024：计算 112.7 µs，读权重 18.5 µs，通信 104.9 µs</title></rect><rect class="f-comp" x="317.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 1,024：计算 112.7 µs，读权重 11.5 µs，通信 105.8 µs</title></rect><text class="s" x="92.0" y="279.0" text-anchor="end">512</text><rect class="f-wt" x="101.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 512：计算 56.4 µs，读权重 455.6 µs，通信 26.7 µs</title></rect><rect class="f-wt" x="137.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 512：计算 56.4 µs，读权重 230.0 µs，通信 40.0 µs</title></rect><rect class="f-wt" x="173.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 512：计算 56.4 µs，读权重 117.2 µs，通信 46.6 µs</title></rect><rect class="f-wt" x="209.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 512：计算 56.4 µs，读权重 60.8 µs，通信 50.0 µs</title></rect><rect class="f-comp" x="245.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 512：计算 56.4 µs，读权重 32.6 µs，通信 51.6 µs</title></rect><rect class="f-comp" x="281.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 512：计算 56.4 µs，读权重 18.5 µs，通信 52.5 µs</title></rect><rect class="f-comp" x="317.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 512：计算 56.4 µs，读权重 11.5 µs，通信 52.9 µs</title></rect><text class="s" x="92.0" y="305.0" text-anchor="end">256</text><rect class="f-wt" x="101.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 256：计算 28.2 µs，读权重 455.6 µs，通信 13.3 µs</title></rect><rect class="f-wt" x="137.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 256：计算 28.2 µs，读权重 230.0 µs，通信 20.0 µs</title></rect><rect class="f-wt" x="173.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 256：计算 28.2 µs，读权重 117.2 µs，通信 23.3 µs</title></rect><rect class="f-wt" x="209.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 256：计算 28.2 µs，读权重 60.8 µs，通信 25.0 µs</title></rect><rect class="f-wt" x="245.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 256：计算 28.2 µs，读权重 32.6 µs，通信 25.8 µs</title></rect><rect class="f-comp" x="281.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 256：计算 28.2 µs，读权重 18.5 µs，通信 26.2 µs</title></rect><rect class="f-comp" x="317.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 256：计算 28.2 µs，读权重 11.5 µs，通信 26.4 µs</title></rect><text class="s" x="92.0" y="331.0" text-anchor="end">128</text><rect class="f-wt" x="101.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 128：计算 14.1 µs，读权重 455.6 µs，通信 6.7 µs</title></rect><rect class="f-wt" x="137.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 128：计算 14.1 µs，读权重 230.0 µs，通信 10.0 µs</title></rect><rect class="f-wt" x="173.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 128：计算 14.1 µs，读权重 117.2 µs，通信 11.7 µs</title></rect><rect class="f-wt" x="209.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 128：计算 14.1 µs，读权重 60.8 µs，通信 12.5 µs</title></rect><rect class="f-wt" x="245.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 128：计算 14.1 µs，读权重 32.6 µs，通信 12.9 µs</title></rect><rect class="f-wt" x="281.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 128：计算 14.1 µs，读权重 18.5 µs，通信 13.1 µs</title></rect><rect class="f-comp" x="317.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 128：计算 14.1 µs，读权重 11.5 µs，通信 13.2 µs</title></rect><text class="s" x="92.0" y="357.0" text-anchor="end">64</text><rect class="f-wt" x="101.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 64：计算 7.0 µs，读权重 455.6 µs，通信 3.3 µs</title></rect><rect class="f-wt" x="137.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 64：计算 7.0 µs，读权重 230.0 µs，通信 5.0 µs</title></rect><rect class="f-wt" x="173.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 64：计算 7.0 µs，读权重 117.2 µs，通信 5.8 µs</title></rect><rect class="f-wt" x="209.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 64：计算 7.0 µs，读权重 60.8 µs，通信 6.2 µs</title></rect><rect class="f-wt" x="245.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 64：计算 7.0 µs，读权重 32.6 µs，通信 6.5 µs</title></rect><rect class="f-wt" x="281.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 64：计算 7.0 µs，读权重 18.5 µs，通信 6.6 µs</title></rect><rect class="f-wt" x="317.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 64：计算 7.0 µs，读权重 11.5 µs，通信 6.6 µs</title></rect><text class="s" x="92.0" y="383.0" text-anchor="end">32</text><rect class="f-wt" x="101.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 32：计算 3.5 µs，读权重 455.6 µs，通信 1.7 µs</title></rect><rect class="f-wt" x="137.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 32：计算 3.5 µs，读权重 230.0 µs，通信 2.5 µs</title></rect><rect class="f-wt" x="173.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 32：计算 3.5 µs，读权重 117.2 µs，通信 2.9 µs</title></rect><rect class="f-wt" x="209.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 32：计算 3.5 µs，读权重 60.8 µs，通信 3.1 µs</title></rect><rect class="f-wt" x="245.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 32：计算 3.5 µs，读权重 32.6 µs，通信 3.2 µs</title></rect><rect class="f-wt" x="281.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 32：计算 3.5 µs，读权重 18.5 µs，通信 3.3 µs</title></rect><rect class="f-wt" x="317.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 32：计算 3.5 µs，读权重 11.5 µs，通信 3.3 µs</title></rect><text class="s" x="118.0" y="408.0" text-anchor="middle">2</text><text class="s" x="154.0" y="408.0" text-anchor="middle">4</text><text class="s" x="190.0" y="408.0" text-anchor="middle">8</text><text class="s" x="226.0" y="408.0" text-anchor="middle">16</text><text class="s" x="262.0" y="408.0" text-anchor="middle">32</text><text class="s" x="298.0" y="408.0" text-anchor="middle">64</text><text class="s" x="334.0" y="408.0" text-anchor="middle">128</text><text class="si" x="226.0" y="428.0" text-anchor="middle">EP</text><circle cx="154.0" cy="119.0" r="9" style="fill:var(--background);stroke:var(--text);stroke-width:1.5"/><text class="h" x="154.0" y="123.5" text-anchor="middle">P</text><circle cx="298.0" cy="301.0" r="9" style="fill:var(--background);stroke:var(--text);stroke-width:1.5"/><text class="h" x="298.0" y="305.5" text-anchor="middle">D</text><text class="h" x="546.0" y="68.0" text-anchor="middle">跨节点，网卡 100 GB/s</text><rect class="f-comm" x="421.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 65,536：计算 7,215.5 µs，读权重 455.6 µs，通信 30,702.3 µs</title></rect><rect class="f-comm" x="457.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 65,536：计算 7,215.5 µs，读权重 230.0 µs，通信 46,053.5 µs</title></rect><rect class="f-comm" x="493.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 65,536：计算 7,215.5 µs，读权重 117.2 µs，通信 53,729.0 µs</title></rect><rect class="f-comm" x="529.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 65,536：计算 7,215.5 µs，读权重 60.8 µs，通信 57,566.8 µs</title></rect><rect class="f-comm" x="565.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 65,536：计算 7,215.5 µs，读权重 32.6 µs，通信 59,485.7 µs</title></rect><rect class="f-comm" x="601.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 65,536：计算 7,215.5 µs，读权重 18.5 µs，通信 60,445.2 µs</title></rect><rect class="f-comm" x="637.0" y="81.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 65,536：计算 7,215.5 µs，读权重 11.5 µs，通信 60,924.9 µs</title></rect><rect class="f-comm" x="421.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 32,768：计算 3,607.8 µs，读权重 455.6 µs，通信 15,351.2 µs</title></rect><rect class="f-comm" x="457.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 32,768：计算 3,607.8 µs，读权重 230.0 µs，通信 23,026.7 µs</title></rect><rect class="f-comm" x="493.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 32,768：计算 3,607.8 µs，读权重 117.2 µs，通信 26,864.5 µs</title></rect><rect class="f-comm" x="529.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 32,768：计算 3,607.8 µs，读权重 60.8 µs，通信 28,783.4 µs</title></rect><rect class="f-comm" x="565.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 32,768：计算 3,607.8 µs，读权重 32.6 µs，通信 29,742.9 µs</title></rect><rect class="f-comm" x="601.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 32,768：计算 3,607.8 µs，读权重 18.5 µs，通信 30,222.6 µs</title></rect><rect class="f-comm" x="637.0" y="107.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 32,768：计算 3,607.8 µs，读权重 11.5 µs，通信 30,462.4 µs</title></rect><rect class="f-comm" x="421.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 16,384：计算 1,803.9 µs，读权重 455.6 µs，通信 7,675.6 µs</title></rect><rect class="f-comm" x="457.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 16,384：计算 1,803.9 µs，读权重 230.0 µs，通信 11,513.4 µs</title></rect><rect class="f-comm" x="493.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 16,384：计算 1,803.9 µs，读权重 117.2 µs，通信 13,432.3 µs</title></rect><rect class="f-comm" x="529.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 16,384：计算 1,803.9 µs，读权重 60.8 µs，通信 14,391.7 µs</title></rect><rect class="f-comm" x="565.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 16,384：计算 1,803.9 µs，读权重 32.6 µs，通信 14,871.4 µs</title></rect><rect class="f-comm" x="601.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 16,384：计算 1,803.9 µs，读权重 18.5 µs，通信 15,111.3 µs</title></rect><rect class="f-comm" x="637.0" y="133.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 16,384：计算 1,803.9 µs，读权重 11.5 µs，通信 15,231.2 µs</title></rect><rect class="f-comm" x="421.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 8,192：计算 901.9 µs，读权重 455.6 µs，通信 3,837.8 µs</title></rect><rect class="f-comm" x="457.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 8,192：计算 901.9 µs，读权重 230.0 µs，通信 5,756.7 µs</title></rect><rect class="f-comm" x="493.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 8,192：计算 901.9 µs，读权重 117.2 µs，通信 6,716.1 µs</title></rect><rect class="f-comm" x="529.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 8,192：计算 901.9 µs，读权重 60.8 µs，通信 7,195.9 µs</title></rect><rect class="f-comm" x="565.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 8,192：计算 901.9 µs，读权重 32.6 µs，通信 7,435.7 µs</title></rect><rect class="f-comm" x="601.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 8,192：计算 901.9 µs，读权重 18.5 µs，通信 7,555.6 µs</title></rect><rect class="f-comm" x="637.0" y="159.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 8,192：计算 901.9 µs，读权重 11.5 µs，通信 7,615.6 µs</title></rect><rect class="f-comm" x="421.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 4,096：计算 451.0 µs，读权重 455.6 µs，通信 1,918.9 µs</title></rect><rect class="f-comm" x="457.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 4,096：计算 451.0 µs，读权重 230.0 µs，通信 2,878.3 µs</title></rect><rect class="f-comm" x="493.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 4,096：计算 451.0 µs，读权重 117.2 µs，通信 3,358.1 µs</title></rect><rect class="f-comm" x="529.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 4,096：计算 451.0 µs，读权重 60.8 µs，通信 3,597.9 µs</title></rect><rect class="f-comm" x="565.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 4,096：计算 451.0 µs，读权重 32.6 µs，通信 3,717.9 µs</title></rect><rect class="f-comm" x="601.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 4,096：计算 451.0 µs，读权重 18.5 µs，通信 3,777.8 µs</title></rect><rect class="f-comm" x="637.0" y="185.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 4,096：计算 451.0 µs，读权重 11.5 µs，通信 3,807.8 µs</title></rect><rect class="f-comm" x="421.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 2,048：计算 225.5 µs，读权重 455.6 µs，通信 959.4 µs</title></rect><rect class="f-comm" x="457.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 2,048：计算 225.5 µs，读权重 230.0 µs，通信 1,439.2 µs</title></rect><rect class="f-comm" x="493.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 2,048：计算 225.5 µs，读权重 117.2 µs，通信 1,679.0 µs</title></rect><rect class="f-comm" x="529.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 2,048：计算 225.5 µs，读权重 60.8 µs，通信 1,799.0 µs</title></rect><rect class="f-comm" x="565.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 2,048：计算 225.5 µs，读权重 32.6 µs，通信 1,858.9 µs</title></rect><rect class="f-comm" x="601.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 2,048：计算 225.5 µs，读权重 18.5 µs，通信 1,888.9 µs</title></rect><rect class="f-comm" x="637.0" y="211.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 2,048：计算 225.5 µs，读权重 11.5 µs，通信 1,903.9 µs</title></rect><rect class="f-comm" x="421.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 1,024：计算 112.7 µs，读权重 455.6 µs，通信 479.7 µs</title></rect><rect class="f-comm" x="457.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 1,024：计算 112.7 µs，读权重 230.0 µs，通信 719.6 µs</title></rect><rect class="f-comm" x="493.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 1,024：计算 112.7 µs，读权重 117.2 µs，通信 839.5 µs</title></rect><rect class="f-comm" x="529.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 1,024：计算 112.7 µs，读权重 60.8 µs，通信 899.5 µs</title></rect><rect class="f-comm" x="565.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 1,024：计算 112.7 µs，读权重 32.6 µs，通信 929.5 µs</title></rect><rect class="f-comm" x="601.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 1,024：计算 112.7 µs，读权重 18.5 µs，通信 944.5 µs</title></rect><rect class="f-comm" x="637.0" y="237.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 1,024：计算 112.7 µs，读权重 11.5 µs，通信 952.0 µs</title></rect><rect class="f-wt" x="421.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 512：计算 56.4 µs，读权重 455.6 µs，通信 239.9 µs</title></rect><rect class="f-comm" x="457.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 512：计算 56.4 µs，读权重 230.0 µs，通信 359.8 µs</title></rect><rect class="f-comm" x="493.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 512：计算 56.4 µs，读权重 117.2 µs，通信 419.8 µs</title></rect><rect class="f-comm" x="529.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 512：计算 56.4 µs，读权重 60.8 µs，通信 449.7 µs</title></rect><rect class="f-comm" x="565.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 512：计算 56.4 µs，读权重 32.6 µs，通信 464.7 µs</title></rect><rect class="f-comm" x="601.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 512：计算 56.4 µs，读权重 18.5 µs，通信 472.2 µs</title></rect><rect class="f-comm" x="637.0" y="263.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 512：计算 56.4 µs，读权重 11.5 µs，通信 476.0 µs</title></rect><rect class="f-wt" x="421.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 256：计算 28.2 µs，读权重 455.6 µs，通信 119.9 µs</title></rect><rect class="f-wt" x="457.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 256：计算 28.2 µs，读权重 230.0 µs，通信 179.9 µs</title></rect><rect class="f-comm" x="493.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 256：计算 28.2 µs，读权重 117.2 µs，通信 209.9 µs</title></rect><rect class="f-comm" x="529.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 256：计算 28.2 µs，读权重 60.8 µs，通信 224.9 µs</title></rect><rect class="f-comm" x="565.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 256：计算 28.2 µs，读权重 32.6 µs，通信 232.4 µs</title></rect><rect class="f-comm" x="601.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 256：计算 28.2 µs，读权重 18.5 µs，通信 236.1 µs</title></rect><rect class="f-comm" x="637.0" y="289.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 256：计算 28.2 µs，读权重 11.5 µs，通信 238.0 µs</title></rect><rect class="f-wt" x="421.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 128：计算 14.1 µs，读权重 455.6 µs，通信 60.0 µs</title></rect><rect class="f-wt" x="457.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 128：计算 14.1 µs，读权重 230.0 µs，通信 89.9 µs</title></rect><rect class="f-wt" x="493.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 128：计算 14.1 µs，读权重 117.2 µs，通信 104.9 µs</title></rect><rect class="f-comm" x="529.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 128：计算 14.1 µs，读权重 60.8 µs，通信 112.4 µs</title></rect><rect class="f-comm" x="565.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 128：计算 14.1 µs，读权重 32.6 µs，通信 116.2 µs</title></rect><rect class="f-comm" x="601.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 128：计算 14.1 µs，读权重 18.5 µs，通信 118.1 µs</title></rect><rect class="f-comm" x="637.0" y="315.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 128：计算 14.1 µs，读权重 11.5 µs，通信 119.0 µs</title></rect><rect class="f-wt" x="421.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 64：计算 7.0 µs，读权重 455.6 µs，通信 30.0 µs</title></rect><rect class="f-wt" x="457.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 64：计算 7.0 µs，读权重 230.0 µs，通信 45.0 µs</title></rect><rect class="f-wt" x="493.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 64：计算 7.0 µs，读权重 117.2 µs，通信 52.5 µs</title></rect><rect class="f-wt" x="529.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 64：计算 7.0 µs，读权重 60.8 µs，通信 56.2 µs</title></rect><rect class="f-comm" x="565.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 64：计算 7.0 µs，读权重 32.6 µs，通信 58.1 µs</title></rect><rect class="f-comm" x="601.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 64：计算 7.0 µs，读权重 18.5 µs，通信 59.0 µs</title></rect><rect class="f-comm" x="637.0" y="341.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 64：计算 7.0 µs，读权重 11.5 µs，通信 59.5 µs</title></rect><rect class="f-wt" x="421.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP2，T = 32：计算 3.5 µs，读权重 455.6 µs，通信 15.0 µs</title></rect><rect class="f-wt" x="457.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP4，T = 32：计算 3.5 µs，读权重 230.0 µs，通信 22.5 µs</title></rect><rect class="f-wt" x="493.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP8，T = 32：计算 3.5 µs，读权重 117.2 µs，通信 26.2 µs</title></rect><rect class="f-wt" x="529.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP16，T = 32：计算 3.5 µs，读权重 60.8 µs，通信 28.1 µs</title></rect><rect class="f-wt" x="565.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP32，T = 32：计算 3.5 µs，读权重 32.6 µs，通信 29.0 µs</title></rect><rect class="f-comm" x="601.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP64，T = 32：计算 3.5 µs，读权重 18.5 µs，通信 29.5 µs</title></rect><rect class="f-comm" x="637.0" y="367.0" width="34" height="24.0" rx="3" opacity="0.88"><title>EP128，T = 32：计算 3.5 µs，读权重 11.5 µs，通信 29.7 µs</title></rect><text class="s" x="438.0" y="408.0" text-anchor="middle">2</text><text class="s" x="474.0" y="408.0" text-anchor="middle">4</text><text class="s" x="510.0" y="408.0" text-anchor="middle">8</text><text class="s" x="546.0" y="408.0" text-anchor="middle">16</text><text class="s" x="582.0" y="408.0" text-anchor="middle">32</text><text class="s" x="618.0" y="408.0" text-anchor="middle">64</text><text class="s" x="654.0" y="408.0" text-anchor="middle">128</text><text class="si" x="546.0" y="428.0" text-anchor="middle">EP</text><circle cx="474.0" cy="119.0" r="9" style="fill:var(--background);stroke:var(--text);stroke-width:1.5"/><text class="h" x="474.0" y="123.5" text-anchor="middle">P</text><circle cx="618.0" cy="301.0" r="9" style="fill:var(--background);stroke:var(--text);stroke-width:1.5"/><text class="h" x="618.0" y="305.5" text-anchor="middle">D</text><text class="si" x="92.0" y="64.0" text-anchor="end">每卡 T</text><text class="s" x="100.0" y="448.0" text-anchor="start">P：官方 prefill 配置（EP4，T = 32,768）　D：官方 decode 配置（EP64，T = 256）</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 4：每种 EP 与每卡 T 的组合下，三项中最慢的一项。左为 NVLink 900 GB/s，右为跨节点、网卡 100 GB/s；P、D 为官方的 prefill 和 decode 配置。</figcaption>
</figure>

### Prefill：读权重已不是瓶颈，取最小 EP

Prefill 每卡每步处理数万个 token，EP4 时 $r$ 已是阈值的十几倍，读权重的时间只有计算的十几分之一。继续加大 EP，能节省的读权重时间很有限，通信与计算之比却从 0.71 升到 0.83（按卡去重时也从 0.39 升到 0.61），专家负载不均的影响也在变大；官方实测的 prefill moe_lb 在 EP2 时为 1.045，EP4 时为 1.074。因此 prefill 在显存允许的范围内取最小的 EP，下限取决于每张卡能否放下 384 / EP 个专家的权重。

### Decode：读权重是瓶颈，加大 EP 直到它降到计算以下

Decode 的 $T$ 由并发数和延迟要求决定，每卡只有 256 个 token。EP8 时 $r$ 只有 32 行，读权重约需 117 µs，计算只需 28 µs。$T$ 无法增大，能把 $r$ 推过阈值的只有 EP：EP 从 8 增到 64，读权重的时间降到约 19 µs，低于计算时间。部署文档对 Wide-EP 的解释也正是这一点：它大幅减少了 MoE 专家部分每步的显存流量。官方曲线也反映了这一点（图 5），每卡 64 个请求时，EP64 每卡约 15.7K tok/s，每个用户约 246 tok/s；EP8 分别约为 12.6K 和 198。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 430" role="img" aria-label="DeepInfer official decode throughput versus per-user speed, EP64 and EP8" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><line class="grid" x1="90.0" y1="360.0" x2="640.0" y2="360.0"/><text class="s" x="82.0" y="364.0" text-anchor="end">0K</text><line class="grid" x1="90.0" y1="311.7" x2="640.0" y2="311.7"/><text class="s" x="82.0" y="315.7" text-anchor="end">5K</text><line class="grid" x1="90.0" y1="263.3" x2="640.0" y2="263.3"/><text class="s" x="82.0" y="267.3" text-anchor="end">10K</text><line class="grid" x1="90.0" y1="215.0" x2="640.0" y2="215.0"/><text class="s" x="82.0" y="219.0" text-anchor="end">15K</text><line class="grid" x1="90.0" y1="166.7" x2="640.0" y2="166.7"/><text class="s" x="82.0" y="170.7" text-anchor="end">20K</text><line class="grid" x1="90.0" y1="118.3" x2="640.0" y2="118.3"/><text class="s" x="82.0" y="122.3" text-anchor="end">25K</text><line class="grid" x1="90.0" y1="70.0" x2="640.0" y2="70.0"/><text class="s" x="82.0" y="74.0" text-anchor="end">30K</text><line class="axis" x1="90.0" y1="360.0" x2="90.0" y2="364.0"/><text class="s" x="90.0" y="378.0" text-anchor="middle">0</text><line class="axis" x1="168.6" y1="360.0" x2="168.6" y2="364.0"/><text class="s" x="168.6" y="378.0" text-anchor="middle">50</text><line class="axis" x1="247.1" y1="360.0" x2="247.1" y2="364.0"/><text class="s" x="247.1" y="378.0" text-anchor="middle">100</text><line class="axis" x1="325.7" y1="360.0" x2="325.7" y2="364.0"/><text class="s" x="325.7" y="378.0" text-anchor="middle">150</text><line class="axis" x1="404.3" y1="360.0" x2="404.3" y2="364.0"/><text class="s" x="404.3" y="378.0" text-anchor="middle">200</text><line class="axis" x1="482.9" y1="360.0" x2="482.9" y2="364.0"/><text class="s" x="482.9" y="378.0" text-anchor="middle">250</text><line class="axis" x1="561.4" y1="360.0" x2="561.4" y2="364.0"/><text class="s" x="561.4" y="378.0" text-anchor="middle">300</text><line class="axis" x1="640.0" y1="360.0" x2="640.0" y2="364.0"/><text class="s" x="640.0" y="378.0" text-anchor="middle">350</text><line class="axis" x1="90.0" y1="360.0" x2="640.0" y2="360.0"/><text class="si" x="365.0" y="400.0" text-anchor="middle">每个用户的速度（tok/s）</text><text class="si" x="82.0" y="56.0" text-anchor="start">每卡吞吐（tok/s/GPU）</text><line class="ln comp" x1="90" y1="26.0" x2="116.0" y2="26.0"/><text class="t" x="122.0" y="30.0" text-anchor="start">EP64</text><line class="ln wt dash" x1="180.8" y1="26.0" x2="206.8" y2="26.0"/><text class="t" x="212.8" y="30.0" text-anchor="start">EP8</text><polyline class="ln comp" points="600.7,334.9 585.0,311.2 556.7,268.2 476.6,208.2 369.7,139.6 251.9,105.8"/><circle class="dot f-comp" cx="600.7" cy="334.9" r="4"><title>EP64，每卡 8 个请求：每用户 325 tok/s，每卡 2,600 tok/s</title></circle><circle class="dot f-comp" cx="585.0" cy="311.2" r="4"><title>EP64，每卡 16 个请求：每用户 315 tok/s，每卡 5,050 tok/s</title></circle><circle class="dot f-comp" cx="556.7" cy="268.2" r="4"><title>EP64，每卡 32 个请求：每用户 297 tok/s，每卡 9,500 tok/s</title></circle><circle class="dot f-comp" cx="476.6" cy="208.2" r="5.5"><title>EP64，每卡 64 个请求：每用户 246 tok/s，每卡 15,700 tok/s</title></circle><circle class="dot f-comp" cx="369.7" cy="139.6" r="4"><title>EP64，每卡 128 个请求：每用户 178 tok/s，每卡 22,800 tok/s</title></circle><circle class="dot f-comp" cx="251.9" cy="105.8" r="4"><title>EP64，每卡 256 个请求：每用户 103 tok/s，每卡 26,300 tok/s</title></circle><polyline class="ln wt dash" points="464.0,341.6 451.4,324.2 435.7,291.9 401.1,237.7 325.7,174.4 244.0,119.3"/><circle class="dot f-wt" cx="464.0" cy="341.6" r="4"><title>EP8，每卡 8 个请求：每用户 238 tok/s，每卡 1,900 tok/s</title></circle><circle class="dot f-wt" cx="451.4" cy="324.2" r="4"><title>EP8，每卡 16 个请求：每用户 230 tok/s，每卡 3,700 tok/s</title></circle><circle class="dot f-wt" cx="435.7" cy="291.9" r="4"><title>EP8，每卡 32 个请求：每用户 220 tok/s，每卡 7,050 tok/s</title></circle><circle class="dot f-wt" cx="401.1" cy="237.7" r="5.5"><title>EP8，每卡 64 个请求：每用户 198 tok/s，每卡 12,650 tok/s</title></circle><circle class="dot f-wt" cx="325.7" cy="174.4" r="4"><title>EP8，每卡 128 个请求：每用户 150 tok/s，每卡 19,200 tok/s</title></circle><circle class="dot f-wt" cx="244.0" cy="119.3" r="4"><title>EP8，每卡 256 个请求：每用户 98 tok/s，每卡 24,900 tok/s</title></circle><text class="t" x="486.6" y="200.2" text-anchor="start">64 个请求：15.7K，246</text><text class="t" x="389.1" y="242.7" text-anchor="end">64 个请求：12.6K，198</text><text class="s" x="90.0" y="420.0" text-anchor="start">数据从 DeepInfer README 的 decode 曲线读出，为近似值；context 131,072，nextn = 3。</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 5：DeepInfer 官方的 decode 曲线，从 README 的图中读出，为近似值。</figcaption>
</figure>

$r$ 越过 $r^{*}$ 之后，读权重已降到计算以下，再加大 EP 也省不下时间：计算保持在 28 µs，通信和专家负载不均却继续加重。按本文的估算，这个交点在 EP32 与 EP64 之间，官方推荐的 EP64 到 EP128 正好在它之后不远。在这个范围内，通信与计算之比约为 0.93，每层还要与 63 张以上的卡同步，一个负载为平均值 10 倍的专家就能让 $lb$ 达到 2.5。因此 decode 需要放在 64 到 128 卡的 NVLink 域内，并依赖 EPLB 复制负载高的专家、定期重排，官方实测的 decode moe_lb 为 1.090。

<figure id="epx" style="margin:1.8em 0">
<style>
#epx .row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 10px;font-size:0.92em}
#epx .lab{min-width:3.2em;color:var(--secondary);font-style:italic}
#epx button{font:inherit;font-size:0.92em;padding:2px 10px;border-radius:4px;border:1px solid var(--quiet-line);background:transparent;color:var(--text);cursor:pointer}
#epx button.on{border-color:var(--technical);color:var(--technical)}
#epx .axis{position:relative;height:46px;margin:16px 0 4px}
#epx .track{position:absolute;left:0;right:0;top:20px;height:2px;background:var(--quiet-line)}
#epx .thr{position:absolute;top:8px;width:0;height:26px;border-left:1.5px dashed var(--reflection)}
#epx .thrt{position:absolute;top:-12px;font-size:0.8em;font-style:italic;color:var(--reflection);transform:translateX(-50%);white-space:nowrap}
#epx .dot{position:absolute;top:13px;width:16px;height:16px;border-radius:50%;background:var(--technical);transform:translateX(-50%);transition:left .3s}
#epx .ticks{position:relative;height:18px;font-size:0.78em;color:var(--secondary)}
#epx .ticks span{position:absolute;transform:translateX(-50%)}
#epx .bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:0.92em}
#epx .bar .n{min-width:4em;color:var(--secondary)}
#epx .bar .w{flex:1;height:14px}
#epx .bar .f{height:14px;border-radius:3px;transition:width .3s}
#epx .bar .v{min-width:5.5em;text-align:right;font-variant-numeric:tabular-nums}
#epx .out{margin-top:10px}
</style>
<div class="row"><span class="lab">EP</span><span data-k="ep"></span></div>
<div class="row"><span class="lab">每卡 T</span><span data-k="t"></span></div>
<div class="axis"><div class="track"></div><div class="thr" data-k="thr"><span class="thrt">阈值 ≈ 150 行</span></div><div class="dot" data-k="dot"></div></div>
<div class="ticks" data-k="ticks"></div>
<div class="bar"><span class="n">计算</span><div class="w"><div class="f" data-k="bc" style="background:var(--technical)"></div></div><span class="v" data-k="vc"></span></div>
<div class="bar"><span class="n">读权重</span><div class="w"><div class="f" data-k="bw" style="background:var(--reflection)"></div></div><span class="v" data-k="vw"></span></div>
<div class="bar"><span class="n">通信</span><div class="w"><div class="f" data-k="bm" style="background:var(--secondary)"></div></div><span class="v" data-k="vm"></span></div>
<p class="out" data-k="out"></p>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 6：每层每卡的三项时间（B200，NVLink 900 GB/s）。圆点是每个专家分到的行数，横轴按对数刻度。</figcaption>
<script>
(function(){
const root=document.getElementById('epx');
const q=k=>root.querySelector('[data-k="'+k+'"]');
const EPS=[2,4,8,16,32,64,128],TS=[[256,'256'],[1024,'1K'],[4096,'4K'],[32768,'32K']];
const FL=70778880,F=4.5e15,HBM=8e12,NV=900e9,EB=18.8e6,SB=35.4e6,REC=15616,THR=150,LO=1,HI=100000;
let ep=8,ti=0;
const pos=v=>Math.log(v/LO)/Math.log(HI/LO)*100;
function buttons(k,list,cur,set){const el=q(k);el.innerHTML='';list.forEach((v,i)=>{const b=document.createElement('button');b.textContent=Array.isArray(v)?v[1]:'EP'+v;if(i===cur)b.className='on';b.onclick=()=>{set(i);draw();};el.appendChild(b);});}
q('ticks').innerHTML=[1,10,100,1000,10000,100000].map(v=>'<span style="left:'+pos(v)+'%">'+v.toLocaleString()+'</span>').join('');
q('thr').style.left=pos(THR)+'%';
function draw(){
buttons('ep',EPS,EPS.indexOf(ep),i=>ep=EPS[i]);
buttons('t',TS,ti,i=>ti=i);
const T=TS[ti][0],per=384/ep,rows=ep*T*6/384;
const copies=6*(1-1/ep);
const tc=T*7*FL/F*1e6,tw=(per*EB+SB)/HBM*1e6,tm=T*copies*REC/NV*1e6;
const mx=Math.max(tc,tw,tm),fmt=v=>v>=1000?(v/1000).toFixed(2)+' ms':v.toFixed(v<10?1:0)+' µs';
q('dot').style.left=Math.min(100,pos(Math.max(rows,LO)))+'%';
[['bc','vc',tc],['bw','vw',tw],['bm','vm',tm]].forEach(([b,v,x])=>{q(b).style.width=(x/mx*100)+'%';q(v).textContent=fmt(x);});
const r=rows/THR,big=[['计算',tc],['读权重',tw],['通信',tm]].sort((a,b)=>b[1]-a[1])[0][0];
q('out').textContent='每卡 '+per+' 个专家，每个专家分到 '+Math.round(rows).toLocaleString()+' 行，是阈值的 '+(r<1?r.toFixed(2):r.toFixed(1))+' 倍；最长的一项是'+big+'。';
}
draw();
})();
</script>
</figure>

在图 6 中可以自己验证这两种取舍：$T$ 取 32K 时，无论 EP 取多少，三项中都是计算最长；$T$ 取 256 时，随着 EP 从 8 增大，读权重的时间迅速下降，直到 $r$ 越过阈值，此后通信与计算接近，继续加大 EP 已省不下时间。

## 五、阈值随硬件和精度的变化

阈值 $r^{*}$ 只由权重精度和硬件的算力带宽比决定。图 7 列出了几种常见组合：FP4 权重的字节数是 FP8 的一半，阈值也减半，这是 V4.1 在 B200 上只需约 150 行的原因；如果激活也用 FP4，算力翻倍，阈值又回到约 300 行。H20 的算力相对显存带宽很低，阈值只有约 37 行。

<figure style="margin:1.8em 0">
<svg xmlns="http://www.w3.org/2000/svg" class="ev" viewBox="0 0 720 406" role="img" aria-label="Weight-reuse threshold r* by GPU and weight precision" style="width:100%;height:auto;display:block"><style>
.ev text{font-family:'Spectral',Georgia,'Songti SC',STSong,serif}
.ev .t{fill:var(--text);font-size:13px}
.ev .s{fill:var(--secondary);font-size:12px}
.ev .si{fill:var(--secondary);font-size:12.5px;font-style:italic}
.ev .h{fill:var(--text);font-size:13.5px;font-weight:500}
.ev .grid{stroke:var(--quiet-line);stroke-width:1}
.ev .axis{stroke:var(--secondary);stroke-width:1}
.ev .ref{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.ev .ref2{stroke:var(--secondary);stroke-width:1.2;stroke-dasharray:1.5 3.5;fill:none}
.ev .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ev .dash{stroke-dasharray:7 5}
.ev .dot{stroke:var(--background);stroke-width:2}
.ev .band{fill:var(--quiet-line)}
.ev .comp{stroke:#215d96}.ev .f-comp{fill:#215d96}
.ev .wt{stroke:#b45151}.ev .f-wt{fill:#b45151}
.ev .comm{stroke:#b7933f}.ev .f-comm{fill:#b7933f}
.ev .ink{stroke:var(--text)}.ev .f-ink{fill:var(--text)}
html[data-theme="dark"] .ev .comp{stroke:#5e97d0}html[data-theme="dark"] .ev .f-comp{fill:#5e97d0}
html[data-theme="dark"] .ev .wt{stroke:#b14e4e}html[data-theme="dark"] .ev .f-wt{fill:#b14e4e}
html[data-theme="dark"] .ev .comm{stroke:#b48e34}html[data-theme="dark"] .ev .f-comm{fill:#b48e34}
</style><text class="si" x="250.0" y="34.0" text-anchor="start">权重复用阈值 r*（行）＝ 每参数字节 ÷ 2 × 稠密算力 ÷ 显存带宽</text><line class="grid" x1="250.0" y1="52.0" x2="250.0" y2="336.0"/><text class="s" x="250.0" y="352.0" text-anchor="middle">0</text><line class="grid" x1="375.0" y1="52.0" x2="375.0" y2="336.0"/><text class="s" x="375.0" y="352.0" text-anchor="middle">100</text><line class="grid" x1="500.0" y1="52.0" x2="500.0" y2="336.0"/><text class="s" x="500.0" y="352.0" text-anchor="middle">200</text><line class="grid" x1="625.0" y1="52.0" x2="625.0" y2="336.0"/><text class="s" x="625.0" y="352.0" text-anchor="middle">300</text><text class="t" x="238.0" y="80.0" text-anchor="end">H20，FP8 权重</text><rect class="f-ink" x="250" y="66.0" width="46.2" height="20" rx="4" opacity="0.55"><title>H20，FP8 权重：r* ≈ 37 行</title></rect><text class="t" x="304.2" y="81.0" text-anchor="start">37</text><text class="t" x="238.0" y="126.0" text-anchor="end">B200，FP4 权重 × FP8 激活</text><rect class="f-comp" x="250" y="112.0" width="186.8" height="20" rx="4" opacity="1"><title>B200，FP4 权重 × FP8 激活：r* ≈ 149 行</title></rect><text class="t" x="444.8" y="127.0" text-anchor="start">149</text><text class="t" x="238.0" y="172.0" text-anchor="end">H200，FP8 权重</text><rect class="f-ink" x="250" y="158.0" width="257.7" height="20" rx="4" opacity="0.55"><title>H200，FP8 权重：r* ≈ 206 行</title></rect><text class="t" x="515.7" y="173.0" text-anchor="start">206</text><text class="t" x="238.0" y="218.0" text-anchor="end">B200，FP8 权重</text><rect class="f-ink" x="250" y="204.0" width="351.6" height="20" rx="4" opacity="0.55"><title>B200，FP8 权重：r* ≈ 281 行</title></rect><text class="t" x="609.6" y="219.0" text-anchor="start">281</text><text class="t" x="238.0" y="264.0" text-anchor="end">H100，FP8 权重</text><rect class="f-ink" x="250" y="250.0" width="369.2" height="20" rx="4" opacity="0.55"><title>H100，FP8 权重：r* ≈ 295 行</title></rect><text class="t" x="627.2" y="265.0" text-anchor="start">295</text><text class="t" x="238.0" y="310.0" text-anchor="end">B200，FP4 权重 × FP4 激活</text><rect class="f-ink" x="250" y="296.0" width="373.5" height="20" rx="4" opacity="0.55"><title>B200，FP4 权重 × FP4 激活：r* ≈ 299 行</title></rect><text class="t" x="631.5" y="311.0" text-anchor="start">299</text><text class="s" x="250.0" y="376.0" text-anchor="start">蓝色为本文的配置。按公开的稠密峰值和显存带宽计算。</text></svg>
<figcaption style="color:var(--secondary);font-size:0.9em;text-align:center;margin-top:0.6em">图 7：不同 GPU 与权重精度下的权重复用阈值，按公开的稠密峰值和显存带宽计算。</figcaption>
</figure>

令 $r = r^{*}$，可以得到 decode 越过阈值所需的最小 EP：

$$
EP_{\min} = \frac{r^{*} \times 384}{6 \times T}
$$

每卡 256 个 token 时，B200 上的 FP4 专家约需 EP38，H100 上的 FP8 专家约需 EP74，H20 上约需 EP9。同样的模型和负载，在 H20 上把 EP 开到 8 到 16 就基本越过了阈值，继续加大 EP 能省下的时间很有限；在 H100 上则要开得比 B200 更大，而一台 H100 机器只有 8 张卡，EP74 必须跨节点，通信又会变成新的瓶颈，情形与图 4 右侧类似。换到别的模型时，把 384 和 6 换成它的路由专家数和每个 token 选择的专家数即可。

## 六、估算与实际情况的差距

前面的数字都按理想情况估算：算力和显存带宽取峰值，路由完全均匀，通信可以完全藏在计算后面，并且只看 MoE 层。实际系统在每一项上都会打折扣，但这些折扣大多改变数值，不改变比较的方向。

矩阵乘和显存读写都达不到峰值。阈值 $r^{*}$ 取决于两者之比，算力跑到峰值的七成、带宽跑到八成五时，阈值约为 120 行；计入每行激活的读写，又会把它抬高约两成。阈值在 120 到 180 行之间浮动，decode 越过它所需的 EP 在 30 到 45 之间：EP8 远远不够，EP64 足够，结论不变。

路由也不会完全均匀。专家负载不均让最忙的卡多算、多收，计算和通信按同样的倍数放大，两者之比不变；官方开启 EPLB 后实测的放大倍数在 1.045 到 1.090 之间，影响有限。

通信也不一定能完全藏在计算后面。每层的同步和启动都有一段固定开销，与要传多少数据无关。prefill 每层的计算约 3.6 ms，这段开销可以忽略；decode 每层的计算只有约 28 µs，它的占比就不小了，而且参与同步的卡越多越明显，这也是 decode 的 EP 停在 64 到 128、不再继续加大的原因之一。

最后，本文只看 MoE 层。官方曲线上，decode 每卡 64 个请求时一步约 13 ms，MoE 层只占其中一部分，其余是注意力和大量与批量无关的固定开销。所以本文的数字适合用来判断不同 EP 下谁是瓶颈、EP 该往哪个方向调，不能直接当作每步的实际耗时；官方 EP64 与 EP8 的实测差距，方向与本文的推导一致。注意力和这些固定开销如何影响 EP 的选择，留待后续分析。

## 小结

Prefill 与 decode 的 EP 选择看似相反，遵循的却是同一个规律：EP 不改变每张卡的计算量，只改变每个专家的行数、通信份数和专家负载不均。每个专家的行数随 EP × T 变化，决定读权重的时间；通信份数和专家负载不均只随 EP 变化。

- **Prefill**：$T$ 足够大，$r$ 早已越过阈值，加大 EP 只会增加通信、加重专家负载不均，所以取显存允许的最小值。
- **Decode**：$T$ 受并发和延迟限制，只能用 EP 把 $r$ 推过阈值，直到读权重的时间降到计算以下，所以取 64 到 128，并以 NVLink 域承担增加的通信，以 EPLB 缓解专家负载不均。

这个判断并不只适用于 V4.1。DeepSeek 在 V3/R1 的推理系统中把 decode 的 EP 开到 144，给出的理由同样是让每个专家分到足够大的 batch；NVIDIA 在 GB200 NVL72 上推行 Wide-EP，MegaScale-Infer 把注意力和专家拆到不同的 GPU 上汇聚 token，出发点也相同。换到别的模型和硬件，用第五节的两个式子，就能大致判断 EP 该往哪个方向调。


## 参考

1. DeepInfer, [Inference deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)
2. DeepInfer, [README: Performance benchmarks](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)
3. DeepSeek, [Day 6: DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
4. NVIDIA, [Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack Scale Systems](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems)
5. Zhu et al., [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263)
