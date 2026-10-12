# Coordinate checks for SwissAI-DLM: width, depth, and width-depth co-scaling

**This document is self-contained.** It specifies the measurements that decide whether a
width/depth scaling implementation actually works. Everything needed to run them is here:
the quantity, channel, norm, time window, ladder, arms and pass criterion for each axis,
plus the no-op tests. It is written to be handed to an agent and executed. The failure
modes listed are ones we hit and paid for, not hypotheticals.

A separate document, `SwissAI-DLM_width_depth_scaling.md`, specifies the *code changes*
these checks verify. You do not need it to run what is here, but §0a summarises the knobs
the checks switch on and off.

Sources: CompleteP (Dey et al., *Don't be lazy: CompleteP enables compute-efficient deep
transformers*, arXiv:2505.01618), **Figure 7**; the refined coordinate check (Haas et al.,
*On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling*,
arXiv:2505.22491); and the measured results from the AdamW BST scaling project.

---

## 0. The single most important thing

**Width, depth and co-scaling are three DIFFERENT measurements, not one measurement on
three ladders.** Applying the width recipe to a depth ladder is not a weaker check, it is
the *wrong quantity*. We learned this expensively: on BST the per-module fit gave depth
"worst |exponent| 0.780 ± 0.24", while the correct quantity gave **+0.901 ± 0.011** — a
9x tighter band and a clean verdict instead of a shrug.

| | **width** | **depth** | **co-scale** |
|---|---|---|---|
| quantity | `‖(W_t − W_0)·x_t‖`, **per module** | `‖h_L^t − h_L^0‖`, **one scalar** | same as depth |
| norm | `rmsnorm` | **`l2norm`** | **`rmsnorm`** |
| time window | mean over the **last 10%** of steps | **steps 2…10** | **steps 2…10** |
| x-axis | width, log2 | depth L, log2 | width, log2 (L tied to it) |
| pass | every exponent ≈ 0 | flat line in L | flat line |

The norm row is not pedantry. At fixed width `l2 = rms·√w`, a constant, so the depth
exponent is identical either way. **Under co-scaling the width moves too, and `l2norm` then
adds a spurious +0.5 per width doubling.** Print which norm you used; never leave it
implicit.

---

## 0a. What is being tested, in one page

These checks verify a muP-style parametrisation on three axes. The knobs they switch are:

| knob | meaning | off = |
|---|---|---|
| `scaling.enabled` | per-tensor init and LR laws derived from each tensor's fan-in, instead of one global rule | today's behaviour, byte-identical |
| `scaling.base_width` | the width the LR was tuned at; the laws reproduce exactly that LR at this rung | — |
| `depth_scaling` | each residual **branch output** multiplied by `base_depth / n_layer`, so the residual stream stays Theta(1) as L grows (CompleteP) | multiplier 1.0 |
| `scaling.base_depth` | the depth the LR was tuned at | — |
| `peri_ln` | an RMSNorm on each residual **branch output** (not its input), which is what makes depth scaling behave | `nn.Identity`, an exact no-op |
| `scaling.output_fwd_scale` | `1/m` applied to the readout's input; required if embeddings and readout are tied | 1.0 |

The three things a correct implementation must achieve, and which check proves each:

1. **Width**: every tensor's activation, propagated change and update stay the same size as
   the model widens. -> §3.
2. **Depth**: the residual stream's movement stays Theta(1) as L grows -- neither vanishing
   (the "lazy" regime) nor exploding. -> §4.
3. **Composition**: 1 and 2 hold simultaneously on the ladder real runs use. -> §5.

If you are working on a different codebase, the only repo-specific things below are the
module names in §1 (`output_layer.norm_final`, `blocks.<l>.norm1`) and the `cond_dim`
caveat in §3. Substitute your own residual-stream module and everything else carries over.

## 1. Instrumentation — install it, do not write it

SwissAI-DLM currently has **no activation instrumentation at all** (no forward hooks, no
module monitor, no coordinate-check code). Do not hand-roll it.

```bash
pip install torch-module-monitor
```

It implements the RCC from Haas et al. 2025 and emits exactly the channels below. BST used
it unmodified. Hook the model once, after construction:

```python
from torch_module_monitor import Monitor
monitor = Monitor(model, log_every=1, modules=r".*")   # regex filter
```

Channels you will need (names as emitted):

```
activation/<module>/rmsnorm                 # forward activation RMS
RCC (x_t-x_0)/<module>/rmsnorm              # change in the module's INPUT since step 0
RCC (x_t-x_0)/<module>/l2norm               # same, Frobenius
```

**Verify the channel exists before trusting any result.** A set flag is not an engaged
flag:

```bash
grep -c 'RCC (x_t-x_0)/output_layer.norm_final' module_monitor.jsonl   # 0 means absent
```

### Running it on the GIDD-HF path (what is implemented)

`scripts/analysis/coordinate_check.py` (run) + `coordinate_check_fit.py` (fit) +
`scripts/training/submit_coordinate_check.sh` (sbatch) implement sections 3-6 on the real
training path, with their own hooks (no `torch-module-monitor` dependency). Module names there:
residual stream `h_L` = input of `model.model.norm`; stream entering block `l` = input of
`model.model.layers.<l>.attn_layernorm`. The GIDD-HF model has no adaLN / `cond_dim`, so the
`cond_dim` caveat of section 3 does not apply; `head_dim` is fixed at 64 by the script.

**Arm mapping.** `completep`, `mup` and `standard` all carry a `resid_scale / n_blocks` branch
multiplier, which is *already* a depth control, so they cannot serve as the `off` arm as they
stand. The script defines `off` as `mup` (`standard` on the width axis) with
`model.resid_scale = n_blocks`, i.e. a branch multiplier of exactly 1.0 (the reference's
`depth_scaling=False`); `on` is `completep` with `resid_mult=1`, multiplier `base_depth / L`.
At `L == base_depth` both are 1.0 and every LR/eps scale is equal, so the base-rung no-op
(6b) is `--axis noop`.

### Which module is the residual stream?

The depth and co-scale checks need `h_L`, the residual stream after the **last block's
merge**. In this repo that is the **input to `output_layer.norm_final`**
(`models/common.py:329`), which is the analogue of nanoGPT's `transformer.ln_f`. Per-layer,
the stream entering block `l` is the input to `blocks[l].norm1`.

```
h_L            = RCC (x_t-x_0)/output_layer.norm_final/l2norm
stream at l    = RCC (x_t-x_0)/blocks.<l>.norm1/l2norm
```

⚠️ **Baseline differs from the paper by one step.** The channel is `x_t − x_0`, so you get
`‖h_L^t − h_L^0‖`; CompleteP Fig 7 plots `‖h_L^t − h_L^1‖`. Measured on BST,
`‖h_L^1−h_L^0‖ = 6.7e−1` against `‖h_L^2−h_L^0‖ = 2.0e+2`, so the baselines differ by 0.3%
at t=2 and less after. Report it; do not silently assume it away.

---

## 2. Always run two arms

One arm can only say "the line looks flat". Two say whether the *correction* is what
flattened it. This maps onto CompleteP Fig 7's columns:

| arm | setting | Fig 7 column |
|---|---|---|
| `off` | `depth_scaling=False` (and `scaling.enabled=False` for width) | **column 1** — SP/muP, no depth control |
| `on` | `depth_scaling=True` with the right `base_depth` | **column 4** — alpha = 1.0 |

Measured on BST (w512, L ∈ {8,16,32,64}, 4 seeds, lr 7.74264e-3):

```
arm    mean |exponent| (t=2..10)    worst
off            0.901                +0.924     <- grows as ~L^0.9, column 1 reproduced
on             0.101                -0.151     <- flat,            column 4 reproduced
```

and on the co-scaled ladder (w/L = 64, w ∈ {512,768,1024,1536}, `rmsnorm`):
**off 0.896 → on 0.105**. Seed bands ±0.01, so the arms are ~80 seed-sigmas apart.
**That is what a real depth result looks like. If your bands are ±0.2 you are fitting the
wrong quantity** — go back to §0.

---

## 3. The width check

**Ladder.** `hidden_size ∈ {256, 512, 1024, 2048}` with `base_width = 256`, `head_dim`
FIXED (64) and `n_heads = hidden_size // head_dim` derived. ≥4 rungs, ≥4 seeds.

**Do not put a rung below ~512 if `cond_dim` stays 128.** A "fixed" dimension comparable to
the width means that rung is not in the asymptotic regime and will dominate the fit. We
measured this: on our own Mamba-3 problem, where `head_dim` 64 and `state_size` 128 are
fixed, dropping the 256 rung moved the score from 0.5797 to 0.8055 and the limiting channel
from 0.981 to 0.316. The base rung was not measuring the parametrisation.

**Quantity.** Per module, the end value of each channel, as the mean over the **last 10%**
of steps, averaged over seeds; fit `slope of log2(value) vs log2(width)`.

**Report all three channels, not just activations:**

```
act   activation/<module>/rmsnorm                     is the output the right size?
prop  RCC (x_t-x_0)/<module>/rmsnorm                  is the propagated change right?
eff   ‖(W_t − W_0)·x_t‖ / rmsnorm                     is the update right?
```

A systematic drift can sit entirely in `prop` and be invisible in `act`. On our own DiT
problem the five worst channels were all `adaln_*|prop` at +0.105…+0.120 while every `act`
was under 0.067 — an act-only figure would have shown nothing.

**Pass:** every exponent ≈ 0. Judge against the **seed scatter**, not against zero: with 4
seeds we have problems whose worst channel sits within 2 standard errors of zero, where the
difference is not real. Compute `t = |exponent| / (seed_sd/√n_seeds)` and treat `t < 2` as
unresolved rather than as a pass or a fail.

---

## 4. The depth check — CompleteP Figure 7

**Quantity.** `‖h_L^t − h_L^0‖₂` — **one scalar for the whole network**, the last layer's
residual stream. Not per-module. Depth control acts on the residual *branches*, so what it
controls is the residual *stream*; a per-module weight-update norm is a different object
that can look fine while the stream explodes.

**Norm:** `l2norm`. **Window:** steps **2…10**.

**Ladder.** Fixed width (512 is enough), `L ∈ {8, 16, 32, 64}`, ≥3 seeds.

**Pass:** the mean |exponent| over t = 2…10 is flat in L. CompleteP's own framing, which
should be yours: *"This test is a necessary but not sufficient condition for satisfying
Desiderata 2."* Desideratum 2 is Maximal Residual Stream Update — the function must move by
Θ(1) as depth grows, neither vanishing (the "lazy" regime the paper is named against) nor
exploding.

**Depth instability is an EARLY phenomenon — visible by step 2.** Two consequences:

* A last-10% read can hide it behind a run that has already diverged or saturated.
* **A depth coordinate check is answerable within MINUTES of launch**, because steps 2–10
  arrive immediately. Use that to fail fast; do not wait for the run.

### Ladder rules that are not optional

* **`base_depth` must match your base model, and the default will not.** Set it per ladder
  and assert it. If your tuned base is L12 and `base_depth` defaults to 8, the base model
  gets `depth_scale = 8/12` and stops being the model you tuned.
* **No rung may sit AT `base_depth`.** There the multiplier is exactly 1, no correction is
  applied, and that rung tests nothing — the same trap as the muP base *width*. Want **≥3
  rungs strictly above `base_depth`**.

---

## 5. The co-scaling check

**Ladder.** Hold the aspect ratio fixed — BST used `width/L = 64`, i.e.
`w ∈ {512, 768, 1024, 1536}` with `L = w/64`. x-axis is width (log2), with L moving in
lockstep.

**Norm: `rmsnorm`, NOT `l2norm`.** This is the one that bites. `l2 = rms·√w`, so on a ladder
where width moves, `l2norm` adds a spurious **+0.5 per width doubling** and you will
"discover" an instability that is an artefact of your norm choice.

**What co-scale can and cannot tell you.** A co-scale ladder that must contain your base
model cannot satisfy the "≥3 rungs above `base_depth`" rule on its depth component: a w/L=64
ladder containing w768_L12 has `L ∈ {8,12,16,24}` — only two rungs above base_depth=12, and
one sits exactly on it. The next aspect-64 rung up is a 1.82 B-parameter model. This is
inherent, not an oversight.

> **Certify depth on the PURE-DEPTH axis. Use co-scale only to show that width and depth
> compose on the ladder the science runs actually use. Never quote a flat co-scale result as
> independent evidence about depth.**

---

## 6. The no-op tests — run these first, they are free and decisive

### 6a. Scaling disabled is byte-identical

With `scaling.enabled=False`, `peri_ln=False`, `depth_scaling=False`, assert the state-dict
keys, the parameter count and a fixed-seed forward pass are **bit-identical** to the
pre-change model. `peri_ln=False` must be `nn.Identity`, an exact no-op. This is what lets
the change land without re-validating existing results.

### 6b. ★ The base-rung no-op test — run it before believing ANY depth result

At `L == base_depth` the multiplier is exactly 1 (`12/12 == 1.0`, and `x * 1.0` is
bit-exact), so **`depth_scaling=True` must train IDENTICALLY to `depth_scaling=False`**. If
it does not, the flag is doing something other than what it claims, and every depth
conclusion built on it is suspect.

It is free: a co-scale ladder contains the base model in *both* arms at matched seeds, so
the control is already on disk.

Measured on BST (w768_L12, base_depth=12, 4 seeds):

```
step 0        BITWISE identical -- train/loss, activation rms, RCC x_t, grad_norm, all seeds
t = 2..10     worst rel diff 1.2e-05        <- the window the verdict is read from
t = 40        1.8e-04
t >= 80       up to 3.4e-01                 <- 34%
```

**The late divergence is NOT a depth_scaling effect.** Step 0 is bitwise identical, so the
two are the same computation. GPU nondeterminism (flash-attention backward atomics, TF32)
seeds ~1e-6 at the first optimizer step, and a **constant-peak-LR** run amplifies it
chaotically. Confirming signature: the sign of the loss difference is **mixed** across
seeds (−0.022, +0.011, −0.042, +0.011); a real implementation difference shifts every seed
the same way.

**This is also the strongest argument for reading the depth check early.** Two
bit-identical runs at constant peak LR carry ~10% spread in `‖Δh_L‖` by step 100:

| read | seed band on the exponent |
|---|---|
| Fig-7 window t = 2…10 | **±0.01** |
| per-module `eff`, last 10% of 500 steps | ±0.13 … ±0.27 |

A late read of *any* coordinate-check quantity is measuring chaos as much as
parametrisation. Note the contrast with a schedule that has a cooldown: there, two
mathematically-identical arms *contract* (sd 0.0098 → 0.0038 over 24k steps) because the
decaying LR pulls trajectories together. **"Identical runs stay close" is a property of the
SCHEDULE, not a given.** Check it before using late-step agreement as evidence of anything.

### 6c. Classification coverage

Iterate `named_parameters()` on every config in `configs/model/` and assert the parameter
classifier raises for none of them. Make the classifier **raise** on an unknown name rather
than defaulting — a silently mis-bucketed tensor is a bug you find three weeks later.

This is not hypothetical for this repo. Its current param-group matcher looks for
`embed_tokens`/`lm_head`, inherited verbatim from the JAX upstream, while the native modules
are `vocab_embed`/`output_layer`. Nothing matches, nothing raises, and the embedding and
readout silently receive the hidden-weight learning rate. Upstream's own `label_leaf`
*does* `raise ValueError` on an unknown parameter; the port replaced that with a silent
fallthrough. **This test is the one that would have caught it on day one.**

---

## 7. Runbook

1. `pip install torch-module-monitor`; hook the model; **grep the channel** to confirm it
   is emitted (§1).
2. Run **6a** and **6c**. Both are CPU-only and cost nothing. Do not proceed past a failure.
3. **Width**, both arms: 4 rungs × 4 seeds × ~100 steps. Fit per module, `rmsnorm`,
   last 10%. Report act / prop / eff and the per-channel `t` statistic.
4. **Depth**, both arms: L ∈ {8,16,32,64} at fixed width, ≥3 seeds. Fit
   `‖h_L^t − h_L^0‖` with `l2norm` over t = 2…10. **Read it minutes after launch.**
5. Run **6b** on the base rung before believing step 4.
6. **Co-scale**, both arms: w/L fixed, `rmsnorm`, same window. Present as composition
   evidence only.
7. Plot in the Fig-7 layout: one panel per arm, log-log, one line per step t, a seed band,
   and a dashed flat reference.

## 8. Pass criteria, in one table

| check | quantity | norm | window | pass |
|---|---|---|---|---|
| width | per-module act / prop / eff | `rmsnorm` | last 10% | all \|exponent\| ≈ 0, judged against seed scatter (`t ≥ 2` to be real at all) |
| depth | `‖h_L^t − h_L^0‖` | `l2norm` | t = 2…10 | mean \|exponent\| flat in L; `off` arm should FAIL (≈0.9) or the ladder proves nothing |
| co-scale | `‖h_L^t − h_L^0‖` | `rmsnorm` | t = 2…10 | flat; composition evidence only |
| no-op (scaling off) | full state + forward | — | step 0 | **bit-identical** |
| no-op (base rung) | full state + forward | — | step 0 | **bit-identical**; drift after t≈40 is GPU nondeterminism, check the sign is mixed across seeds |

A final warning worth repeating because it is the most common way these go wrong: **if the
`off` arm also looks flat, you have not proved the correction works — you have proved your
ladder cannot see the effect.** The `off` arm failing is part of the pass criterion.
