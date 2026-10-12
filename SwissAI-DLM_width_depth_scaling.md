# Adding width + depth scaling (muP / CompleteP) and Peri-LN to SwissAI-DLM

Target: `https://github.com/Zahra-D/SwissAI-DLM-test` @ `main` (read 2026-10-08).
Reference implementation: the BST scaling project, `code/moe_scaling/{scaling.py,model.py}`,
which already does width-muP + depth scaling + Peri-LN for AdamW on a GPT/MoE backbone.

This is a change specification. Nothing here was executed against the repo.

---

## 0. The short version

The codebase has **one** width-scaling gesture — `bulk_lr = base_lr / hidden_size` in
`algorithms/base.py:_build_adamw_jax_style_param_groups` — and it is applied through a
name matcher that **does not match this repo's own modules**. There is no depth scaling of
any kind, no width-aware initialisation, and no Peri-LN. The net effect is that neither
learning rate nor initialisation transfers across width, and nothing at all controls the
residual stream as depth grows.

Six defects, in descending order of how much they break transfer:

| # | Defect | Where |
|---|---|---|
| 1 | Embedding/readout never reach their own param group; they get `lr/hidden_size` | `algorithms/base.py:424-430` |
| 2 | No width-aware init: embeddings use `kaiming_uniform_(a=√5)`, so init std **shrinks** as `1/√dim` | `models/common.py:300` |
| 3 | No residual-branch depth scaling: the residual stream grows with `L` | `models/block_dit.py:209-218` |
| 4 | No Peri-LN: nothing normalises the **branch output** | `models/block_dit.py` |
| 5 | Attention uses `1/√head_dim` while `head_dim` varies across configs | `models/block_dit.py:167,316` |
| 6 | Width/depth scaling logic would have to be duplicated into 4 model files | `models/*.py` |

Weight tying is **not** on that list: it is compatible with muP provided the output scaling
moves into a forward multiplier (§1.7). It becomes a defect only if combined with the
zero-init readout, which would zero the embedding.

---

## 0b. Provenance: this is a partial port of `dvruette/gidd-easydel`

**Read this before implementing anything in §2.** The SwissAI-DLM author states the code
came from [`dvruette/gidd-easydel`](https://github.com/dvruette/gidd-easydel) (JAX/EasyDeL,
the codebase for *Scaling Behavior of Discrete Diffusion Language Models*,
arXiv:2512.10858). **That is true**, and it changes what this document is: most of §2 is
*recovering what the upstream already does*, not inventing it.

The lineage is explicit — SwissAI-DLM's function is named
`_build_adamw_jax_style_param_groups` and its docstring says "similar to EasyDeL JAX
train.py grouping". The four group names (`bulk_params`, `ln_params`, `bias_params`,
`emb_unemb_params`), the `aux_lr_factor` default of 0.02, and `lr / hidden_size` are
copied from upstream's `param_label_fn` and `optax.multi_transform` block.

What upstream has, and what survived the port:

| knob | `gidd-easydel` (`gidd_easydel/train.py`) | survived? |
|---|---|---|
| hidden init | `init_scale / hidden_size**0.5` (line 233) | **no** |
| embedding init | `emb_init_scale` = 0.02, width-independent (234) | **no** |
| readout init | `0.0 if zero_head_init else 0.02` (235) | partly — zero-init is there but is a no-op under tying (§1.7) |
| **readout forward multiplier** | `head_scaling = head_scale / hidden_size` (237) | **no** |
| attention | `head_dim` FIXED at 64; `num_attn_heads = hidden_size // head_dim` (132-141) | **no** — head_dim varies 64→128 across configs |
| bulk LR | `lr / hidden_size` (265) | yes, but the matcher misses emb/head |
| aux LR | `aux_lr = lr * aux_lr_factor` (144) | yes |
| Adam eps | `adam_eps / hidden_size / num_layers` — width AND depth (297) | **no** |
| bulk weight decay | `weight_decay * hidden_size` (299), cancelling `lr/hidden_size` so decoupled decay is width-invariant | **no** |
| residual | `resid_scale` threaded into the model (232) | **no** |

**And the port's central bug is a direct consequence of the copy.** Upstream's
`label_leaf` matches `"embed_tokens"` and `"lm_head"` — which *are* the names in EasyDeL's
JAX parameter tree, so it is correct there. The PyTorch port carried those literal strings
into a codebase whose modules are `vocab_embed` and `output_layer`. The match silently
fails and the embedding and readout land in `bulk_params` at `lr/hidden_size` (§1.1).

Two consequences for how to proceed:

1. **Port the rest of upstream rather than re-deriving it.** The constants
   (`init_scale` 0.4, `aux_init_scale` 0.02, `head_scale` 512, `resid_scale` 4.0,
   `aux_lr_factor` 0.02, `head_dim` 64) are tuned and published. Re-deriving them from
   scratch discards that work and risks landing somewhere worse.
2. **Upstream already answers two open questions in this document.** It picks §1.6
   option (a) — fix `head_dim`, grow the head count. And `head_scaling = head_scale /
   hidden_size` is exactly the readout forward multiplier of §1.7/§2.5, so the tying fix is
   a port, not a design.

What upstream does **not** give you, and this document still has to: **Peri-LN** (§2.2),
and an explicit **`1/L` residual-branch depth law** (§1.3) — upstream's depth handling is
limited to the `eps / num_layers` term and a `resid_scale` constant that does not vary
with `L`.

## 1. What is wrong, precisely

### 1.1 The param-group matcher misses this repo's modules — defect 1

`_build_adamw_jax_style_param_groups` buckets by substring:

```python
if "norm" in lname:                                          -> ln_params      (lr = base*0.02)
elif "embed_tokens" in lname or "lm_head" in lname:          -> emb_unemb      (own lr)
elif "bias" in lname:                                        -> bias_params    (lr = base*0.02)
elif param.ndim > 1:                                         -> bulk_params    (lr = base/hidden)
```

But the native DiT names its modules `self.vocab_embed` (`models/dit.py:86`) and
`self.output_layer` (`models/dit.py:110`). Neither contains `embed_tokens` or `lm_head`.
**So the embedding and the readout fall through to `bulk_params` and receive
`base_lr / hidden_size`.** Under muP the embedding is an *input* weight and its learning
rate must be **width-independent**; dividing it by width makes the embedding learn
`1/width` slower at every rung, which is precisely the drift muP exists to remove. The
`emb_unemb_params` group is dead code for every model except the HF ones.

Two further problems in the same function:

* `aux_lr = base_lr * aux_lr_factor` (default 0.02) is a width-independent constant for
  norms and biases. That happens to be right for muP (gains and biases are Θ(1) tensors),
  but it is right by accident, not by derivation, and the 0.02 is a tuned constant that
  will silently stop being right if the base width changes.
* `bulk_lr = base_lr / hidden_size` divides by the **model width**, not by each tensor's
  **fan-in**. These coincide for square hidden weights and are wrong for everything else —
  notably `adaLN_modulation`, whose fan-in is `cond_dim` (see §1.5).

### 1.2 Initialisation is width-dependent in the wrong direction — defect 2

`EmbeddingLayer` (`models/common.py:296-300`):

```python
self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
```

`kaiming_uniform_` derives `fan_in` from `tensor.shape[1]`, which here is `dim`, the model
width. So the embedding's init std scales as `1/√dim` — it **shrinks** as the model widens.
muP requires an input weight's init std to be **width-independent** (`Θ(1)`), because its
fan-in (the vocabulary one-hot) does not grow. This is backwards, and it is replicated in
`models/dit_candi.py:31` and `models/flexmdm_transformer.py:240`.

Every other weight uses PyTorch's `nn.Linear` default, which is also
`kaiming_uniform_(a=√5)` → std `∝ 1/√fan_in`. For *hidden* weights that is the correct
width law, so hidden init is accidentally fine; only the scale constant is unprincipled.

Two things are already right and must be preserved: `DDiTFinalLayer.linear` is zero-init
(`common.py:330`) and every `adaLN_modulation` is zero-init. Zero-init of the readout is
the muP-correct choice and it has a useful side effect — it makes the readout's `init`
coordinate-check channel structurally zero rather than a failure.

### 1.3 Nothing controls depth — defect 3

The residual writes are (`models/block_dit.py:209-218`):

```python
x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
x = bias_dropout_scale_fn(self.mlp(...),    None, gate_mlp, x,      self.dropout)
```

i.e. `x ← x + gate * branch(x)` with no `1/L` factor. With `L` blocks each writing an
`Θ(1)` update into the residual stream, the stream's magnitude grows like `L` (or `√L` if
branch outputs were independent, which after training they are not). The practical
consequence is that a learning rate tuned at `L = 12` is wrong at `L = 24`, independently
of width. `n_blocks` is already a config knob (12 in `configs/model/*.yaml`, 32 in
`gidd_hf_8b.yaml`), so this is a live axis that is currently unmanaged.

*Mitigating detail:* `gate_msa`/`gate_mlp` come from a **zero-initialised**
`adaLN_modulation`, so at step 0 every branch contributes exactly zero and the stream is
`Θ(1)` by construction. That hides the problem at initialisation and lets it appear during
training, which is the worst failure shape — it looks fine in a forward-pass check.

### 1.4 No Peri-LN — defect 4

The blocks are Pre-LN: `norm1`/`norm2` normalise the branch **input**
(`block_dit.py:195-205`), and nothing normalises the branch **output** before it is added
to the stream. Peri-LN adds exactly that, and it is what makes depth scaling behave: with a
norm on the branch output, each branch writes a magnitude that is controlled by
construction rather than by hoping the pre-norm plus the gate keep it bounded.

### 1.5 `cond_dim` is fixed while `hidden_size` scales — a classification trap

`configs/model/small.yaml` and `block_dit.yaml` both set `hidden_size: 768`, `cond_dim: 128`.
`adaLN_modulation` is `nn.Linear(cond_dim, 6*dim)`: **fan-in pinned at 128, fan-out ∝ width.**

By the published definition that makes `adaLN_modulation` an **input-like** weight
(width-independent init, width-independent LR), *not* a hidden one — even though the
reference DiT muP work (arXiv:2505.15270) marks adaLN hidden in its released code, because
in *their* DiT the conditioning width equals the model width. Here it does not.

This is not hypothetical. In ScalingBench we have the identical situation in
`mdlm_dit_adamw` (cond fixed at 128, fan-out 6N) versus `dit_adamw` (cond width = model
width). Classifying adaLN hidden in the fixed-cond case puts a systematic drift on the
`prop` channel of every adaLN tensor. **Decide this per config, from the actual fans, not
from the tensor's name.** If you later make `cond_dim` scale with `hidden_size`, the
classification flips to hidden — which is an argument for deriving the role from fans
rather than hardcoding it.

### 1.6 Attention scale versus `head_dim` — defect 5

`block_dit.py:167,316` use `scale = 1/√head_dim`. Across the shipped configs `head_dim` is
**not** constant: `768/12 = 64` in `small.yaml`, `4096/32 = 128` in `gidd_hf_8b.yaml`. So
width is being scaled partly by growing `head_dim`, and under muP the attention logit scale
must then be `1/head_dim`, not `1/√head_dim`.

You must pick one and state it:

* **(a) Fix `head_dim`, grow `n_heads`** — `n_heads = hidden_size / head_dim`. Then
  `1/√head_dim` is a constant and the existing code is already correct. *Recommended:* it is
  the smaller change and it is what most muP references assume.
* **(b) Grow `head_dim`** — then change the attention scale to `1/head_dim` and record it as
  a forward-scaling decision in the scaling config.

Silently mixing the two across configs, which is the current state, is the one option that
cannot be made correct.

### 1.7 Weight tying is compatible with muP — but only via a forward multiplier

`models/dit.py:119`: `self.output_layer.linear.weight = self.vocab_embed.embedding` when
`tie_word_embeddings` is set.

It is tempting to call this a conflict, because muP gives input and output weights
different init and LR laws and one tensor cannot have two. That reasoning is wrong, and
EleutherAI's `nanoGPT-mup` shows why: **move the output-side scaling out of the weight and
into the forward pass.** Their `model.py` ties unconditionally
(`self.transformer.wte.weight = self.lm_head.weight`, line 156) and is still muP-correct:

| what | how they do it | line |
|---|---|---|
| tied tensor init | plain `init_std`, **width-independent** — the INPUT law | `_init_weights`, and the muP block rescales only `c_attn/c_fc/c_proj` |
| tied tensor LR | `lr_scale = 1` — the INPUT law; it sits in `decay_params`, not `mup_decay_params` | 319 |
| the OUTPUT law | a forward multiplier: `x *= mup_output_alpha / mup_width_multiplier` **before** `lm_head(x)` | 219 |
| embedding side | a matching tunable `x *= mup_input_alpha` after the embedding | 208 |

Why that is correct rather than a fudge. Write `m = width / base_width`. With the readout
pre-scaled by `1/m`, the logits are `Θ(1)` in the aligned regime, which is what the output
law asks for. And the gradient reaching the tied tensor *through the readout path* is
scaled by the same `1/m`, so a width-independent LR produces an output-path update of
`Θ(1/m)` — again exactly the output law. The `1/m` has simply been moved from the weight's
init-and-LR into the forward pass, where one tensor can serve both roles.

#### What this repo actually does — tying is implemented, the scaling is not

`dit.py:117-119` and `block_dit.py:436-437`:

```python
self.output_layer = DDiTFinalLayer(...)         # __init__ does linear.weight.data.zero_()
if getattr(config.model, 'tie_word_embeddings', False):
    self.output_layer.linear.weight = self.vocab_embed.embedding
```

Three compounding problems, none of which raises anything:

1. **There is no forward multiplier at all.** Grepping the whole package for
   `output_alpha|input_alpha|output_mult|width_mult|mup_` returns nothing in the model
   code. So under tying the output law is supplied by *nothing* — not by the init, not by
   the LR, not by a multiplier.
2. **The zero-init is silently discarded.** `DDiTFinalLayer.__init__` zero-inits
   `self.linear.weight` at construction (`common.py:330`), and the tie then **rebinds**
   `linear.weight` to the embedding tensor. The zero-init therefore lands on a tensor that
   is thrown away one line later. The readout ends up carrying the embedding's
   `kaiming_uniform_` init, and the intended zero-init never happens — with nothing in the
   code or the logs to say so. (Note the ordering: the embedding is *not* destroyed, which
   is the failure people usually expect here. It is the safeguard that is destroyed.)
3. **The param-group bug (§1.1) hits the tied tensor hardest.** Under tying the shared
   tensor must follow the **input** law, `lr_scale = 1`. Because neither `vocab_embed` nor
   `output_layer` matches `embed_tokens`/`lm_head`, it lands in `bulk_params` and gets
   `base_lr / hidden_size` — the output law, applied to a tensor that needs the input one,
   and the exact opposite of what `nanoGPT-mup` does.

So: **tying here needs correction, not just completion.** The fix is §2.5, and the
zero-init must become conditional — not because it would destroy the embedding, but because
under tying it is a no-op that disguises a missing multiplier.

One more thing to carry over: **`output_alpha` and `input_alpha` are tunable base
multipliers**, not fixed constants. They are exactly the kind of knob that has to be found
by experiment rather than read off a table — in our own measurements base multipliers were
worth 0.12-0.22 of coordinate-check score on problems whose exponents were already right.

Also worth copying from the same file: their `c_proj` init carries an extra
`1/sqrt(2 * n_layer)` (line 168), i.e. a **depth-aware residual init**. That is a second,
independent way of controlling the residual stream with depth, and it composes with the
`1/L` branch scaling of §1.3 — do not apply both blindly without deciding which carries the
depth law.

### 1.8 Four copies of the same block — defect 6

`adaLN_modulation` appears in `block_dit.py`, `encoder_decoder.py`,
`flexmdm_transformer.py`, `modeling_gidd_hf.py`; `kaiming_uniform_` in `common.py`,
`dit_candi.py`, `flexmdm_transformer.py`. Any scaling change applied to one and not the
others produces models that *look* like they share a parametrisation and do not. We have
been bitten by exactly this: several ScalingBench problems turned out not to be the
architecture they claimed because a port copied a sibling and inherited its deviations.
Centralise (§2.1) rather than patch in four places.

---

## 2. The change set

### 2.1 New file: `src/discrete_diffusion/models/scaling.py`

Port BST's `code/moe_scaling/scaling.py` structure. One pluggable object owns every
scaling decision, so `model.py` and the optimiser never hardcode a rule.

```python
"""Width and depth scaling. One object owns init std, per-parameter LR, and the
forward-pass scales, so no rule is hardcoded in a model file."""

from dataclasses import dataclass

def compute_fans(param, role=None):
    """(fan_in, fan_out). nn.Linear weight is [out, in]; an embedding table is
    [vocab, dim] and its fan_in is the VOCAB side, which does not scale."""
    if param.dim() == 2:
        if role == "embedding":
            return param.shape[0], param.shape[1]
        return param.shape[1], param.shape[0]
    return 1, 1

def classify_parameter(name: str) -> str:
    """-> embedding | output | adaln | attn_qkv | attn_proj | mlp_up | mlp_down
          | timestep | norm | bias

    Matches THIS repo's module names. The existing matcher looks for
    'embed_tokens'/'lm_head', which never appear here -- that is defect 1.
    """
    if name.endswith(".bias"):                       return "bias"
    if "norm" in name or "peri_" in name:            return "norm"   # incl. the new Peri-LN gains
    if "vocab_embed" in name:                        return "embedding"
    if "output_layer.linear" in name:                return "output"
    if "adaLN_modulation" in name:                   return "adaln"
    if "sigma_map" in name or "cond_map" in name:    return "timestep"
    if "attn_qkv" in name:                           return "attn_qkv"
    if "attn_out" in name:                           return "attn_proj"
    if "mlp.0" in name:                              return "mlp_up"
    if "mlp.2" in name:                              return "mlp_down"
    raise ValueError(f"Cannot classify parameter: {name}")   # fail loudly, never default
```

`raise` rather than a default role is deliberate: a silently mis-bucketed tensor is a bug
you find three weeks later in a coordinate check.

```python
@dataclass
class MuPScaling:
    base_width: int = 768        # the width LR was tuned at
    base_depth: int = 12         # the depth LR was tuned at
    width: int = 768
    depth: int = 12
    adaln_is_input: bool = True  # True when cond_dim is FIXED (see 1.5)

    @property
    def mw(self):  return self.width / self.base_width      # width multiplier
    @property
    def md(self):  return self.depth / self.base_depth      # depth multiplier

    def init_std(self, name, param):
        role = classify_parameter(name)
        fan_in, _ = compute_fans(param, role)
        if role in ("norm", "bias"):            return None        # leave as ones/zeros
        if role == "output":                    return 0.0         # zero-init readout
        if role == "embedding":                 return self.base_init           # Theta(1)
        if role == "adaln" and self.adaln_is_input:  return 0.0    # already zero-init; keep
        return self.base_init / (fan_in ** 0.5)                    # hidden: 1/sqrt(fan_in)

    def lr_scale(self, name, param):
        """Multiplier on base_lr. AdamW: hidden ~ 1/mw, input/gain ~ 1, output ~ 1/mw."""
        role = classify_parameter(name)
        if role in ("norm", "bias", "embedding", "timestep"):      return 1.0
        if role == "adaln" and self.adaln_is_input:                return 1.0
        return 1.0 / self.mw

    def eps_scale(self, name, param):
        """AdamW eps must fall with the *gradient* scale or it stops being negligible.
        Hidden: 1/mw (sqrt(v) shrinks with width). Embedding / norm / bias: 1. Readout: 1 too --
        its lr is 1/mw but its per-entry gradient (dL/dlogit * h_j) is Theta(1), so eps does NOT
        follow lr there. (This is what the bst-scaling reference and optimizers/adamw.py do.)"""
        role = classify_parameter(name)
        if role in ("norm", "bias", "embedding", "timestep", "output"):  return 1.0
        if role == "adaln" and self.adaln_is_input:                      return 1.0
        return 1.0 / self.mw

    @property
    def residual_scale(self):
        """CompleteP / depth-muP: each branch output is multiplied by base_depth/depth."""
        return 1.0 / self.md
```

Two notes on the laws above. `lr_scale` divides by `mw = width/base_width`, **not** by
`fan_in` directly — that keeps the base rung at exactly `base_lr` so an LR tuned at
`base_width` is reproduced unchanged there, which is the property that makes a transfer
sweep meaningful. And `eps_scale` tracks `lr_scale` for hidden weights only (the TP-IVb rule: a fixed
`eps` stops being negligible against `√v` once the gradient shrinks); the readout keeps `eps` fixed
because its gradient does not shrink. Under CompleteP the in-block weights additionally get
`eps *= base_depth / n_layers`.

### 2.2 Peri-LN in the block — `models/block_dit.py`

Follow BST `model.py:139-173` exactly, including the no-op guarantee.

In `DDiTBlock.__init__`:

```python
# Peri-LN: normalise each residual BRANCH OUTPUT, so the write into the stream is
# width- and depth-controlled by construction. peri_ln=False must be an EXACT no-op
# (nn.Identity), verifiable on parameter count and on a forward pass.
self.peri_attn = LayerNorm(dim) if config.peri_ln else nn.Identity()
self.peri_mlp  = LayerNorm(dim) if config.peri_ln else nn.Identity()
self.residual_scale = scaling.residual_scale if config.depth_scaling else 1.0
```

In `forward`, the two residual writes become:

```python
x = bias_dropout_scale_fn(
        self.peri_attn(self.attn_out(x)), None,
        gate_msa * self.residual_scale, x_skip, self.dropout)
x = bias_dropout_scale_fn(
        self.peri_mlp(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp))), None,
        gate_mlp * self.residual_scale, x, self.dropout)
```

Folding `residual_scale` into the existing `gate` argument avoids touching
`bias_dropout_add_scale`'s signature and keeps the fused train/inference paths identical.

Optionally add Peri-LN part 1 (`peri_ln_embed`): `x = Norm(vocab_embed(x))` right after the
embedding in `models/dit.py:137`. BST exposes this as a separate flag because it is a
separate claim; keep it separately switchable.

### 2.3 Apply the init — `models/dit.py`

Add one pass after the modules are built (`models/dit.py`, end of `__init__`):

```python
if config.model.get("scaling", None) is not None:
    for name, p in self.named_parameters():
        std = self.scaling.init_std(name, p)
        if std is None:      continue
        if std == 0.0:       p.data.zero_()
        else:                p.data.normal_(0.0, std)
```

This replaces the `kaiming_uniform_` in `EmbeddingLayer.__init__` for the scaled path.
Leave the unscaled path byte-identical so `scaling=None` reproduces today's model exactly.

### 2.4 Replace the param-group builder — `algorithms/base.py`

Replace the substring matcher in `_build_adamw_jax_style_param_groups` with one group per
parameter, driven by the scaling object:

```python
groups = []
for name, p in self._named_trainable_parameters():
    groups.append({
        "params": [p],
        "name": name,
        "lr": base_lr * self.scaling.lr_scale(name, p),
        "eps": base_eps * self.scaling.eps_scale(name, p),
        "weight_decay": wd_for(classify_parameter(name)),
    })
```

One group per tensor is the honest implementation and costs nothing measurable in AdamW.
If you prefer fewer groups, bucket by `(role, fan_in)` — never by name substring.

**Weight decay:** use **fully decoupled** decay, i.e. pass `λ/α` as PyTorch's
`weight_decay` so the per-step shrink is `λ·W`, independent of the learning rate. Otherwise
the per-tensor LR scaling silently rescales the decay too, and your decay becomes a
function of width. This is what we do across all nine ScalingBench problems.

### 2.5 Weight tying, as a supported option

Follow `EleutherAI/nanoGPT-mup`. Two forward multipliers and one classification rule.

**Config:**

```yaml
scaling:
  tie_word_embeddings: false   # when true, the readout MUST NOT be zero-init
  input_alpha: 1.0             # tunable; multiplies the embedding output
  output_alpha: 1.0            # tunable; multiplies the readout input
```

**`scaling.py`** — the tied tensor takes the INPUT law on both knobs:

```python
def init_std(self, name, param):
    role = classify_parameter(name)
    ...
    if role == "output":
        # Untied: zero-init is correct and makes the init channel structurally zero.
        # Tied:   the tensor IS the embedding. Today's code zero-inits linear.weight and
        #         then rebinds it, so the zero-init is a silent no-op; the output law must
        #         come from output_alpha/mw in the forward instead (nanoGPT-mup).
        return self.base_init if self.tie_word_embeddings else 0.0
    ...

def lr_scale(self, name, param):
    role = classify_parameter(name)
    if role == "output" and self.tie_word_embeddings:
        return 1.0          # input law; the 1/mw lives in the forward multiplier
    ...

@property
def output_fwd_scale(self):
    """Pre-multiplier on the readout input. 1/mw is what makes a tied tensor
    satisfy the output law without touching its init or its LR."""
    return self.output_alpha / self.mw

@property
def input_fwd_scale(self):
    return self.input_alpha
```

**`models/dit.py`** — two lines in `forward`:

```python
x = self.vocab_embed(x)
if self.scaling is not None:
    x = x * self.scaling.input_fwd_scale          # after the embedding
...
if self.scaling is not None:
    x = x * self.scaling.output_fwd_scale         # BEFORE output_layer, not after
x = self.output_layer(x, c=t_cond)
```

Scale `x` rather than the logits, as `nanoGPT-mup` notes at its line 219: it keeps the
multiplier visible to a coordinate check on the readout's input, which is where you want to
see it.

**`models/common.py`** — make the conflict impossible rather than documented:

```python
class DDiTFinalLayer(nn.Module):
    def __init__(self, ..., tie_word_embeddings: bool = False):
        ...
        if not tie_word_embeddings:
            self.linear.weight.data.zero_()   # untied: muP zero-init readout
        # tied: left at the embedding's init; output law comes from output_fwd_scale
```

and assert loudly at construction if `tie_word_embeddings and readout_zero_init` are both
requested — today that combination is accepted and silently resolves to neither.

**Tests to add** alongside §3: with tying on, (a) the embedding and readout are the same
storage, (b) the tied tensor's init std is identical at two widths, (c) its LR is identical
at two widths, (d) `output_fwd_scale` halves when width doubles.

### 2.6 SCION

`optimizers/scion.py` is a norm-constrained optimiser (Muon-family). **Its width law is not
AdamW's.** For an orthogonalised update the Newton–Schulz output has spectral norm ≈1
regardless of shape, so to keep `ΔW·x` at `Θ(1)` per coordinate the *update radius* must
carry `√(fan_out/fan_in)` — and since both fans scale with width, that ratio is
**width-independent**. So the per-tensor width *exponent* for a SCION-governed hidden
weight is **0**, with `√(fan_out/fan_in)` as a constant per-tensor multiplier.

Do not reuse `lr_scale` for SCION groups. Give `MuPScaling` a second method
(`radius_scale`) and select on which optimiser owns the tensor. The AdamW-governed tensors
in a mixed setup (embedding, readout, gains, biases) keep the AdamW laws above.

### 2.7 Config surface

Add to `configs/model/*.yaml`:

```yaml
scaling:
  enabled: false        # false => byte-identical to today
  base_width: 768
  base_depth: 12
  adaln_is_input: true  # true while cond_dim is FIXED; see 1.5
peri_ln: false
peri_ln_embed: false
depth_scaling: false
```

Every flag defaults to the current behaviour. `n_heads` should be **derived** —
`n_heads = hidden_size // head_dim` with `head_dim` fixed — if you take option (a) of §1.6.

---

## 2b. The coordinate checks that decide whether any of this worked

Specified separately, in `SwissAI-DLM_coordinate_checks.md`: width, depth and
width-depth co-scaling, with the exact quantity, norm, time window, ladder and pass
criterion for each, plus the no-op tests. **Width, depth and co-scale are three different
measurements, not one measurement on three ladders** — applying the width recipe to a depth
ladder gives the wrong quantity and a band four times too wide to conclude anything. Read
that document before running anything on a GPU.

## 3. How to verify, without a training run

All of these are CPU-only and cheap. Do them in this order.

1. **No-op test.** With every flag off, assert the state dict keys, parameter count and a
   fixed-seed forward pass are **bit-identical** to the pre-change model. BST treats this as
   the gate for Peri-LN specifically (`peri_ln=False` is `nn.Identity`), and it is what lets
   you land the change without re-validating existing results.
2. **Classification coverage.** Iterate `named_parameters()` on every model in
   `configs/model/` and assert `classify_parameter` raises for none of them. This is the
   test that would have caught defect 1 on day one.
3. **Fan-derivation test.** For two widths, assert `compute_fans` returns what you expect
   per role, especially for `vocab_embed` (fan_in = vocab, not width) and
   `adaLN_modulation` (fan_in = cond_dim).
4. **Init-law test.** Build at width 256 and 512 and assert: embedding std is **unchanged**,
   hidden std falls by `1/√2`, readout is exactly zero.
5. **LR-law test.** Assert the embedding group's LR is identical at both widths and the
   hidden groups' LR halves.
6. **Depth no-op and law.** At `L = base_depth`, `residual_scale == 1.0` exactly. At
   `L = 2·base_depth`, it is `0.5`.
7. **Coordinate check.** The real verification: 4 widths × ≥4 seeds × ~100 steps, measuring
   per-tensor activation / propagation / effective-update RMS, and fitting the width
   exponent of each. All exponents ≈ 0 is the goal. Then repeat on the depth axis.

On the coordinate check, three lessons from our runs, each of which cost us a wrong
conclusion:

* Report **all three** channels, not just activations. A systematic drift can sit entirely
  in propagation and be invisible in the activation channel.
* Check the **seed scatter** before believing a number. At 4 seeds we have problems whose
  worst channel is within 2 standard errors of zero — differences there are not real.
* Watch the **base rung**. If a "fixed" dimension (here `cond_dim = 128`, or `head_dim`) is
  comparable to the width at your smallest rung, that rung is not in the asymptotic regime
  and will dominate the fit. With `cond_dim = 128` you should not put a rung below ~512.

---

## 4. Suggested order of work

1. `scaling.py` + the classification and fan tests (§2.1, §3.2-3.3). No behaviour change.
2. Param groups (§2.4). **This alone fixes defect 1** and is the single biggest win.
3. Init pass (§2.3). Fixes defect 2.
4. Peri-LN (§2.2) with the no-op test. Fixes defect 4.
5. Depth scaling (§2.2, `residual_scale`). Fixes defect 3.
6. Settle `head_dim` (§1.6). If tying is wanted, add the two forward multipliers (§1.7, §2.5).
7. Coordinate check on width, then on depth.
8. SCION radius law (§2.6) — last, because it is a different derivation and should not be
   mixed into the AdamW validation.

Steps 1–2 are worth doing even if nothing else happens: right now the embedding and the
readout are being trained at `1/width` of their intended learning rate in every native DiT
config in the repo.
