"""muP / CompleteP optimizer-hyperparameter scaling for the GIDD-HF model.

Port of ``moe_scaling/scaling.py`` (``MupDense``) and the ``depth_scaling`` branch of
``model.configure_optimizers`` from bst-scaling.  Pure Python (no torch import) so the
rules can be unit-tested anywhere.

Each parameter gets a *role*.  A ``Parametrization`` maps a role to

* ``lr_scale``  -- multiplier on the global learning rate,
* ``eps_scale`` -- multiplier on Adam's epsilon.

Rules (``n`` = hidden_size, ``n0`` = ``base_width``, ``L`` = n_blocks, ``L0`` = ``base_depth``).
"input-like" roles are embedding, norm gains and biases::

    role                     | init std (already so) | lr          | eps
    -------------------------+-----------------------+-------------+-----------------
    embedding, norm, bias    | n^0                   | 1           | 1
    hidden matrices          | n^-1/2                | n0/n        | n0/n
    output (lm_head)         | 0                     | n0/n        | 1
    in-block, CompleteP only |                       | (unchanged) | extra factor L0/L

``eps`` of hidden matrices follows ``1/n`` because ``sqrt(v)`` shrinks with width, and with
CompleteP the in-block eps shrinks with depth because the branch gradient does.

The model is the unmodified GIDD architecture (``model.parametrization`` set => its legacy
scale knobs are neutral and :mod:`.parametrized_init` initialises it: hidden std
``1 / sqrt(fan_in)``, constant embedding std, zero head).  Its residual multiplier
``resid_scale / n_blocks`` is already a 1/L form already (``standard`` / ``mup`` keep it); ``completep`` uses the reference
``resid_mult * base_depth / n_blocks`` instead and, like the reference, also scales the in-block
epsilon.  The Adam LR is depth-independent for ``alpha = 1``.  ``validate_model_config`` rejects settings that would
break the rules (tied embeddings, non-zero head init, extra head / activation scales).
Width scaling needs ``intermediate_size`` and ``n_heads`` to grow with
``hidden_size`` (fixed head_dim).
"""

from __future__ import annotations

from dataclasses import dataclass

ROLES = (
    'embedding', 'output', 'attn_qkv', 'attn_proj', 'mlp_up', 'mlp_down', 'norm', 'bias', 'other')

# Roles inside a transformer block that get the CompleteP depth correction on eps.
IN_BLOCK_ROLES = frozenset({'attn_qkv', 'attn_proj', 'mlp_up', 'mlp_down'})
_INPUT_LIKE = frozenset({'embedding', 'norm', 'bias'})


def classify_parameter(name: str) -> str:
  """Map a GIDD-HF parameter name to its scaling role.

  Works on the names of the wrapped backbone (``model.model.layers.3.self_attn.q_proj.weight``)
  and tolerates DDP / FSDP / checkpoint prefixes since it only matches substrings.
  Anything unrecognised (e.g. noise-schedule parameters) is ``other`` and left unscaled.
  """
  lname = name.lower()
  if lname.endswith('bias'):  # up_proj.bias, down_proj.bias, k_bias, v_bias
    return 'bias'
  if 'norm' in lname:  # attn_layernorm, mlp_layernorm, q_norm, k_norm, final norm
    return 'norm'
  if 'embed_tokens' in lname:
    return 'embedding'
  if 'lm_head' in lname:
    return 'output'
  if 'self_attn.' in lname and any(
      f'.{p}_proj' in lname for p in ('q', 'k', 'v')):
    return 'attn_qkv'
  if 'self_attn.o_proj' in lname:
    return 'attn_proj'
  if 'mlp.up_proj' in lname:
    return 'mlp_up'
  if 'mlp.down_proj' in lname:
    return 'mlp_down'
  return 'other'


@dataclass(frozen=True)
class Parametrization:
  """Standard parametrization: every role has lr/eps scale 1."""
  hidden_size: int
  n_blocks: int
  base_width: int = 256
  base_depth: int = 12

  name = 'standard'

  def lr_scale(self, role: str) -> float:
    return 1.0

  def eps_scale(self, role: str) -> float:
    return 1.0

  def residual_multiplier(self, resid_scale: float, resid_mult: float = 1.0) -> float:
    """Multiplier on every residual branch output: the original GIDD ``resid_scale / L``."""
    return resid_scale / self.n_blocks


@dataclass(frozen=True)
class MuP(Parametrization):
  """muP for AdamW (multiplier-free form, reference ``MupDense``).

  ``lr_scale == eps_scale == 1`` at ``hidden_size == base_width``, so the global LR is
  the learning rate of the width-``base_width`` model.
  """
  name = 'mup'

  def _width(self) -> float:
    return self.base_width / self.hidden_size

  def lr_scale(self, role: str) -> float:
    if role in _INPUT_LIKE or role == 'other':
      return 1.0
    return self._width()

  def eps_scale(self, role: str) -> float:
    if role in _INPUT_LIKE or role in ('output', 'other'):
      return 1.0
    return self._width()


@dataclass(frozen=True)
class CompleteP(MuP):
  """muP in width plus the CompleteP (alpha = 1) depth correction.

  The residual branches are already multiplied by ``resid_scale / n_blocks`` in the model;
  the matching optimizer-side rule is ``eps *= base_depth / n_blocks`` for in-block weights.
  The Adam learning rate is depth-independent.
  """
  name = 'completep'

  def residual_multiplier(self, resid_scale: float, resid_mult: float = 1.0) -> float:
    # Reference ``depth_scale = base_depth / n_layer`` (alpha = 1): exactly ``resid_mult`` at
    # the base depth, and the branch contribution of the whole network stays O(1) in depth.
    return resid_mult * self.base_depth / self.n_blocks

  def eps_scale(self, role: str) -> float:
    scale = super().eps_scale(role)
    if role in IN_BLOCK_ROLES:
      scale *= self.base_depth / self.n_blocks
    return scale


PARAMETRIZATIONS = {
    'standard': Parametrization,
    'mup': MuP,
    'completep': CompleteP,
}


def get_parametrization(
    name: str, *, hidden_size: int, n_blocks: int,
    base_width: int = 256, base_depth: int = 12) -> Parametrization:
  key = str(name).lower()
  if key not in PARAMETRIZATIONS:
    raise ValueError(
        f'Unknown parametrization {name!r}. Available: {", ".join(PARAMETRIZATIONS)}')
  if hidden_size <= 0 or n_blocks <= 0 or base_width <= 0 or base_depth <= 0:
    raise ValueError('hidden_size, n_blocks, base_width and base_depth must be positive.')
  return PARAMETRIZATIONS[key](
      hidden_size=int(hidden_size), n_blocks=int(n_blocks),
      base_width=int(base_width), base_depth=int(base_depth))


def validate_model_config(parametrization: Parametrization, model_cfg) -> None:
  """Reject model settings that break the parametrization (muP / CompleteP only)."""
  if parametrization.name == 'standard':
    return
  tag = f'model.parametrization={parametrization.name}'
  if bool(getattr(model_cfg, 'tie_word_embeddings', False)):
    raise ValueError(
        f'{tag} needs untied embeddings: with tying the matrix is input- and output-like at '
        'once and no single LR scale is correct.')
  if not bool(getattr(model_cfg, 'zero_head_init', True)):
    raise ValueError(
        f'{tag} needs model.zero_head_init=true (the muP output layer is zero-initialised; '
        'a constant-std head breaks width transfer).')
  if float(getattr(model_cfg, 'activation_scale', 1.0)) != 1.0:
    raise ValueError(f'{tag} requires model.activation_scale=1.0.')
  if bool(getattr(model_cfg, 'head_output_scaling_enabled', False)):
    raise ValueError(
        f'{tag} is the multiplier-free form; disable model.head_output_scaling_enabled.')
