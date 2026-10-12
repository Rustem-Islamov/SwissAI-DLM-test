"""Initialisation of the GIDD-HF model from a muP / CompleteP parametrization.

Used when ``model.parametrization`` is set.  The model code itself is the unmodified GIDD
architecture; all of its legacy scale knobs (``init_scale``, ``weight_scaling``,
``head_scaling``, ``activation_scale``) are neutral in this mode, and this function owns the
init, following the bst-scaling ``MupDense`` / ``StandardScaling`` rules:

* embedding            N(0, aux_init_scale^2)            (width-independent)
* hidden matrices      N(0, (init_std_mult / sqrt(fan_in))^2)
* lm_head              zeros (or N(0, aux_init_scale^2) if ``zero_head_init`` is false; muP
                       requires zeros)
* biases zeros; RMSNorm gain zeros if ``norm_zero_centered`` (``1 + weight``) else ones
"""

from __future__ import annotations

import torch

from .scaling import IN_BLOCK_ROLES, classify_parameter, get_parametrization


@torch.no_grad()
def init_parametrized(module: torch.nn.Module, model_cfg) -> None:
  init_std_mult = float(getattr(model_cfg, 'init_std_mult', 1.0))
  emb_std = float(model_cfg.aux_init_scale)
  head_std = 0.0 if bool(model_cfg.zero_head_init) else emb_std
  zero_centered_norm = bool(getattr(model_cfg, 'norm_zero_centered', True))
  seen = set()
  for name, param in module.named_parameters():
    if id(param) in seen:
      continue
    seen.add(id(param))
    role = classify_parameter(name)
    if role == 'embedding':
      torch.nn.init.normal_(param, mean=0.0, std=emb_std)
    elif role == 'output':
      if head_std > 0.0:
        torch.nn.init.normal_(param, mean=0.0, std=head_std)
      else:
        torch.nn.init.zeros_(param)
    elif role in IN_BLOCK_ROLES and param.ndim == 2:
      fan_in = param.shape[1]
      torch.nn.init.normal_(param, mean=0.0, std=init_std_mult * fan_in ** -0.5)
    elif role == 'norm' and not zero_centered_norm:
      torch.nn.init.ones_(param)
    else:  # bias, zero-centered norm gain (the norm computes ``1 + weight``), anything else
      torch.nn.init.zeros_(param)


def set_residual_scale(module: torch.nn.Module, model_cfg) -> float:
  """Set the residual-branch multiplier of every layer from the parametrization (1/L form).

  GIDD stores it as ``resid_scale`` on each ``GiddLayer`` (and on the model / pipeline stage
  that built them); all of those are overwritten with the same value.
  """
  parametrization = get_parametrization(
      model_cfg.parametrization, hidden_size=model_cfg.hidden_size,
      n_blocks=model_cfg.n_blocks, base_width=getattr(model_cfg, 'base_width', 256),
      base_depth=getattr(model_cfg, 'base_depth', 12))
  value = parametrization.residual_multiplier(
      float(model_cfg.resid_scale), float(getattr(model_cfg, 'resid_mult', 1.0)))
  for m in module.modules():
    if hasattr(m, 'resid_scale'):
      m.resid_scale = value
  if hasattr(module, 'resid_scale'):
    module.resid_scale = value
  return value
