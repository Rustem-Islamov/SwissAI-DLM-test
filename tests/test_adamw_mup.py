"""Tests for muP / CompleteP scaling rules and the decoupled-weight-decay AdamW.

The scaling tests are torch-free; the optimizer tests are skipped without torch.
"""

import importlib.util
import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'discrete_diffusion'
_spec = importlib.util.spec_from_file_location('gidd_scaling', _SRC / 'models' / 'scaling.py')
scaling = importlib.util.module_from_spec(_spec)
sys.modules['gidd_scaling'] = scaling
_spec.loader.exec_module(scaling)

NAMES = {
    'model.model.embed_tokens.weight': 'embedding',
    'model.lm_head.weight': 'output',
    'model.model.layers.3.self_attn.q_proj.weight': 'attn_qkv',
    'model.model.layers.3.self_attn.k_proj.weight': 'attn_qkv',
    'model.model.layers.3.self_attn.v_proj.weight': 'attn_qkv',
    'model.model.layers.3.self_attn.o_proj.weight': 'attn_proj',
    'model.model.layers.3.self_attn.k_bias': 'bias',
    'model.model.layers.3.self_attn.v_bias': 'bias',
    'model.model.layers.3.self_attn.q_norm.weight': 'norm',
    'model.model.layers.3.attn_layernorm.weight': 'norm',
    'model.model.layers.3.mlp_layernorm.weight': 'norm',
    'model.model.layers.3.mlp.up_proj.weight': 'mlp_up',
    'model.model.layers.3.mlp.up_proj.bias': 'bias',
    'model.model.layers.3.mlp.down_proj.weight': 'mlp_down',
    'model.model.layers.3.mlp.down_proj.bias': 'bias',
    'model.model.norm.weight': 'norm',
    '_fsdp_wrapped_module.backbone.model.model.layers.0.mlp.up_proj.weight': 'mlp_up',
    'noise.sigma': 'other',
}


@pytest.mark.parametrize('name,role', NAMES.items())
def test_classify(name, role):
  assert scaling.classify_parameter(name) == role


def _p(name, **kw):
  return scaling.get_parametrization(name, hidden_size=1024, n_blocks=24, **kw)


def test_standard_is_identity():
  p = _p('standard')
  assert all(p.lr_scale(r) == 1 == p.eps_scale(r) for r in scaling.ROLES)


def test_mup_rules():
  p = _p('mup')  # width factor 256 / 1024
  for role in ('attn_qkv', 'attn_proj', 'mlp_up', 'mlp_down', 'output'):
    assert p.lr_scale(role) == 0.25
  for role in ('attn_qkv', 'attn_proj', 'mlp_up', 'mlp_down'):
    assert p.eps_scale(role) == 0.25
  assert p.eps_scale('output') == 1.0
  for role in ('embedding', 'norm', 'bias'):
    assert p.lr_scale(role) == 1.0 and p.eps_scale(role) == 1.0


def test_mup_is_identity_at_base_width():
  p = scaling.get_parametrization('mup', hidden_size=256, n_blocks=12)
  assert all(p.lr_scale(r) == 1 == p.eps_scale(r) for r in scaling.ROLES)


def test_completep_depth_eps_only_in_block():
  m, c = _p('mup'), _p('completep')  # depth factor 12 / 24
  for role in scaling.ROLES:
    assert c.lr_scale(role) == m.lr_scale(role)  # Adam LR is depth-independent
    expected = m.eps_scale(role) * (0.5 if role in scaling.IN_BLOCK_ROLES else 1.0)
    assert c.eps_scale(role) == expected


def test_completep_equals_mup_at_base_depth():
  m = scaling.get_parametrization('mup', hidden_size=512, n_blocks=12)
  c = scaling.get_parametrization('completep', hidden_size=512, n_blocks=12)
  assert all(m.eps_scale(r) == c.eps_scale(r) for r in scaling.ROLES)


def test_residual_multiplier_is_1_over_L():
  for name in ('standard', 'mup'):
    assert _p(name).residual_multiplier(4.0) == 4.0 / 24
  c = _p('completep')
  assert c.residual_multiplier(4.0, 1.0) == 12 / 24
  assert c.residual_multiplier(4.0, 2.0) == 2 * 12 / 24
  deep = scaling.get_parametrization('completep', hidden_size=1024, n_blocks=48)
  assert deep.residual_multiplier(4.0) == c.residual_multiplier(4.0) / 2


def test_unknown_parametrization():
  with pytest.raises(ValueError):
    _p('nope')


class _Cfg:
  def __init__(self, **kw):
    self.__dict__.update(kw)


def test_validate_model_config():
  ok = _Cfg(tie_word_embeddings=False, zero_head_init=True, head_output_scaling_enabled=False)
  scaling.validate_model_config(_p('mup'), ok)
  for bad in (dict(tie_word_embeddings=True), dict(zero_head_init=False),
              dict(head_output_scaling_enabled=True)):
    with pytest.raises(ValueError):
      scaling.validate_model_config(_p('completep'), _Cfg(**{**ok.__dict__, **bad}))
    scaling.validate_model_config(_p('standard'), _Cfg(**{**ok.__dict__, **bad}))


torch = pytest.importorskip('torch')


def _load_optim():
  # Import through the package so the relative import of ..models.scaling resolves.
  from discrete_diffusion.optimizers.adamw import (
      DecoupledAdamW, build_adamw_param_groups)
  return DecoupledAdamW, build_adamw_param_groups


def test_decoupled_decay_is_independent_of_lr_and_scale():
  DecoupledAdamW, _ = _load_optim()
  wd = 1e-2
  results = []
  for lr in (1e-3, 1e-1):
    w = torch.nn.Parameter(torch.ones(4))
    opt = DecoupledAdamW([{'params': [w], 'lr': lr, 'eps': 1e-8}], lr=lr,
                         lambda_decay=wd)
    w.grad = torch.zeros_like(w)  # zero grad: Adam step is exactly 0, only decay remains
    opt.step()
    results.append(w.detach().clone())
  for r in results:
    assert torch.allclose(r, torch.full((4,), 1 - wd))


def test_decay_follows_schedule():
  DecoupledAdamW, _ = _load_optim()
  w = torch.nn.Parameter(torch.ones(2))
  opt = DecoupledAdamW([w], lr=1e-2, lambda_decay=0.1)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
  w.grad = torch.zeros_like(w)
  opt.step()
  assert torch.allclose(w.detach(), torch.full((2,), 1 - 0.5 * 0.1))


def test_matches_torch_adamw_when_wd_zero_and_decouples_otherwise():
  DecoupledAdamW, _ = _load_optim()
  torch.manual_seed(0)
  a = torch.nn.Parameter(torch.randn(8))
  b = torch.nn.Parameter(a.detach().clone())
  ref = torch.optim.AdamW([a], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
  new = DecoupledAdamW([b], lr=1e-2, betas=(0.9, 0.95), lambda_decay=0.0)
  for _ in range(5):
    g = torch.randn(8)
    a.grad, b.grad = g.clone(), g.clone()
    ref.step(); new.step()
  assert torch.allclose(a, b)


def test_param_groups_carry_mup_scales():
  _, build = _load_optim()
  m = torch.nn.ModuleDict({
      'embed_tokens': torch.nn.Embedding(4, 8),
      'mlp': torch.nn.ModuleDict({'up_proj': torch.nn.Linear(8, 8)}),
      'lm_head': torch.nn.Linear(8, 4, bias=False)})
  p = scaling.get_parametrization('completep', hidden_size=1024, n_blocks=24)
  groups = {g['name']: g for g in build(m.named_parameters(), p, lr=1e-3, eps=1e-8,
                                        weight_decay=1e-4)}
  assert groups['embedding']['lr'] == 1e-3
  assert groups['output']['lr'] == pytest.approx(1e-3 * 0.25)
  assert groups['mlp_up']['eps'] == pytest.approx(1e-8 * 0.25 * 0.5)
  assert groups['bias']['lr'] == 1e-3
  assert all(g['lambda_decay'] == 1e-4 for g in groups.values())


def test_init_parametrized():
  from discrete_diffusion.models.parametrized_init import init_parametrized
  m = torch.nn.ModuleDict({
      'embed_tokens': torch.nn.Embedding(512, 1024),
      'up_proj': torch.nn.Linear(1024, 4096),
      'layers': torch.nn.ModuleDict({'0': torch.nn.ModuleDict({
          'self_attn': torch.nn.ModuleDict({'q_proj': torch.nn.Linear(1024, 1024)}),
          'attn_layernorm': torch.nn.LayerNorm(1024)})}),
      'lm_head': torch.nn.Linear(1024, 512, bias=False)})
  cfg = _Cfg(aux_init_scale=0.02, zero_head_init=True, init_std_mult=2.0)
  init_parametrized(m, cfg)
  assert m['lm_head'].weight.abs().max() == 0
  assert m['embed_tokens'].weight.std().item() == pytest.approx(0.02, rel=0.05)
  q = m['layers']['0']['self_attn']['q_proj']
  assert q.weight.std().item() == pytest.approx(2.0 / 1024 ** 0.5, rel=0.05)
  assert q.bias.abs().max() == 0


def _tiny_layer(**kw):
  pytest.importorskip('transformers')
  from discrete_diffusion.models.configuration_gidd import GiddConfig
  from discrete_diffusion.models.modeling_gidd_hf import GiddLayer
  cfg = GiddConfig(vocab_size=16, hidden_size=32, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=2, max_position_embeddings=8, attn_performer='eager',
                   attention_bias=False, mlp_bias=False, torch_dtype=torch.float32, **kw)
  return cfg, GiddLayer(cfg, 0, resid_scale=1.0)


def test_peri_ln_layer_structure_and_reference_options():
  cfg, layer = _tiny_layer(
      peri_ln=True, attn_soft_cap=0.0, mlp_activation='gelu', norm_zero_centered=False,
      qk_norm_per_head=True, qk_norm_affine=False, rms_norm_eps=None)
  names = {n for n, _ in layer.named_parameters()}
  assert {'attn_layernorm.weight', 'mlp_layernorm.weight', 'peri_attn_layernorm.weight',
          'peri_mlp_layernorm.weight'} <= names
  assert not any('q_norm' in n or 'k_norm' in n for n in names)  # gain-free per-head QK-norm
  assert torch.all(layer.peri_attn_layernorm.weight == 1)
  x = torch.randn(2, 8, 32)
  pos = torch.arange(8)[None].expand(2, 8)
  from discrete_diffusion.models.modeling_gidd_hf import compute_basic_frequencies
  freqs = compute_basic_frequencies(base=cfg.rope_theta, rotary_dim=16, max_position_embeddings=8)
  out = layer(x, attention_mask=None, position_ids=pos, frequencies=freqs).hidden_states
  assert out.shape == x.shape and torch.isfinite(out).all()
  # Peri-LN: each branch adds an RMS-normalised update, so the residual change has RMS ~ 1 per branch.
  assert (out - x).pow(2).mean().sqrt() < 3.0


def test_peri_ln_off_is_original_layer():
  _, layer = _tiny_layer()
  assert isinstance(layer.peri_attn_layernorm, torch.nn.Identity)
  assert layer.attn_layernorm.zero_centered and layer.attn_layernorm.weight.abs().max() == 0


def test_wsd_multiplier():
  from discrete_diffusion.utils.utils import wsd_lr_multiplier as f
  n = 1000  # warmup 100, stable until 800, linear decay to 0 over the last 200
  assert f(0, n) == pytest.approx(1 / 101) and f(99, n) == pytest.approx(100 / 101)
  assert f(100, n) == 1.0 and f(799, n) == 1.0
  assert f(800, n) == 1.0 and f(900, n) == pytest.approx(0.5)
  assert f(999, n) == pytest.approx(1 / 200) and f(1000, n) == 0.0
