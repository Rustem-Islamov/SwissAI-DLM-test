#!/usr/bin/env python
"""Coordinate checks for the GIDD-HF muP / CompleteP parametrization.

Implements SwissAI-DLM_coordinate_checks.md on the *real* training path: the GIDD algorithm
module, its loss, ``optim=adamw_mup`` and the nemotron pretok data, driven by a plain loop
(no Lightning ``Trainer``), constant peak LR, no warmup, no gradient clipping.

Axes (``--axis``)
  width   hidden_size ladder at fixed depth.        per-module act / prop / eff, last 10% of steps
  depth   n_blocks ladder at fixed width.            one scalar ||h_L^t - h_L^0||_2, steps 2..10
  cosc    width and depth together (w / L fixed).    same scalar, rmsnorm, steps 2..10
  noop    base model, both arms, matched seeds.      test 6b: the arms must be bit-identical

Arms (``--arm``)
  on   model.parametrization=completep
  off  width axis: ``standard`` (no width scaling);   depth / cosc / noop: ``mup`` (width muP, no
       depth correction).  In both, ``model.resid_scale = n_blocks`` so the residual-branch
       multiplier ``resid_scale / n_blocks`` is exactly 1.0, as ``depth_scaling=False`` is in the
       bst-scaling reference.  (``mup`` / ``standard`` alone keep a 1/L branch multiplier, which
       is already a depth control and would make the ``off`` arm look flat.)
  At L == base_depth the ``on`` multiplier is ``base_depth / L == 1.0`` too, so on/off must
  coincide there (``--axis noop``).

Channels, per monitored module, all measured on the *same* batch / noised input ``z_t``
against a frozen copy of the model at initialisation (the refined coordinate check):
  act   rms of the module output                               is the output the right size?
  prop  rms of (x_t - x_0), the module-input change             is the propagated change right?
  eff   rms of ((W_t - W_0) x_t)                                is the update right?
plus ``stream/*``: the residual stream after the last block (= input of the final norm),
``stream/l2`` = rms * sqrt(width) (per-token l2) and ``stream/rms``.

Output: one JSONL per (axis, arm, width, depth, seed) in ``--out-dir``; fit with
``coordinate_check_fit.py``.  Run inside the training container on one GPU, e.g. via
``scripts/training/submit_coordinate_check.sh``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import pathlib
import re
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import hydra  # noqa: E402
import lightning as L  # noqa: E402
import omegaconf  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

import discrete_diffusion.__main__  # noqa: E402,F401  (registers the OmegaConf resolvers)
from discrete_diffusion.data import get_dataloaders, get_tokenizer  # noqa: E402
from discrete_diffusion.models.modeling_gidd_hf import GiddRMSNorm, ScaledLinear  # noqa: E402
from discrete_diffusion.models.scaling import classify_parameter  # noqa: E402

CONFIG_DIR = (REPO_ROOT / 'configs').as_posix()
LAYER_RE = re.compile(r'(?:^|\.)layers\.(\d+)\.')


# --------------------------------------------------------------------------------------
# Ladders
# --------------------------------------------------------------------------------------
def ladder(args):
  """-> list of (hidden_size, n_blocks) rungs for the axis."""
  if args.axis == 'width':
    return [(w, args.depth) for w in args.widths]
  if args.axis == 'depth':
    return [(args.width, d) for d in args.depths]
  if args.axis == 'cosc':
    return [(w, max(1, round(w / args.aspect))) for w in args.widths]
  if args.axis == 'noop':
    return [(args.width, args.base_depth)]
  raise ValueError(args.axis)


def check_ladder(args, rungs):
  heads = [w % args.head_dim for w, _ in rungs]
  if any(heads):
    raise SystemExit(f'hidden sizes must be multiples of head_dim={args.head_dim}: {rungs}')
  if len(rungs) < 4 and args.axis != 'noop':
    print(f'[warn] only {len(rungs)} rungs; the docs ask for >= 4 (width) / >= 3 above base (depth).')
  if args.axis in ('depth', 'cosc'):
    above = sum(1 for _, d in rungs if d > args.base_depth)
    at = sum(1 for _, d in rungs if d == args.base_depth)
    print(f'[ladder] depth rungs strictly above base_depth={args.base_depth}: {above}, at base: {at}')
    if above < 3 and args.axis == 'depth':
      print('[warn] depth ladder has < 3 rungs strictly above base_depth (doc section 4).')
  if args.axis == 'width' and args.base_width in [w for w, _ in rungs]:
    print(f'[ladder] the width {args.base_width} rung is the base: scales are exactly 1 there.')


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def build_overrides(args, hidden, depth, arm, seed):
  if arm == 'on':
    param, resid = 'completep', ['model.resid_mult=1.0']
  else:
    param = 'standard' if args.axis == 'width' else 'mup'
    resid = [f'model.resid_scale={float(depth)}']  # branch multiplier resid_scale / n_blocks == 1
  loader_bs = args.batch_size
  ov = [
      f'seed={seed}',
      f'data={args.data}',
      f'model=gidd_hf_ref',
      f'model.parametrization={param}',
      f'model.base_width={args.base_width}',
      f'model.base_depth={args.base_depth}',
      f'model.hidden_size={hidden}',
      f'model.n_heads={hidden // args.head_dim}',
      f'model.n_blocks={depth}',
      f'model.length={args.seq_len}',
      f'model.max_tokens=1024',
      f'model.dropout=0.0',
      f'model.attn_attn_backend={args.attn_backend}',
      f'algo=gidd',
      f'algo.loss_type=gidd_easydel_lowmem',
      f'algo.hybrid_mixing_shift=-1000',
      f'algo.loss_weighting=dynamic',
      f'algo.low_discrepancy_sampling=true',
      f'algo.time_sampling_scope=local',
      # constant peak LR, no warmup / decay (the bst-scaling coordinate-check setting)
      f'lr_scheduler=wsd',
      f'lr_scheduler.warmup_frac=0.0',
      f'lr_scheduler.decay_frac=0.0',
      f'optim=adamw_mup',
      f'optim.lr={args.lr}',
      f'optim.weight_decay={args.weight_decay}',
      f'optim.beta1=0.9',
      f'optim.beta2=0.95',
      f'optim.eps=1e-14',
      f'trainer.max_steps={args.steps}',
      f'trainer.devices=1',
      f'trainer.num_nodes=1',
      f'trainer.accumulate_grad_batches=1',
      f'loader.global_batch_size={loader_bs}',
      f'loader.eval_global_batch_size={loader_bs}',
      f'loader.batch_size={loader_bs}',
      f'loader.eval_batch_size={loader_bs}',
      f'loader.num_workers={args.num_workers}',
      f'loader.pin_memory=false',
      f'training.antithetic_sampling=true',
      f'training.ema=0',
      f'checkpointing.save_dir={args.out_dir}/_unused',
      f'++wandb=null',
  ]
  if args.pretok_dir:
    ov.append(f'data.pretok_local_dir={args.pretok_dir}')
  if args.data_cache_dir:
    ov.append(f'data.cache_dir={args.data_cache_dir}')
  return ov + resid + list(args.override)


# --------------------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------------------
def rms(x):
  return x.float().pow(2).mean().sqrt().item()


def chunked_rms_of(fn, x, chunk=2048):
  """rms of fn(x_chunk) concatenated over rows of ``x`` [..., d], without materialising it."""
  x = x.reshape(-1, x.shape[-1])
  total, count = 0.0, 0
  for i in range(0, x.shape[0], chunk):
    y = fn(x[i:i + chunk]).float()
    total += y.pow(2).sum().item()
    count += y.numel()
  return math.sqrt(total / max(count, 1))


def short(name):
  return name[len('model.model.'):] if name.startswith('model.model.') else name


class CoordinateMonitor:
  """Refined coordinate check against a frozen init copy of the backbone."""

  def __init__(self, backbone, layers, stream_only):
    self.cur = backbone
    self.ref = copy.deepcopy(backbone).eval()
    for p in self.ref.parameters():
      p.requires_grad_(False)
    self.ref_mods = dict(self.ref.named_modules())
    self.cur_mods = dict(self.cur.named_modules())
    self.names = self._select(layers, stream_only)
    self.final_norm = 'model.model.norm'
    assert self.final_norm in self.names, 'final norm not monitored'
    self._hooks = []
    self.z_t = None
    self.metrics = {}
    self._cur_in, self._cur_out = {}, {}
    self.active = False
    self.cur.register_forward_pre_hook(self._grab_input)

  def _select(self, layers, stream_only):
    names = []
    for name, m in self.cur.named_modules():
      if name == 'model.model.norm':
        names.append(name)
        continue
      match = LAYER_RE.search(name)
      if stream_only:
        if match and name.endswith('attn_layernorm'):
          names.append(name)  # stream entering block l
        continue
      if name in ('model.model.embed_tokens', 'model.lm_head'):
        names.append(name)
      elif isinstance(m, (ScaledLinear, GiddRMSNorm)) and match and int(match.group(1)) in layers:
        names.append(name)
    return names

  def _grab_input(self, module, args):
    if self.active:
      self.z_t = args[0].detach()

  # -- one monitored step -----------------------------------------------------------
  def begin(self):
    self.metrics, self._cur_in, self._cur_out = {}, {}, {}
    self.active = True
    for name in self.names:
      self._hooks.append(self.cur_mods[name].register_forward_hook(self._cur_hook(name)))

  def _clear(self):
    for h in self._hooks:
      h.remove()
    self._hooks = []

  def _cur_hook(self, name):
    def hook(module, args, output):
      key = short(name)
      out = output[0] if isinstance(output, tuple) else output
      self.metrics[f'{key}|act'] = rms(out) if not name.endswith('lm_head') else rms(out[..., :4096])
      if isinstance(module, torch.nn.Embedding):
        self._cur_out[name] = out.detach()
      else:
        self._cur_in[name] = args[0].detach()
    return hook

  @torch.no_grad()
  def finish(self, autocast_dtype):
    """Run the init copy on the same z_t and compute prop / eff.  Call after the current forward."""
    self._clear()
    self.active = False
    handles = [self.ref_mods[n].register_forward_hook(self._ref_hook(n)) for n in self.names]
    try:
      with torch.autocast('cuda', dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
        self.ref(self.z_t)
    finally:
      for h in handles:
        h.remove()
    self._cur_in.clear(), self._cur_out.clear()
    return self.metrics

  def _ref_hook(self, name):
    def hook(module_ref, args, output):
      key = short(name)
      module = self.cur_mods[name]
      if isinstance(module, torch.nn.Embedding):
        out_ref = output[0] if isinstance(output, tuple) else output
        self.metrics[f'{key}|eff'] = rms(self._cur_out[name].float() - out_ref.float())
        return
      x_cur, x_ref = self._cur_in[name], args[0]
      delta = x_cur.float() - x_ref.float()
      prop = rms(delta)
      self.metrics[f'{key}|prop'] = prop
      if name == self.final_norm:
        width = x_cur.shape[-1]
        self.metrics['stream/rms'] = prop
        self.metrics['stream/l2'] = prop * math.sqrt(width)
      if isinstance(module, ScaledLinear):
        dw = module.weight.detach().float() - module_ref.weight.float()
        if name.endswith('lm_head'):
          # sample of rows only: the full [tokens, vocab] product is too large and unnecessary
          rows = torch.linspace(0, dw.shape[0] - 1, 4096, device=dw.device).long()
          dw = dw[rows]
        self.metrics[f'{key}|eff'] = chunked_rms_of(
            lambda xc: F.linear(xc.float(), dw), x_cur)
      elif isinstance(module, GiddRMSNorm) and module.weight is not None:
        eps = torch.finfo(x_cur.dtype).eps if module.epsilon is None else module.epsilon
        dg = (module.weight.detach().float() - module_ref.weight.float())
        def fn(xc):
          xc = xc.float()
          return xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps) * dg
        self.metrics[f'{key}|eff'] = chunked_rms_of(fn, x_cur)
    return hook


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------
def monitored(step, steps, every):
  tail = max(1, math.ceil(0.1 * steps))
  return step <= 10 or step >= steps - tail or step % every == 0


def role_report(algo):
  roles = {}
  for name, p in algo._named_trainable_parameters():
    roles.setdefault(classify_parameter(name), []).append(name)
  if 'other' in roles:
    # test 6c: nothing may be silently unclassified
    raise RuntimeError(f'unclassified parameters (role "other"): {roles["other"][:10]}')
  return {r: len(v) for r, v in roles.items()}


def run_one(args, tokenizer, hidden, depth, arm, seed):
  tag = f'{args.axis}_{arm}_w{hidden}_L{depth}_s{seed}'
  out_path = pathlib.Path(args.out_dir) / f'{tag}.jsonl'
  if out_path.exists() and not args.overwrite:
    print(f'[skip] {tag} (exists)')
    return
  overrides = build_overrides(args, hidden, depth, arm, seed)
  cfg = compose(config_name='config', overrides=overrides)
  L.seed_everything(seed)
  algo_cls = hydra.utils.get_class(cfg.algo._target_)
  algo = algo_cls(cfg, tokenizer=tokenizer)
  algo.cuda().train()
  roles = role_report(algo)

  optimizer = algo.configure_optimizers()[0][0]
  groups = [{'name': g['name'], 'lr': g['lr'], 'eps': g['eps'], 'n': len(g['params']),
             'lambda_decay': g['lambda_decay']} for g in optimizer.param_groups]
  n_params = sum(p.numel() for p in algo.parameters())
  print(f'[run] {tag}: {n_params / 1e6:.1f}M params, resid_scale={algo.backbone.model.model.resid_scale}')
  for g in groups:
    print(f"        group {g['name']:<10} lr={g['lr']:.4e} eps={g['eps']:.3e} tensors={g['n']}")

  # lr_scheduler=wsd with warmup_frac = decay_frac = 0 is the constant multiplier 1, so every
  # group keeps lr * lr_scale(role); the scheduler is never stepped.
  autocast_dtype = torch.bfloat16 if args.autocast == 'bf16' else torch.float32
  layers = sorted({0, depth // 2, depth - 1})
  stream_only = args.axis in ('depth', 'cosc')
  mon = CoordinateMonitor(algo.backbone, layers, stream_only)

  train_loader, _ = get_dataloaders(cfg, tokenizer, skip_valid=True)
  batches = iter(itertools.cycle(train_loader))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  with open(out_path, 'w') as fh:
    fh.write(json.dumps({
        'meta': True, 'axis': args.axis, 'arm': arm, 'width': hidden, 'depth': depth,
        'seed': seed, 'lr': args.lr, 'warmup_steps': args.warmup_steps, 'grad_clip': args.grad_clip, 'steps': args.steps, 'batch_size': args.batch_size, 'accum': args.accum, 'global_batch': args.batch_size * args.accum,
        'seq_len': args.seq_len, 'base_width': args.base_width, 'base_depth': args.base_depth,
        'parametrization': cfg.model.parametrization, 'groups': groups, 'roles': roles,
        'n_params': n_params, 'monitored_modules': [short(n) for n in mon.names],
        'resid_multiplier': float(algo.backbone.model.model.resid_scale),
    }) + '\n')
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    for step in range(args.steps):
      if args.warmup_steps > 0:  # linear warmup of every group's lr, applied before optimizer.step()
        scale = min(1.0, (step + 1) / args.warmup_steps)
        for g, lr0 in zip(optimizer.param_groups, base_lrs):
          g['lr'] = lr0 * scale
      t_step = time.time()
      optimizer.zero_grad(set_to_none=True)
      watch = monitored(step, args.steps, args.log_every)
      loss_sum = 0.0
      for micro in range(args.accum):
        batch = next(batches)
        x0 = batch['input_ids'].cuda(non_blocking=True)
        valid = batch['attention_mask'].cuda(non_blocking=True)
        last = micro == args.accum - 1
        if watch and last:  # module statistics come from the last micro-batch only
          mon.begin()
        with torch.autocast('cuda', dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
          loss = algo._loss(x0, valid, 0, True).loss
        loss_sum += loss.item()
        (loss / args.accum).backward()  # frees the autograd graph; weights unchanged until optimizer.step()
        if watch and last:
          row_stats = mon.finish(autocast_dtype)  # init-copy pass on the same z_t, at W_t
      row = {'step': step, 'loss': loss_sum / args.accum}
      if watch:
        row.update(row_stats)
      row['grad_norm'] = torch.nn.utils.clip_grad_norm_(algo.parameters(), args.grad_clip).item()  # logged pre-clip norm
      optimizer.step()
      torch.cuda.synchronize()
      row['dt'] = round(time.time() - t_step, 4)  # wall time of this step (incl. monitoring when watched)
      if not math.isfinite(row['loss']):
        fh.write(json.dumps(row) + '\n')
        print(f'[run] {tag}: non-finite loss at step {step}, stopping.')
        break
      fh.write(json.dumps(row) + '\n')
      fh.flush()
  print(f'[run] {tag} done in {time.time() - t0:.0f}s -> {out_path}')
  del algo, mon, optimizer, batches, train_loader
  torch.cuda.empty_cache()


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('--axis', choices=['width', 'depth', 'cosc', 'noop'], required=True)
  p.add_argument('--arm', choices=['on', 'off', 'both'], default='both')
  p.add_argument('--out-dir', required=True)
  p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
  p.add_argument('--steps', type=int, default=None,
                 help='default: 100 (width), 20 (depth / cosc, window is steps 2..10), 12 (noop)')
  p.add_argument('--widths', type=int, nargs='+', default=None,
                 help='width: 256 512 1024 2048 | cosc: 512 768 1024 1536')
  p.add_argument('--depths', type=int, nargs='+', default=[8, 16, 32, 64])
  p.add_argument('--width', type=int, default=512, help='fixed width of the depth axis / noop')
  p.add_argument('--depth', type=int, default=12, help='fixed depth of the width axis')
  p.add_argument('--aspect', type=int, default=64, help='cosc: n_blocks = width / aspect')
  p.add_argument('--base-width', type=int, default=256)
  p.add_argument('--base-depth', type=int, default=12)
  p.add_argument('--head-dim', type=int, default=64, help='fixed; n_heads = hidden_size / head_dim')
  p.add_argument('--lr', type=float, default=2.19116e-2,
                 help='peak LR at base_width (width-256 convention of optim=adamw_mup)')
  p.add_argument('--weight-decay', type=float, default=1e-4)
  p.add_argument('--warmup-steps', type=int, default=0, help='linear LR warmup steps (0 = constant peak LR)')
  p.add_argument('--grad-clip', type=float, default=float('inf'), help='global grad-norm clip (inf = none)')
  p.add_argument('--batch-size', type=int, default=4, help='micro-batch (sequences per forward/backward)')
  p.add_argument('--accum', type=int, default=1, help='gradient-accumulation micro-batches; global batch = batch_size * accum')
  p.add_argument('--seq-len', type=int, default=2048)
  p.add_argument('--log-every', type=int, default=10)
  p.add_argument('--autocast', choices=['bf16', 'fp32'], default='bf16')
  p.add_argument('--attn-backend', default='flash_attn2')
  p.add_argument('--data', default='nemotron-cc-pretok')
  p.add_argument('--pretok-dir', default=None)
  p.add_argument('--data-cache-dir', default=None)
  p.add_argument('--num-workers', type=int, default=0)
  p.add_argument('--override', action='append', default=[], help='extra Hydra override (repeatable)')
  p.add_argument('--overwrite', action='store_true')
  p.add_argument('--shard-id', type=int, default=0, help='this worker runs tasks[shard_id::num_shards]')
  p.add_argument('--num-shards', type=int, default=1, help='one worker per GPU of the node')
  args = p.parse_args()

  if args.steps is None:
    args.steps = {'width': 100, 'depth': 20, 'cosc': 20, 'noop': 12}[args.axis]
  if args.widths is None:
    args.widths = [512, 768, 1024, 1536] if args.axis == 'cosc' else [256, 512, 1024, 2048]
  arms = ['on', 'off'] if args.arm == 'both' or args.axis == 'noop' else [args.arm]
  rungs = ladder(args)
  check_ladder(args, rungs)
  pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

  torch.set_float32_matmul_precision('high')
  with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg0 = compose(config_name='config', overrides=[f'data={args.data}', 'model=gidd_hf_ref'])
    tokenizer = get_tokenizer(cfg0)
    # seed-major so that, for a given seed, all rungs / arms see the same data stream
    tasks = [(seed, arm, hidden, depth) for seed in args.seeds for arm in arms
             for hidden, depth in rungs]
    mine = tasks[args.shard_id::args.num_shards]
    print(f'[shard {args.shard_id}/{args.num_shards}] {len(mine)} of {len(tasks)} runs')
    for seed, arm, hidden, depth in mine:
      run_one(args, tokenizer, hidden, depth, arm, seed)


if __name__ == '__main__':
  main()
