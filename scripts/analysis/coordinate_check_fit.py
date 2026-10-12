#!/usr/bin/env python
"""Fit the exponents of the coordinate-check JSONL written by ``coordinate_check.py``.

  python coordinate_check_fit.py width DIR     per-module act/prop/eff vs log2(width), last 10% of steps
  python coordinate_check_fit.py depth DIR     ||h_L^t - h_L^0||_2 vs log2(L), steps 2..10
  python coordinate_check_fit.py cosc  DIR     same scalar, rmsnorm, vs log2(width), L tied to it
  python coordinate_check_fit.py noop  DIR     base-rung no-op test: on vs off, matched seeds

Every exponent comes with a seed standard error; ``t = |exponent| / se`` below 2 is reported as
"unresolved", not as a pass or a fail.  Pass criteria: see SwissAI-DLM_coordinate_checks.md
section 8 (the ``off`` arm FAILING is part of the depth criterion).  ``--plot`` writes the
Fig-7 layout (one panel per arm, log-log, one line per step) for depth / cosc.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import numpy as np

CHANNELS = ('act', 'prop', 'eff')


def load(directory, axis):
  """-> {arm: {(width, depth): {seed: [row, ...]}}}, meta"""
  runs = collections.defaultdict(lambda: collections.defaultdict(dict))
  for path in sorted(glob.glob(os.path.join(directory, f'{axis}_*.jsonl'))):
    with open(path) as fh:
      lines = [json.loads(l) for l in fh if l.strip()]
    if not lines or not lines[0].get('meta'):
      continue
    meta, rows = lines[0], lines[1:]
    runs[meta['arm']][(meta['width'], meta['depth'])][meta['seed']] = rows
  return runs


def slope(x, y):
  return float(np.polyfit(np.log2(x), np.log2(y), 1)[0])


def exponent(xs, per_seed_values):
  """Fit log2 value vs log2 x for each seed, return (mean exponent, standard error, n_seeds).

  ``per_seed_values[i]`` is {seed: value} at ``xs[i]``; only seeds present at every rung count.
  """
  seeds = set.intersection(*[set(v) for v in per_seed_values])
  slopes = []
  for s in sorted(seeds):
    ys = [v[s] for v in per_seed_values]
    if all(y > 0 and math.isfinite(y) for y in ys):
      slopes.append(slope(xs, ys))
  if not slopes:
    return float('nan'), float('nan'), 0
  sl = np.array(slopes)
  se = sl.std(ddof=1) / math.sqrt(len(sl)) if len(sl) > 1 else float('nan')
  return float(sl.mean()), float(se), len(sl)


def verdict(e, se):
  if not math.isfinite(se) or se == 0:
    return '?'
  t = abs(e) / se
  return 'unresolved' if t < 2 else f't={t:.1f}'


def last_decile(rows):
  steps = [r['step'] for r in rows]
  hi = max(steps)
  lo = hi - max(1, math.ceil(0.1 * (hi + 1))) + 1
  return [r for r in rows if lo <= r['step'] <= hi]


def mean_key(rows, key):
  vals = [r[key] for r in rows if key in r and r[key] is not None and math.isfinite(r[key])]
  return float(np.mean(vals)) if vals else float('nan')


# --------------------------------------------------------------------------------------
def fit_width(runs, args):
  for arm in sorted(runs):
    rungs = sorted(runs[arm])
    widths = [w for w, _ in rungs]
    print(f'\n=== width check, arm={arm}  widths={widths}  seeds/rung={[len(runs[arm][r]) for r in rungs]}')
    keys = collections.OrderedDict()
    for rows in runs[arm][rungs[0]].values():
      for r in rows:
        for k in r:
          if k.split('|')[-1] in CHANNELS:
            keys[k] = None
    table = []
    for key in keys:
      vals = []
      for r in rungs:
        vals.append({s: mean_key(last_decile(rows), key) for s, rows in runs[arm][r].items()})
      if any(all(math.isnan(v) for v in d.values()) for d in vals):
        continue
      e, se, n = exponent(widths, vals)
      table.append((key, e, se, n))
    table = [t for t in table if math.isfinite(t[1])]
    by_channel = collections.defaultdict(list)
    for key, e, se, n in table:
      by_channel[key.split('|')[-1]].append((abs(e), key, e, se))
    for ch in CHANNELS:
      if not by_channel[ch]:
        continue
      es = sorted(by_channel[ch], reverse=True)
      print(f'  [{ch}] modules={len(es)}  mean|e|={np.mean([a for a, *_ in es]):.3f}  worst={es[0][2]:+.3f}')
    print(f'  worst {args.top} channels:')
    for key, e, se, n in sorted(table, key=lambda t: -abs(t[1]))[:args.top]:
      print(f'    {key:<48} e={e:+.3f} +- {se:.3f}  ({verdict(e, se)}, n={n})')


# --------------------------------------------------------------------------------------
def stream_curve(runs_arm, key, norm_label, steps):
  """-> xs, {step: [ {seed: value} per rung ]}"""
  rungs = sorted(runs_arm)
  per_step = {}
  for t in steps:
    per_step[t] = []
    for r in rungs:
      d = {}
      for s, rows in runs_arm[r].items():
        for row in rows:
          if row['step'] == t and key in row:
            d[s] = row[key]
      per_step[t].append(d)
  return rungs, per_step


def fit_scalar(runs, args, axis):
  key = 'stream/l2' if axis == 'depth' else 'stream/rms'
  norm = 'l2norm' if axis == 'depth' else 'rmsnorm'
  xlabel = 'depth L' if axis == 'depth' else 'width (L tied)'
  steps = list(range(args.t_min, args.t_max + 1))
  summary = {}
  for arm in sorted(runs):
    rungs = sorted(runs[arm])
    xs = [d if axis == 'depth' else w for w, d in rungs]
    _, per_step = stream_curve(runs[arm], key, norm, steps)
    print(f'\n=== {axis} check, arm={arm}  {xlabel}={xs}  norm={norm}  window=steps {steps[0]}..{steps[-1]}')
    es = []
    for t in steps:
      if not all(per_step[t]):
        continue
      e, se, n = exponent(xs, per_step[t])
      es.append((t, e, se, n))
      print(f'    t={t:<3} exponent={e:+.3f} +- {se:.3f}  (n_seeds={n})')
    if es:
      mean_abs = float(np.mean([abs(e) for _, e, _, _ in es]))
      worst = max(es, key=lambda x: abs(x[1]))
      mean_se = float(np.nanmean([se for _, _, se, _ in es]))
      summary[arm] = mean_abs
      print(f'  -> mean |exponent| = {mean_abs:.3f}   worst = {worst[1]:+.3f} (t={worst[0]})   mean se = {mean_se:.3f}')
      if mean_se > 0.05:
        print('     WARNING: seed band > 0.05; the doc reports +-0.01 for this quantity. '
              'Check the norm / window before reading anything into the exponent.')
  if {'on', 'off'} <= set(summary):
    print(f'\n  off = {summary["off"]:.3f}, on = {summary["on"]:.3f}')
    if summary['off'] < 0.4:
      print('  The off arm is flat: the ladder cannot see the effect, so this proves nothing about the correction.')
    elif summary['on'] > 0.25:
      print('  The on arm is NOT flat: depth control is not working on this ladder.')
    else:
      print('  off fails, on is flat: the depth correction does its job.')
  if axis == 'cosc':
    print('  (co-scale is composition evidence only; certify depth on the pure-depth axis.)')
  if args.plot:
    plot_fig7(runs, args, axis, key, norm, xlabel, steps)


def plot_fig7(runs, args, axis, key, norm, xlabel, steps):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  arms = [a for a in ('off', 'on') if a in runs]
  fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), sharey=True, squeeze=False)
  cmap = plt.get_cmap('viridis')
  for ax, arm in zip(axes[0], arms):
    rungs = sorted(runs[arm])
    xs = [d if axis == 'depth' else w for w, d in rungs]
    _, per_step = stream_curve(runs[arm], key, norm, steps)
    for i, t in enumerate(steps):
      if not all(per_step[t]):
        continue
      mean = [np.mean(list(d.values())) for d in per_step[t]]
      lo = [np.min(list(d.values())) for d in per_step[t]]
      hi = [np.max(list(d.values())) for d in per_step[t]]
      c = cmap(i / max(1, len(steps) - 1))
      ax.plot(xs, mean, '-o', color=c, ms=3, label=f't={t}')
      ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
    ref = [np.mean(list(d.values())) for d in per_step[steps[0]]] if all(per_step[steps[0]]) else None
    if ref:
      ax.plot(xs, [ref[0]] * len(xs), 'k--', lw=1)
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    ax.set_xlabel(xlabel), ax.set_title(f'{arm}  ({norm})')
  axes[0][0].set_ylabel(r'$\|h_L^t - h_L^0\|$')
  axes[0][-1].legend(fontsize=6, ncol=2)
  path = os.path.join(args.dir, f'{axis}_fig7.png')
  fig.tight_layout(), fig.savefig(path, dpi=150)
  print(f'  wrote {path}')


# --------------------------------------------------------------------------------------
def fit_noop(runs, args):
  """Test 6b: at L == base_depth the 'on' multiplier is exactly 1, so on == off."""
  if not {'on', 'off'} <= set(runs):
    raise SystemExit('need both arms')
  (rung,) = set(runs['on']) & set(runs['off'])
  seeds = sorted(set(runs['on'][rung]) & set(runs['off'][rung]))
  print(f'\n=== base-rung no-op, rung (w,L)={rung}, seeds={seeds}')
  for s in seeds:
    a, b = runs['on'][rung][s], runs['off'][rung][s]
    n = min(len(a), len(b))
    print(f'  seed {s}: step-0 loss on={a[0]["loss"]!r} off={b[0]["loss"]!r}  '
          f'{"BITWISE identical" if a[0]["loss"] == b[0]["loss"] else "DIFFERENT"}')
    worst = {}
    for window, (lo, hi) in {'t=0..1': (0, 1), 't=2..10': (2, 10), 't>10': (11, 10**9)}.items():
      rel = []
      for ra, rb in zip(a[:n], b[:n]):
        if lo <= ra['step'] <= hi:
          for k, v in ra.items():
            if k != 'step' and k in rb and isinstance(v, float) and math.isfinite(v) and v != 0:
              rel.append(abs(v - rb[k]) / abs(v))
      worst[window] = max(rel) if rel else float('nan')
    print('           worst rel diff: ' + '  '.join(f'{w}: {x:.2e}' for w, x in worst.items()))
  print('  Step 0 must be bitwise identical.  Late drift with MIXED loss-difference signs across '
        'seeds is GPU nondeterminism, not the flag.')


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument('axis', choices=['width', 'depth', 'cosc', 'noop'])
  p.add_argument('dir')
  p.add_argument('--top', type=int, default=12)
  p.add_argument('--t-min', type=int, default=2)
  p.add_argument('--t-max', type=int, default=10)
  p.add_argument('--plot', action='store_true')
  args = p.parse_args()
  runs = load(args.dir, args.axis)
  if not runs:
    raise SystemExit(f'no {args.axis}_*.jsonl in {args.dir}')
  {'width': fit_width, 'noop': fit_noop}.get(
      args.axis, lambda r, a: fit_scalar(r, a, args.axis))(runs, args)


if __name__ == '__main__':
  main()
