import sys, collections, math, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/users/rislamov/SwissAI-DLM-test/scripts/analysis')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coordinate_check_fit as F

D = '/users/rislamov/SwissAI-DLM-test/outputs/coordinate_checks/gidd_hf_ref_completep_B128_500steps'
COL = {'off': '#eb6834', 'on': '#2a78d6'}   # categorical slots 2 / 1
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e6e5e1'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .7})

def scalar_curves(axis):
  runs = F.load(f'{D}/{axis}', axis)
  key = 'stream/l2' if axis == 'depth' else 'stream/rms'
  steps = list(range(2, 11)); out = {}
  for arm in runs:
    rungs = sorted(runs[arm]); xs = [d if axis == 'depth' else w for w, d in rungs]
    _, ps = F.stream_curve(runs[arm], key, '', steps)
    out[arm] = [(t, *F.exponent(xs, ps[t])[:2]) for t in steps]
  return out

def width_exps():
  runs = F.load(f'{D}/width', 'width'); res = {}
  for arm in runs:
    rungs = sorted(runs[arm]); ws = [w for w, _ in rungs]
    keys = {k for rows in runs[arm][rungs[0]].values() for r in rows for k in r if k.split('|')[-1] in F.CHANNELS}
    for k in keys:
      vals = [{s: F.mean_key(F.last_decile(rows), k) for s, rows in runs[arm][r].items()} for r in rungs]
      if any(all(math.isnan(v) for v in d.values()) for d in vals): continue
      e, se, n = F.exponent(ws, vals)
      if math.isfinite(e): res.setdefault(arm, {})[k] = e
  return res

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for a, axis, title in [(ax[0], 'depth', 'Depth: exponent of $\\|h_L^t-h_L^0\\|$ vs L'),
                       (ax[1], 'cosc', 'Co-scale (w/L=64): exponent vs width')]:
  c = scalar_curves(axis)
  for arm in ('off', 'on'):
    t, e, se = map(np.array, zip(*c[arm]))
    a.plot(t, e, '-o', ms=4, lw=2, color=COL[arm])
    a.fill_between(t, e - se, e + se, color=COL[arm], alpha=.18, lw=0)
    a.text(10.2, e[-1], f'{arm}', color=INK, va='center', fontsize=9)
  a.axhline(0, color=INK, lw=1, ls='--')
  a.set_xlabel('step t'); a.set_ylabel('fitted exponent (ideal = 0)'); a.set_title(title, fontsize=10.5, loc='left')
  a.set_xlim(1.5, 11); a.legend(handles=[plt.Line2D([], [], color=COL[x], lw=2, marker='o', ms=4, label=f'{x} (±1 s.e.)') for x in ('off', 'on')], frameon=False, loc='center left')

w = width_exps(); a = ax[2]; rng = np.random.default_rng(0)
for i, ch in enumerate(F.CHANNELS):
  for j, arm in enumerate(('off', 'on')):
    ks = [k for k in w[arm] if k.endswith('|' + ch)]; ys = np.array([w[arm][k] for k in ks])
    x = i + (j - .5) * .34 + rng.uniform(-.07, .07, len(ys))
    a.scatter(x, ys, s=16, color=COL[arm], alpha=.75, lw=0)
    a.hlines(np.mean(np.abs(ys)), i + (j - .5) * .34 - .12, i + (j - .5) * .34 + .12, color=INK, lw=2)
a.axhline(0, color=INK, lw=1, ls='--'); a.set_xticks(range(3)); a.set_xticklabels(['act', 'prop', 'eff'])
a.set_ylabel('exponent vs width (ideal = 0)'); a.set_title('Width: per-module exponents (bar = mean |e|)', fontsize=10.5, loc='left')
a.legend(handles=[plt.Line2D([], [], color=COL[x], marker='o', ls='', ms=5, label=x) for x in ('off', 'on')], frameon=False, loc='upper right')
fig.suptitle('CompleteP coordinate checks: off arm fails, on arm ≈ flat (noop check: step-0 loss bitwise identical, 4/4 seeds)', x=0.01, ha='left', fontsize=11.5)
fig.savefig('coordcheck_summary.png', dpi=160)

# layer profile of residual width drift (on arm)
fig, a = plt.subplots(figsize=(7, 4), constrained_layout=True)
for ch, m in (('act', 'o'), ('eff', 's')):
  pass
lay = collections.defaultdict(list)
for k, e in w['on'].items():
  if k.endswith('|act') and k.startswith('layers.'): lay[int(k.split('.')[1])].append(e)
L = sorted(lay); a.bar(L, [np.mean(np.abs(lay[l])) for l in L], color=COL['on'], width=.6)
a.set_xlabel('layer'); a.set_ylabel('mean |exponent| of act, on arm'); a.set_xticks(L)
a.set_title('Width drift (on arm) by layer: layer 6 stands out', fontsize=10.5, loc='left')
fig.savefig('coordcheck_width_layers.png', dpi=160)
print({l: round(float(np.mean(np.abs(lay[l]))), 3) for l in L})
