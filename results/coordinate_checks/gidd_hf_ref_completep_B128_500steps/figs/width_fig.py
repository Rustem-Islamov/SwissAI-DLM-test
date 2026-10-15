import sys, os, math, collections
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/users/rislamov/SwissAI-DLM-test/scripts/analysis')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coordinate_check_fit as F

D = '/users/rislamov/SwissAI-DLM-test/outputs/coordinate_checks/gidd_hf_ref_completep_B128_500steps/width'
runs = F.load(D, 'width')
TOP = 14
cmap = plt.get_cmap('tab20')

def series(arm):
  """-> widths, seeds, mod_steps, {key: array[width, seed, step]}, all_steps, loss, grad_norm"""
  rungs = sorted(runs[arm]); ws = [w for w, _ in rungs]
  seeds = sorted(set.intersection(*[set(runs[arm][r]) for r in rungs]))
  by_step = {(r, s): {row['step']: row for row in runs[arm][r][s]} for r in rungs for s in seeds}
  all_steps = sorted(set.intersection(*[set(d) for d in by_step.values()]))
  mod_steps = [t for t in all_steps if all(any('|' in k for k in d[t]) for d in by_step.values())]
  keys = [k for k in by_step[(rungs[0], seeds[0])][mod_steps[0]] if '|' in k]
  def arr(k, st):
    return np.array([[[by_step[(r, s)][t].get(k) if by_step[(r, s)][t].get(k) is not None else np.nan
                       for t in st] for s in seeds] for r in rungs], float)
  data = {k: arr(k, mod_steps) for k in keys}
  return ws, seeds, np.array(mod_steps), data, np.array(all_steps), arr('loss', all_steps), arr('grad_norm', all_steps)

def exp_over_steps(ws, a):
  """a: [width, seed, step] -> exponents [seed, step] of log2 a vs log2 w"""
  x = np.log2(ws); out = np.full(a.shape[1:], np.nan)
  for s in range(a.shape[1]):
    for t in range(a.shape[2]):
      y = a[:, s, t]
      if np.all(y > 0) and np.all(np.isfinite(y)): out[s, t] = np.polyfit(x, np.log2(y), 1)[0]
  return out

def panel(ax, steps, curves, title, ylabel, tail):
  """curves: {name: exps[seed, step]}"""
  end = {n: (np.nanmean(e[:, -tail:]), np.nanstd(np.nanmean(e[:, -tail:], axis=1))) for n, e in curves.items()}
  order = sorted(curves, key=lambda n: -abs(end[n][0]))
  worst = abs(end[order[0]][0]) if order else float('nan')
  for n in order[TOP:]:
    ax.plot(steps, np.nanmean(curves[n], 0), color='#c8c8c8', lw=.6, zorder=1)
  hs = []
  for i, n in enumerate(order[:TOP]):
    c = cmap(i % 20); m = np.nanmean(curves[n], 0); sd = np.nanstd(curves[n], 0)
    h, = ax.plot(steps, m, color=c, lw=1.1, zorder=3, label=f'{n} : {end[n][0]:+.2f}±{end[n][1]:.2f}')
    ax.fill_between(steps, m - 2 * sd, m + 2 * sd, color=c, alpha=.07, lw=0)
  ax.axhline(0, color='gray', ls='--', lw=.8)
  ax.set_ylim(-.45, .45) if arm == 'on' else None
  ax.set_title(f'{title}   worst|END| = {worst:.3f}   top {min(TOP, len(order))} of {len(order)} labelled (rest gray)', fontsize=9)
  ax.set_ylabel(ylabel, fontsize=9); ax.grid(alpha=.3)
  ax.legend(fontsize=5.5, ncol=2, loc='upper left', framealpha=.85, title='worst nodes  (name : width-exp, mean of last 10% ± SD over seeds)', title_fontsize=5.5)

for arm in ('on', 'off'):
  ws, seeds, steps, data, asteps, loss, gn = series(arm)
  tail = max(1, math.ceil(.1 * len(steps)))
  by = {ch: {k[:-len(ch) - 1]: exp_over_steps(ws, a) for k, a in data.items() if k.endswith('|' + ch)} for ch in F.CHANNELS}
  ratio = {k[:-4]: exp_over_steps(ws, np.abs(data[k[:-4] + '|eff']) / data[k] ) for k in data
           if k.endswith('|act') and k[:-4] + '|eff' in data}
  fig = plt.figure(figsize=(20, 17))
  gs = fig.add_gridspec(4, 2, width_ratios=[2.4, 1], hspace=.38, wspace=.12, top=.92, bottom=.05, left=.05, right=.98)
  panel(fig.add_subplot(gs[0, 0]), steps, by['act'], 'activation  ‖x‖', 'activation ‖x‖\nwidth-exp α', tail)
  panel(fig.add_subplot(gs[1, 0]), steps, by['prop'], 'propagating  ‖W₀·Δx‖', 'propagating\nwidth-exp α', tail)
  panel(fig.add_subplot(gs[2, 0]), steps, by['eff'], 'effective  ‖ΔW·x‖', 'effective\nwidth-exp α', tail)
  panel(fig.add_subplot(gs[3, 0]), steps, ratio, 'effective / act  ‖ΔW·x‖/‖x‖', 'eff / act\nwidth-exp α', tail)
  fig.axes[-1].set_xlabel('step')
  a = fig.add_subplot(gs[0:2, 1]); b = fig.add_subplot(gs[2:4, 1])
  for i, w in enumerate(ws):
    m, sd = np.nanmean(loss[i], 0), np.nanstd(loss[i], 0)
    a.plot(asteps, m, lw=1, label=f'w{w} ({m[-1]:.2f} @step {asteps[-1]}, n={len(seeds)})'); a.fill_between(asteps, m - 2 * sd, m + 2 * sd, alpha=.15, lw=0)
    g = np.exp(np.nanmean(np.log(gn[i]), 0)); lsd = np.nanstd(np.log(gn[i]), 0)
    b.plot(asteps, g, lw=1, label=f'w{w}'); b.fill_between(asteps, g * np.exp(-2 * lsd), g * np.exp(2 * lsd), alpha=.15, lw=0)
  a.set_title(f'loss vs step per width — mean of n={len(seeds)} seeds, band=±2 SD', fontsize=9); a.set_ylabel('per-step train loss'); a.legend(fontsize=7); a.grid(alpha=.3)
  b.set_yscale('log'); b.set_title(f'GLOBAL GRAD NORM per width — geo-mean of n={len(seeds)} seeds, band=±2 log-SD', fontsize=9); b.set_xlabel('step'); b.legend(fontsize=7); b.grid(alpha=.3)
  fig.suptitle(f'GIDD-HF CompleteP width check — arm={arm.upper()} — seeds {seeds}, widths={ws}, L=12\n'
               f'per-step width exponent (log-log fit over widths, each seed); END = mean of last {tail} recorded steps ({len(steps)} logged steps in 0..{steps[-1]}) | lr=2.19e-2 (base), batch=1, seq=2048',
               fontsize=10, fontweight='bold')
  fig.savefig(f'width_modules_{arm}.png', dpi=130)
  print(arm, ws, seeds, len(steps), 'worst', {ch: max(abs(np.nanmean(e[:, -tail:])) for e in by[ch].values()) for ch in by})
