import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/noop'
def load(arm, s):
    rows = [json.loads(l) for l in open(f'{D}/noop_{arm}_w768_L12_s{s}.jsonl') if l.strip()][1:]
    return {r['step']: r for r in rows}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
for s in (1, 2, 3, 4):
    on, off = load('on', s), load('off', s)
    st = sorted(set(on) & set(off))
    lo, lf = np.array([on[t]['loss'] for t in st]), np.array([off[t]['loss'] for t in st])
    ax[0].plot(st, lo, '-', label=f'on s{s}'); ax[0].plot(st, lf, '--', color='k', lw=.7)
    ax[1].plot(st, lo - lf, '-o', ms=3, label=f'seed {s}')
    k = 'grad_norm'
    ax[2].semilogy(st, np.abs(np.array([on[t][k] for t in st]) / np.array([off[t][k] for t in st]) - 1) + 1e-12, '-o', ms=3, label=f'seed {s}')
ax[0].set_title('loss vs step (solid on, dashed off)'); ax[1].set_title('loss(on) − loss(off): exactly 0 at step 0, mixed signs after')
ax[2].set_title('|grad_norm on/off − 1|'); 
for a in ax: a.set_xlabel('step'); a.grid(alpha=.3); a.legend(fontsize=7)
ax[1].axhline(0, color='gray', ls='--')
fig.suptitle('Base-rung no-op test, w768_L12 (on vs off must be identical)')
fig.savefig(os.path.dirname(os.path.abspath(__file__)) + '/noop_check.png', dpi=130)
