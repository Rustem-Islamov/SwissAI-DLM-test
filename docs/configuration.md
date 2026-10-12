# Configuration reference

SwissAI-DLM uses [Hydra](https://hydra.cc/) and OmegaConf. The main file is
`configs/config.yaml`; its `defaults` list chooses one file from each config
group. A command-line selection such as `model=gidd_hf` replaces the selected
model file, while `model.hidden_size=1024` overrides one value after
composition.

Print the fully composed config before launching an expensive job:

```bash
PYTHONPATH=src python -m discrete_diffusion \
  data=nemotron-cc-pretok model=gidd_hf algo=gidd optim=scion \
  --cfg job --resolve
```

## Main configuration

`configs/config.yaml` defines settings shared by training and evaluation.
`configs/config_ar_perf.yaml` is a specialized autoregressive performance
profile; use it only when reproducing that benchmark.

| Section | Meaning |
| --- | --- |
| `defaults` | Files selected from each Hydra config group. Later CLI overrides win. |
| `seed` | Global reproducibility seed. |
| `scratch_dir` | Dataset/model cache root. Defaults to `DISCRETE_DIFFUSION_SCRATCH_DIR`, then `~/.cache/discrete_diffusion`. |
| `block_size` | Effective token sequence length; normally follows `model.length`. |
| `neg_infinity_mode` | Whether masked logits use a large finite number or true negative infinity. |
| `loader` | Global/evaluation batch sizes, per-machine batches, workers, pinned memory, and multiprocessing mode. |
| `training` | EMA, time sampling, loss precision, compilation, auxiliary metrics, fault-tolerant data order, and final validation. |
| `perf` | Per-step timing and model-FLOP-utilization measurement. Set the GPU peak explicitly for meaningful MFU. |
| `parallel.pipeline` | Experimental pipeline mode, schedule, microbatches, and pipeline checkpoint cadence. |
| `eval` | Checkpoint evaluation, EMA use, sample generation, and result paths. |
| `trainer` | Lightning accelerator, nodes/devices, precision, gradient accumulation, clipping, duration, and validation cadence. |
| `wandb` | W&B project, run identity, grouping, notes, tags, and local save directory. |
| `hydra` | Output directory and working-directory behavior for each run. |
| `checkpointing` | Checkpoint root and resume behavior. |

### Batch-size calculation

`loader.global_batch_size` is the target optimizer-step batch across all nodes
and devices. `loader.batch_size` is the batch loaded per machine in this
codebase. Gradient accumulation is derived as:

```text
global batch / (devices per node × per-machine batch × nodes)
```

The interpolation uses ceiling division. Choose values that divide exactly
when exact token-budget accounting matters, and inspect the resolved config.

## Config groups

### Algorithms: `configs/algo/`

| Choice | Purpose |
| --- | --- |
| `ar` | Autoregressive next-token training. |
| `mdlm` | Masked diffusion language modeling. |
| `udlm` | Uniform discrete diffusion with configurable forward process. |
| `bd3lm` | Block diffusion language modeling. |
| `flexmdm-anyorder` | Any-order FlexMDM training. |
| `gidd` | Generalized interpolating discrete diffusion, including EasyDeL-compatible losses. |
| `sedd` | Score-entropy discrete diffusion. |
| `partition-mdlm` | Partition-based masked diffusion. |
| `candi` | Hybrid continuous/discrete CANDI training. |

Common keys are `_target_` (the Python class Hydra constructs), `name`,
`parameterization`, `time_conditioning`, `T`, and `causal_attention`.
Algorithm-specific files add their own forward-process, loss, and sampler
settings.

Important `gidd` settings:

| Key | Meaning |
| --- | --- |
| `loss_type` | Selects the native, constant-pi, EasyDeL, or low-memory EasyDeL loss. |
| `p_uniform` | Mixture weight for uniform corruption. |
| `t_eps` | Numerical lower bound for sampled diffusion time. |
| `low_discrepancy_sampling` | Uses stratified/low-discrepancy time samples. |
| `time_sampling_scope` | `local` creates a grid per rank; `global_batch` creates and shards one global grid. |
| `loss_weighting` | Dynamic, clipped, or unweighted objective weighting. |
| `min_loss_weight`, `max_loss_weight` | Bounds used by clipped/dynamic weighting. |
| `min_log_snr`, `max_log_snr` | EasyDeL log-SNR range. |
| `hybrid_mixing_scale`, `hybrid_mixing_shift` | Affine controls for hybrid mixing. |
| `prior_distribution` | Masked or uniform GIDD prior. |
| `easydel_lowmem_chunk_size` | Vocabulary/token chunk size for the low-memory loss. |
| `easydel_lowmem_checkpoint_chunks` | Trades compute for activation memory within those chunks. |

### Data: `configs/data/`

| Choices | Purpose |
| --- | --- |
| `openwebtext`, `openwebtext-split`, `openwebtext-streaming` | OpenWebText loading variants. |
| `lm1b`, `lm1b-gpt2`, `lm1b-streaming`, `lm1b-wrap` | One Billion Word variants. |
| `text8`, `text8-crop` | Text8 standard and cropped sequences. |
| `wikitext2`, `wikitext103`, `ptb`, `lambada` | Common language-model evaluation/training corpora. |
| `fineweb-edu`, `slim_pajama`, `tinystories` | Larger or curated web/story corpora. |
| `scientific_papers_arxiv`, `scientific_papers_pubmed` | Scientific paper subsets. |
| `ag_news` | AG News text data. |
| `nemotron-cc-pretok` | Locally pretokenized Nemotron-CC data. Set `NEMOTRON_PRETOK_DIR`. |
| `synthetic`, `synthetic-ar`, `synthetic-gidd` | Generated data for smoke tests and performance checks. |

Typical keys are `train`, `valid`, tokenizer name/path, `cache_dir`,
`streaming`, sequence wrapping/chunking, EOS/special-token insertion, and
minimum lengths. Dataset-specific files may define local paths, shard counts,
or text/token column names. Never commit credentials; authenticate to gated
datasets through the environment or the Hugging Face CLI.

### Models: `configs/model/`

| Choice | Purpose |
| --- | --- |
| `tiny`, `small` | Standard DiT sizes for development and baseline runs. |
| `small-encoder-decoder` | Encoder-decoder baseline. |
| `block_dit` | Block diffusion transformer. |
| `flexmdm_small`, `flexmdm_anyorder` | FlexMDM architectures. |
| `small_candi`, `tiny_candi` | CANDI model sizes. |
| `small_gidd` | Original small GIDD architecture. |
| `gidd_hf`, `gidd_hf_1b`, `gidd_hf_8b` | Hugging Face-style GIDD models at increasing scale. |
| `hf_gpt2` | Hugging Face GPT-2 wrapper. |

Common architecture keys include `_target_`, `hidden_size`,
`intermediate_size`, `n_blocks`, `n_heads`, `length`, `dropout`, and embedding
options. GIDD-HF files additionally control RMSNorm, RoPE, QK normalization,
attention/MLP biases, initialization and residual scales, attention backend,
dtype, activation checkpointing, FSDP dtype compatibility, and tied output
embeddings. `gidd_hf_8b` is approximately 7.9B parameters with the configured
131,072-token vocabulary and untied input/output embeddings.

### Optimizers: `configs/optim/`

| Choice | Purpose |
| --- | --- |
| `adamw` | Standard AdamW. |
| `adamw_grouped` | AdamW with model-aware parameter groups. |
| `adamw_mup` | AdamW with fully decoupled weight decay and muP / CompleteP LR and epsilon scaling (GIDD-HF). |
| `scion` | SCION/Frank-Wolfe optimizer with geometry-specific boundaries. |

For `adamw_mup` (pair it with `model=gidd_hf_std`), `model.parametrization` selects
`standard`, `mup` or `completep`, with `model.base_width` / `model.base_depth` as the reference
model the LR was tuned on. In this mode the GIDD architecture is unchanged but its legacy scale
knobs (`init_scale`, `weight_scaling`, `head_scale`, `activation_scale`) are neutral and the
init comes from the parametrization: hidden matrices `init_std_mult / sqrt(fan_in)`, embeddings
`aux_init_scale`, zero head. `lr` is the peak LR of the width-`base_width` model; hidden and
output groups use `lr * base_width / hidden_size`, embeddings, norm gains and biases use `lr`,
and Adam's epsilon is scaled the same way (CompleteP additionally multiplies the in-block
epsilon by `base_depth / n_blocks`). The residual branch multiplier is a 1/L form:
`resid_scale / n_blocks` for `standard`/`mup` (`resid_scale` is 1.0 in `gidd_hf_std`, 4.0 in the original `gidd_hf`) and
`resid_mult * base_depth / n_blocks` for `completep`. `weight_decay` is a per-step shrink fraction applied as
`sched(t) * weight_decay` to every weight, independent of `lr`, width and the muP scale (equal
to `peak_lr * conventional_wd`). Scaling width requires `intermediate_size` and `n_heads` to grow
with `hidden_size`; untied embeddings and `zero_head_init=true` are enforced. `gidd_hf_std`
defaults to fp32 master weights so the shrink is not rounded away (add
`training.forward_autocast_dtype=bf16` for bf16 compute). Leaving `model.parametrization` unset
keeps the historical init and works with the other optimizers.

`model=gidd_hf_ref` (extends `gidd_hf_std`) additionally matches the bst-scaling dense LM
architecture with Peri-LN ("sandwich") norm placement: `x = x + r * Norm(Module(Norm(x)))` in
both sub-layers, final norm before `lm_head`, optional `peri_ln_embed`; no attention soft-cap, no
biases, GELU MLP with `intermediate_size = 4 * hidden_size`, per-head gain-free QK-norm, RMSNorm
gains initialised to 1 and `eps = finfo(dtype).eps`. All of these are off-by-default options of
the GIDD model (`peri_ln`, `attn_soft_cap=0`, `mlp_activation`, `qk_norm_per_head`,
`norm_zero_centered`, `rms_norm_eps=null`).

For SCION, `lr` is the Frank-Wolfe step size; `momentum` is the new-gradient
weight. `equal_group_lr` applies that step size to every group, while
`boundary_init` enables the Section B.3 boundary initialization. The
`scale_*`, `bias_norm`, `norm_layer_norm`, and `spectral_norm_steps` values
define group geometry. `warmup_iters`, `warmdown_iters`, and `min_lr` define
the schedule. `trace_*` options enable diagnostic smoothness/noise estimates;
they can be expensive and should normally be off.

### Strategies: `configs/strategy/`

| Choice | Purpose |
| --- | --- |
| `single-device` | One device, primarily for debugging/evaluation. |
| `ddp` | Lightning distributed data parallelism. |
| `fsdp` | Full-shard GIDD strategy with transformer-block wrapping. |

The FSDP configuration uses `FULL_SHARD`, distributing parameters, gradients,
and optimizer state. Verify optimizer compatibility before using it; some
geometry-aware optimizers require unflattened two-dimensional parameters.

### Noise schedules: `configs/noise/`

`cosine`, `geometric`, `hybrid`, `inverse-cdf`, `linear`, and `log-linear`
select the time-to-noise mapping. `_target_` chooses the implementation;
remaining parameters control schedule shape and numerical limits.

### Forward processes: `configs/forward_process/`

`absorbing`, `block_absorbing`, `uniform`, and `candi_hybrid` define how clean
tokens are corrupted. These are usually selected by an algorithm config and
must be compatible with its parameterization and prior.

### Sampling: `configs/sampling/`

| Choice | Purpose |
| --- | --- |
| `default` | Standard diffusion sampling defaults. |
| `ar` | Autoregressive decoding. |
| `udlm` | Uniform-diffusion decoding. |
| `candi` | CANDI hybrid sampling. |
| `eb` | Entropy-bounded unmasking. |

Common settings include sampler/predictor selection, denoising `steps`,
temperature, nucleus probability, BOS injection, number of sample batches,
and how many samples are logged.

### Learning-rate schedulers: `configs/lr_scheduler/`

`constant`, `constant_warmup`, `cosine_decay_warmup`, and `step_scheduler`
control optimizer learning-rate evolution. Warmup/decay lengths must be
consistent with `trainer.max_steps` or the optimizer's configured iteration
count.

### Callbacks: `configs/callbacks/`

| Choice | Purpose |
| --- | --- |
| `checkpoint_every_n_steps` | Periodic recovery checkpoints. |
| `checkpoint_monitor` | Best-checkpoint tracking for a monitored metric. |
| `learning_rate_monitor` | Logs optimizer learning rates. |
| `pytorch_profiler` | Optional PyTorch profiling. |
| `sample_saver` | Writes generated validation samples. |
| `training_latency` | Measures step/data/compute timing. |

Callbacks in the main defaults are composed as a list. Disable or override
expensive callbacks for throughput measurements.

### Evaluation: `configs/eval/`

`generate_samples` configures checkpoint sampling. `gen_ppl` configures
generative perplexity evaluation. These files are entry-task configurations,
distinct from the `eval:` section of the main training config.

### Priors: `configs/prior/`

`none` disables an explicit prior. Some algorithms impose constraints on the
prior; for example, autoregressive mode expects `none`.

## Interpolation and environment variables

- `${section.key}` references another config value.
- `${oc.env:NAME,default}` reads an environment variable with a fallback.
- `${div_up:...}`, `${mul:...}`, `${device_count:}`, and `${cwd:}` are custom
  resolvers registered by the application.
- Prefix a new ad-hoc key with `+`; use `++` when a key may or may not exist.

Useful environment variables include:

| Variable | Use |
| --- | --- |
| `DISCRETE_DIFFUSION_SCRATCH_DIR` | Dataset/model cache root. |
| `NEMOTRON_PRETOK_DIR` | Local pretokenized Nemotron dataset. |
| `HF_HOME` | Hugging Face cache location. |
| `WANDB_API_KEY` | W&B authentication; set in the shell, never in a script. |
| `WANDB_ENTITY`, `WANDB_PROJECT` | Default tracking destination for analysis launchers. |
| `HYDRA_FULL_ERROR=1` | Full exception traces while debugging config failures. |

## Preflight checklist

Before submitting a large run:

1. Print and save the resolved Hydra config.
2. Confirm dataset/cache/output paths are on the intended filesystem.
3. Confirm global batch, per-machine batch, devices, nodes, and accumulation.
4. Confirm model length and token budget imply the intended number of steps.
5. Start with `trainer.max_steps=2` and checkpoint/sample generation disabled.
6. Confirm the SLURM account, EDF environment, W&B project, and resume path.
