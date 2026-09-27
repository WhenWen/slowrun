# Professor Forcing experiment

Base: qlabs-eng/slowrun commit `c6f0fa1b415994e0c5cf163990fe253fe1622883`.
Target: root limited-compute track, one 8×H100 node, at most one hour.
The upstream reported record is 3.183 validation loss. No improvement is claimed yet.
The full-context, amortized implementation is undergoing runtime validation.
The earlier 64+16-token standalone PF design is retired: it did not exercise the
model's 1024/2048 attention windows or reuse ordinary training computation.

## Paper and adaptation

Read the original LaTeX source of [Lamb et al. (2016)](https://arxiv.org/abs/1610.09038),
`paper.tex`, sections “Proposed Approach: Professor Forcing” and “Experiments”.
For a discriminator D whose positive class is teacher forcing, the objectives are:

* D: `mean(softplus(-D(real_states))) + mean(softplus(D(free_states)))`.
* Generator: original CE/MTP + `lambda * mean(softplus(-D(free_states)))`.

This is the paper's `NLL + C_f` variant, without its optional `C_t` term. Samples
are discrete, detached draws from the model's full softmax, with temperature 1.
The generator gradient flows through a replay of the sampled contexts, not
through token choices. A causal transformer allows that replay to run in parallel.
This is not scheduled sampling: no real-token CE targets are used on generated contexts.

Changes relative to the paper: use a transformer's final normalized states instead
of GRU pre-tanh activations, a small temporal convolution discriminator instead
of a bidirectional GRU, and sparse conditional rollouts. Each PF branch preserves
the full 2048-token context: `prefix = 2048 - rollout`. The first comparison uses
1920 prompt tokens and 128 generated tokens. A 32-token rollout remains an ablation.
Start at step 192, update every 8 steps, use up to 4 available examples/rank, and
use generator weight 0.02 on active steps (not rescaled by frequency).

The first ordinary CE/MTP microbatch supplies real continuation states, prefix
KV, and the first sampling logits. Incremental sampling detaches that KV cache;
a parallel differentiable replay of only the generated suffix uses the ORIGINAL,
attached KV. This preserves the PF gradient through the shared prefix. The joint
CE/PF backward traverses that prefix once and participates in the existing
first-half meta-gradient adaptation. Cache entries correspond to layer execution
visits, including each repeated decoder pass, rather than just physical layers.
The cached decoder uses symbolic lengths to avoid recompiling once per token.

The discriminator sees only continuation states. Teacher and free branches both
use the ordinary CE training mode, including dropout; using train-mode teacher
states against eval-mode free states would introduce a domain cue. Replay draws
fresh suffix dropout noise, so this is a stochastic replay adaptation, not a
claim to reproduce the sampling pass's exact dropout realization. PF isolates its
RNG use from subsequent CE microbatches. The benchmark's stochastic depth is zero;
cached training rejects nonzero stochastic depth until shared masks are supported.
The discriminator trains at accuracy <=99%; the generator trains only above 75%.
Accuracy and discriminator gradients are synchronized across ranks. Discriminator
initialization and token sampling use isolated RNGs.

The paper reports character-level PTB improvement, but no word-level PTB improvement,
and approximately 3× training cost in its character experiment. Gains here are an
empirical question; short rollouts may be too weak, or overhead may erase the gain.

## Validation and experiment plan

1. Unit tests check actual-GPT cached/full output and gradient equivalence, including
   IHA, sliding windows, repeated layers and checkpointing; closed-gate BF16 dropout
   gradients match ordinary CE, and the shared implementation avoids teacher replay.
2. GPU diagnostics check rectangular attention, compiled dynamic-cache reuse and
   joint gradients. Profile rollout lengths and full-model memory before choosing
   the recipe; these diagnostics do not establish optimizer memory or quality.
3. Full-size eight-GPU smoke, changing only the stopping step to 224. Preserve the
   11-epoch schedule, model, batch, data, optimizer, and evaluation settings.
4. Matched upstream baseline and PF runs on the same hardware and seed. Compare
   validation loss, discriminator accuracy/activation, step time and total wall time.
5. If promising, adjust frequency/rollout/weight and repeat a second seed. Treat
   every exploratory run separately; do not hide extra work inside a submitted run.

Full-model, batch-one A5000 diagnostics passed two compiled BF16 joint backward
passes at both rollout lengths with checkpointing. Warm CE+PF forward/backward
took 1.256s for rollout32 and 3.411s for rollout128; PF forward components were
0.723s and 2.869s respectively. Both peaked at 17,445.5MiB allocated. These
forced-gate, single-microbatch measurements exclude the optimizer and do not
establish eight-GPU overhead or H100 budget compliance. The longer rollout was
selected to test a longer free-running horizon, with its cost explicitly measured.

The script logs both upstream-style training time (which omits warmup steps) and
full script wall time including compilation and evaluation. The accepted
[current record, PR98](https://github.com/qlabs-eng/slowrun/pull/98), explicitly
reports training time: its submitted seed43
[log](https://gist.github.com/dangxingyu/c1c6754312714772b2927b8c6ca0c874)
(`sub-1h-minimal-v2-s43-rerun.train.log`) records 59.28 minutes of training and
71.10 minutes of total wall time. Compare using this demonstrated convention,
while always disclosing both numbers and any changes to warmup accounting.
Our first PF update at step192 currently includes PF compilation in the training
timer; the isolated replay diagnostic does not remove that cost from a run.
`pf_forward_seconds_including_compile` measures sampling, discriminator work and
suffix replay forward, not the inseparable joint CE/PF backward. Use complete
step and run timers to judge total PF overhead.
The two-hour SLURM allocation allows diagnosis/measurement; it does not expand
the one-hour training budget. Smoke results are explicitly labeled and are not
benchmark results.

## Marlowe

Checkout: `/scratch/m000123-pm06/kaiyuew/slowrun-pf`.
Use a dedicated `.venv` with `requirements.txt`. Data comes from unmodified
`prepare_data.py`; verify its exact upstream hashes before every job. Prewarm
`TIKTOKEN_CACHE_DIR=$PWD/.cache/tiktoken` before submitting.

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
bash experiments/professor_forcing/submit.sh smoke
bash experiments/professor_forcing/submit.sh baseline
bash experiments/professor_forcing/submit.sh pf
# Optional sequential comparison on one physical node, only after smoke succeeds:
bash experiments/professor_forcing/submit.sh pair <smoke-job-id>
```

The runner selects the `preempt` partition, normal QoS, and the user's normal
project account. It submits exactly one node and eight GPUs. No automatic retry
loop is enabled during bring-up. A killed/requeued attempt is not a completed
comparison; start a fresh named attempt after diagnosis. Keep an interrupted
attempt's logs and account for any suspension when judging wall-clock eligibility.
Pair mode reserves three hours for two sequential, separately timed runs. It uses
an `afterok` dependency on the user's smoke job and cancels if that dependency
fails. Baseline failure stops the pair before PF starts. Both runs have separate
result/checkpoint directories; sharing a node and compile cache is recorded when
comparing wall time. The per-run training budget remains one hour.
The smoke allocation first runs separate Hopper attention and full-model PF
gradient preflights (per-rank batch4, generator gate forced open), then launches
the normal eight-rank smoke with its natural accuracy gate. Preflight timings
are not training results. Comparison jobs use their own job-specific compile cache.

## Tiger6 development

Tiger6 has RTX A5000 24GB GPUs (Ampere), which cannot run the Hopper FA3 kernel.
`--attention-backend fa2` explicitly selects the pinned major-version-3 Hugging Face
FlashAttention-2 kernel; the default Hopper path remains FA3 version 1.
`run_tiger.sh` uses the full root model and total batch with a device batch of 1,
adjusting accumulation through the existing trainer. It enables non-reentrant
activation checkpointing for transformer and MTP blocks, preserving dropout RNG.
The uncheckpointed four-step test passed but the longer run exhausted GPU memory
after step38, before PF activated; short-run success did not establish sustained
memory headroom. Checkpointing trades recomputation for memory and is disabled
by default on the H100 path. GPU memory must be checked
on the actual run. It accepts 1/2/4/8 GPUs via `TIGER_GPU_IDS` and records GPU type
and world size in results. Start with `TIGER_MAX_STEPS=4` for memory/optimizer
bring-up, then extend the same configuration to 224 steps to reach PF updates.
Neither prefix is a completed performance comparison.
With Tiger's device batch of 1, PF receives one prefix per rank; the H100 device
batch supplies four. Account for this difference when interpreting exploratory
Tiger results.

`PYTHONPATH=. python tests/verify_cached_pf.py --backend fa2 --rollout 32
--batch-size 1 --activation-checkpointing` runs a small-model/full-context CUDA diagnostic.
Add `--full-model` for the actual 1.44B model. It forces the generator gate open,
checks two compiled BF16 joint CE/PF backward passes, and fails if cached decoding
keeps compiling new graphs as its context grows. It does not run the optimizer or
establish distributed training memory. Compare rollout 32 and 128 separately.
The normal 224-step smoke retains the paper's accuracy gate and is still required.
`verify_compiled_pf.py` remains a legacy standalone replay diagnostic only.

Run as an `srun` step within an existing authorized allocation. The script uses a
lock to prevent overlapping experiments from the same checkout. Do not cancel the
holder allocation when stopping an experiment. The model, data, and 11-epoch
schedule stay the same; Ampere timings are exploratory measurements and cannot
substitute for H100 timings in the final limited-track comparison.

Tiger6 communication preflight: native NCCL succeeded on the adjacent GPU8/9 pair,
but eight-GPU collectives timed out. Disabling SHM alone also timed out. The
eight-GPU sum/barrier check initially passed with both P2P and SHM disabled,
using socket transport. A later eight-rank test passed with P2P disabled and
SHM enabled, including verified 256MiB all-reduce, reduce-scatter, and all-gather
buffers. The Tiger runner now sets `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`,
and `NCCL_NET=Socket`, allowing the faster host shared-memory path. These
settings apply only to Tiger; collective timings alone are not training speedups.
The original stalled training step was stopped before model creation; the user's
holder allocation was preserved. GPU attention parity tests passed on A5000 for
sequence lengths80/2048 and three causal window settings, with approximately0.3%
gradient relative RMS error versus an FP32 reference.
