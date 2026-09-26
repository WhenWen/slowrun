# Professor Forcing experiment

Base: qlabs-eng/slowrun commit `c6f0fa1b415994e0c5cf163990fe253fe1622883`.
Target: root limited-compute track, one 8×H100 node, at most one hour.
The upstream reported record is 3.183 validation loss. No improvement is claimed yet.

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
of a bidirectional GRU, and sparse updates on short conditional rollouts. Start
at step 192, update every 8 steps, use up to 4 available prefixes/rank of length 64, roll out 16
tokens, and use generator weight 0.02 on active steps (not rescaled by frequency).
The discriminator gets only the 16 continuation states, with dropout and stochastic
depth off in both domains. It trains at accuracy <=99%; the generator trains only
above 75%, following the paper. Accuracy and discriminator gradients are synchronized
across ranks. PF uses a separate sampling RNG and preserves the base initialization RNG.
Its gradient is accumulated before CE/MTP microbatches, so it participates in the
existing first-order meta-gradient adaptation.

The paper reports character-level PTB improvement, but no word-level PTB improvement,
and approximately 3× training cost in its character experiment. Gains here are an
empirical question; short rollouts may be too weak, or overhead may erase the gain.

## Validation and experiment plan

1. Unit tests check genuinely autoregressive feedback, causality of the actual GPT,
   sampling/replay equivalence, gradient isolation, gating, and RNG isolation.
2. Full-size eight-GPU smoke, changing only the stopping step to 224. Preserve the
   11-epoch schedule, model, batch, data, optimizer, and evaluation settings.
3. Matched upstream baseline and PF runs on the same hardware and seed. Compare
   validation loss, discriminator accuracy/activation, step time and total wall time.
4. If promising, adjust frequency/rollout/weight and repeat a second seed. Treat
   every exploratory run separately; do not hide extra work inside a submitted run.

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

`PYTHONPATH=. python tests/verify_compiled_pf.py --backend fa2` runs a separate
full-model CUDA diagnostic on one GPU. It forces the generator gate open and
checks two compiled BF16 replay backward passes for finite trunk gradients and
gradient isolation. It does not run the training optimizer or establish memory
headroom alongside the training graph. The normal 224-step smoke retains the
paper's accuracy gate and is still required.
Add `--activation-checkpointing` to the diagnostic to exercise the Tiger setting.

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
