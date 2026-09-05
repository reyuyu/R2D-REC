# GRPO-3 User formal run

This launcher runs the historical 200-prompt User hybrid GRPO recipe from the
user-selected GRPO-2 checkpoint 250. It uses the byte-exact deterministic
runtime already validated by the two independent five-step smokes.

The parent is a cumulative continued adapter, so inference uses exactly the
Rec FDR V4.3 full SFT model plus one GRPO-3 checkpoint. Do not stack the GRPO-1
or GRPO-2 adapter again.

The launcher records the frozen six-sample User trend proxy before training
and evaluates the same proxy over all formal checkpoints after training. That
proxy is not the private official aggregate evaluator. Official Overall,
Material, User, Recommendation, and World scores must be added from the same
external evaluation service after the adapter checkpoints are available.

Checkpoints are written outside `/data`, contain adapter weights only for model
parameters, and additionally contain optimizer, constant-scheduler, trainer,
four-rank RNG, lineage, and file-manifest state for continuation.
