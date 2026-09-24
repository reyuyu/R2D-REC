# Contributing to R2D-REC

Start with the [project documentation](../docs/README.md) and the
[method-to-code map](../docs/r2d-rec/METHODS.md). Open project issues and pull
requests in [this repository](https://github.com/reyuyu/R2D-REC).

## Scope and evidence

- Keep changes focused on the selected training or documentation route.
- For training changes, identify the dataset registry, split and digest, parent
  checkpoint, configuration and runtime. State whether the objective, sampler,
  adapter lineage or deterministic behavior changes.
- Preserve frozen reproduction sources and manifests. A new experiment should
  use a separate configuration and output location.
- Separate implementation checks, diagnostic probes and external evaluation.
  Do not present an unrun proposal or a small probe as a verified improvement.
- Public project summaries show the final selected model result. Historical
  evidence remains in the experiment records.

## Validation

Run checks appropriate to the changed behavior. Documentation changes need
valid links and traceable claims; they do not require launching training.
For training code, include the relevant regression or deterministic replay
results and their environment. Record checks actually run in the pull request.

Accelerator CI requires repository-owned runners with the labels configured in
the CUDA/NPU workflows. Set the repository variable ENABLE_ACCELERATOR_CI=true
only after provisioning those runners. CPU checks and documentation builds do
not depend on that variable.

## Files and attribution

Do not commit credentials, tokens, private connection details, raw datasets,
model weights, caches or temporary archives. Provide external artifact paths
through runtime configuration.

Retain the [Apache-2.0 license](../LICENSE) and necessary upstream copyright
notices. The base framework is LLaMA-Factory; its
[documentation and citation](../docs/upstream/LLAMA_FACTORY_ZH.md) are kept
separately from the R2D-REC project presentation.
