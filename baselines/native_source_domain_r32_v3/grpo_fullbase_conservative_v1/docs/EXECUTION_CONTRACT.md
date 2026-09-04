# Execution Contract

## Allowed scope

- Two independent 5-step, four-GPU deterministic smokes.
- One gated 20-step, four-GPU conservative pilot at learning rate `5e-7`.
- Fixed retention probes at smoke steps 0/5 and pilot steps 0/5/10/20.

## Fail-closed gates

Execution stops on a parent, config, dataset, code, lineage, optimizer-parameter, trainable-parameter, non-finite, checkpoint, base-fingerprint, or base-file SHA mismatch. The pilot additionally requires exact A/B smoke evidence and exact final LoRA SHA.

## Evidence

Each rank records rollout group IDs, prompt and completion token fingerprints, rewards, advantages, policy loss, learning rate, per-LoRA gradient norm/fingerprint, global gradient norm, optimizer fingerprint, RNG fingerprints, and sampled adapter fingerprints. Large generated text and model/data bytes are not published.

The canonical full-SFT file is hashed before each run and again after each run. Five selected in-memory base tensors are fingerprinted before and after optimization. The LoRA sampled and full fingerprints must change.
