# GR_REC_OfficialAntiCopy_Mixed_v5

Independent all-domain sibling of the original Video-only V5. It trains the
1545 Probe4-excluded Think business groups from the original V3 source, with
no positive-signal filtering.

Topology is `1 x G4 CoT + 4 x independent Official G8`. Every Official branch
starts after the fixed target-domain prefix and samples exactly ABC3. The four
G8 groups are normalized independently; G32 normalization is forbidden.

Video reward exactly retains original V5 behavior. For ad/prod/living,
non-copy rewards remain `0/0.5/2/8`; complete same-domain history copies use
`0/0.25/1/6` for fresh rollouts 0-49 and `0/0/1/6` thereafter. The reward
stage is frozen when the fresh rollout is created and reused by both policy
iterations. Loss remains `L_cot + L_sid` with a three-token Official SID mask.

Fixed Probe4 remains production Official Beam32 across all four domains. Probe
scores use the original production hierarchy and never apply training copy
discounts; copy/non-copy anatomy is recorded as diagnostics only.

This implementation stage is CPU-only. It does not run preflight or formal
training.
