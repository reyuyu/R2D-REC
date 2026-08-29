# GR_REC_VideoOfficialAntiCopy_v5

Video-only GRPO ablation derived from the verified V3 runtime. It uses the
original V3 route-paired source dataset, keeps the fixed all-domain Probe4
exclusion, selects Think/Video rows, and reserves 12 additional Video groups
for a fixed Official Beam32 probe.

Training topology: `1 x G4 CoT + 4 x independent Official G8`.

Each Official continuation starts after a fixed `<|video_begin|>` and samples
exactly ABC3. A complete history copy receives zero SID reward unless it is an
Exact gold SID; copied Exact remains SID reward 8. Every complete copy,
including copied Exact, contributes zero to CoT reward.

The formal launcher is included for the next reviewed stage. This commit does
not run GPU preflight or formal training.
