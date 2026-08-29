# GR_REC_ThinkExactSharpen_v4

Code-only V4 contract derived from `gr_rec_think_sample8_fullsid_v3`. The fixed
611-group positive-signal Think dataset trains a global G4 CoT branch plus four
independent Free G8 and four independent Official G8 SID branches.

This directory intentionally contains no GPU preflight artifact. Review and an
explicit launch decision are required before running `launch_exact_sharpen_train.sh`.

Key immutable inputs are enforced by the runner: dataset SHA256, row/group count,
Think-only routing, Probe4 exclusion, V3 checkpoint-1500 path, and adapter SHA256.
