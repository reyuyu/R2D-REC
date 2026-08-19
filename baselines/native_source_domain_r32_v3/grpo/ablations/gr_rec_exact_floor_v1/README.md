# GR_REC_ExactFloor_Ablation_v1

Independent advantage-only ablation based on `GR_REC_v1` and the original BATA
adapter. The only training-math change is:

```text
baseline_g = min(mean(group_rewards), 8.0)
advantage_i = (reward_i - baseline_g) / 8.0
```

The formal entry point is `run_exact_floor_train.py`. Do not launch it until a
separate GPU validation is authorized. CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' python3 ablations/gr_rec_exact_floor_v1/test_exact_floor.py
CUDA_VISIBLE_DEVICES='' python3 ablations/gr_rec_exact_floor_v1/audit_historical_rewards.py
```
