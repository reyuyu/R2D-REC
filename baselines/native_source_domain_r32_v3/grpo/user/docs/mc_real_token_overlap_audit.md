# MC_USER_v1 Real-tokenizer Credit-unit Overlap Audit

Conclusion: **TOKENIZER_BOUNDARY_OVERLAP_CONFIRMED**

## Fixed real-smoke reproduction

### Action

- candidate 0: canonical==generated=True, projection_required=False, overlap_tokens=0
- candidate 1: canonical==generated=True, projection_required=False, overlap_tokens=0

### Chain

- candidate 0: canonical==generated=True, projection_required=False, overlap_tokens=0
- candidate 1: canonical==generated=True, projection_required=False, overlap_tokens=4
  - token 76 id=36509 decoded='"},{"' offset=[192, 197]
    - unit 0 event 0: delta=0.138888888889, chars=[55,194)
    - unit 1 event 1: delta=0.138888888889, chars=[195,341)

```text
/长播] <|video_begin|><s_a_1922><s_b_120><s_c_2527>","logic":"兴趣触发：开始关注秋冬服装，初步探索方向为材质（羊毛）和功能性（保暖、防起球）。"},{"date":"2026-01-27","action":"[视频-收藏/长播] <|video_begin|><s_a_6521><s_b_2852><s_c_5932>","logic":
```

| index | id | decoded | offset | text |
|---:|---:|---|---|---|
| 72 | 99287 | `防` | [187, 188] | `防` |
| 73 | 71618 | `起` | [188, 189] | `起` |
| 74 | 77959 | `球` | [189, 190] | `球` |
| 75 | 74276 | `）。` | [190, 192] | `）。` |
| 76 | 36509 | `"},{"` | [192, 197] | `"},{"` |
| 77 | 1028 | `date` | [197, 201] | `date` |
| 78 | 3252 | `":"` | [201, 204] | `":"` |
| 79 | 17 | `2` | [204, 205] | `2` |
| 80 | 15 | `0` | [205, 206] | `0` |
  - token 137 id=36509 decoded='"},{"' offset=[339, 344]
    - unit 1 event 1: delta=0.138888888889, chars=[195,341)
    - unit 2 event 2: delta=0.138888888889, chars=[342,494)
  - token 205 id=36509 decoded='"},{"' offset=[492, 497]
    - unit 2 event 2: delta=0.138888888889, chars=[342,494)
    - unit 3 event 3: delta=0.138888888889, chars=[495,648)
  - token 269 id=36509 decoded='"},{"' offset=[646, 651]
    - unit 3 event 3: delta=0.138888888889, chars=[495,648)
    - unit 4 event 4: delta=-0.111111111111, chars=[649,762)

## Full train_3000 raw-gold audit

| route | candidates with overlap / total | overlap tokens | unit pairs | max units/token |
|---|---:|---:|---:|---:|
| action | 0 / 1500 | 0 | 0 | 0 |
| chain | 0 / 1500 | 0 | 0 | 0 |

## Unchanged objective reproduction

- action: PASS - no overlap error
- chain: ERROR - non-zero credit units overlap at generated indices [76]

## Fixed mixed-sign check

- action: same-sign=0, mixed-sign=0
- chain: same-sign=3, mixed-sign=1
