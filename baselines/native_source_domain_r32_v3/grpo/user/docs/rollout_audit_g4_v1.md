# GR_USER_v1 G=4 rollout-only audit

Run: `GR-USER-G4-AUDIT-4GPU-20260819-181843`

## Configuration

- Parent: Beta baseline Epoch 2 checkpoint-1106
- GPU IDs: [0, 1, 2, 3]
- G: 4
- Sampling: temperature=0.9, top_p=0.95, seed=20260819
- max_new_tokens: 512
- Training: disabled

## Group variance

| Route | Zero std | Mean group std | Median group std |
|---|---:|---:|---:|
| Action | 4.00% | 0.125720 | 0.108085 |
| Chain | 0.00% | 0.073249 | 0.068846 |

Assessment: **G4 healthy**

### Per-bucket variance

| Route | Bucket | Prompts | Zero std | Mean std | Median std |
|---|---|---:|---:|---:|---:|
| Action | 1-5 | 8 | 12.50% | 0.115844 | 0.108850 |
| Action | 11-20 | 15 | 0.00% | 0.163311 | 0.154736 |
| Action | 21-30 | 10 | 0.00% | 0.138198 | 0.143364 |
| Action | 31-40 | 5 | 0.00% | 0.094770 | 0.085057 |
| Action | 41+ | 2 | 0.00% | 0.101266 | 0.101266 |
| Action | 6-10 | 10 | 10.00% | 0.085121 | 0.062453 |
| Chain | 2 | 8 | 0.00% | 0.105927 | 0.098917 |
| Chain | 3 | 27 | 0.00% | 0.070951 | 0.061877 |
| Chain | 4 | 12 | 0.00% | 0.058714 | 0.055711 |
| Chain | 5 | 3 | 0.00% | 0.064940 | 0.054479 |


## Candidate metrics

- Action F1 mean/p50/p90: 0.625160 / 0.666667 / 0.961538
- Action precision/recall: 0.719128 / 0.637474; JSON valid: 100.00%; exact set: 6.50%
- Action hallucination/duplicate/wrong selection: 6.50% / 4.00% / 73.50%
- Chain reward mean/p50/p90: 0.457633 / 0.463499 / 0.621653
- Chain Action/Logic Alignment F1: 0.669349 / 0.245917; JSON valid: 100.00%
- Chain candidate grounding (grounded/partial/ungrounded): 56.50% / 41.00% / 2.50%
- Action penalty-mask fraction mean: 0.5136%
- Chain penalty-mask fraction mean: 2.9829%
- Action truncation: 0.00%; Chain truncation: 0.00%

### Chain constraint candidate rates

| Constraint | Rate | Violation records |
|---|---:|---:|
| `hallucinated_sid` | 1.50% | 3 |
| `date_mismatch` | 30.50% | 71 |
| `action_mismatch` | 14.00% | 31 |
| `duplicate_event` | 2.00% | 4 |
| `chronology_violation` | 3.00% | 6 |
| `excess_event` | 0.00% | 0 |

### Violation cooccurrence

- Action: `{"duplicate_sid+hallucinated_sid": 2, "duplicate_sid+wrong_selection_sid": 8, "hallucinated_sid+wrong_selection_sid": 10}`
- Chain: `{"action_mismatch+date_mismatch": 4, "action_mismatch+duplicate_event": 2, "chronology_violation+date_mismatch": 1, "date_mismatch+duplicate_event": 1, "date_mismatch+hallucinated_sid": 1}`

## Representative examples

### Action High Reward Hallucination

- `5b2e7e603229aeeae399ec52a7c0e9f0f22f4e4f7a3f1e126a4642c57bf96a49` candidate 0, reward=0.950000, violations=['hallucinated_sid']: ["<\|video_begin\|><s_a_3801><s_b_5106><s_c_8085>", "<\|video_begin\|><s_a_6140><s_b_4932><s_c_1004>", "<\|video_begin\|><s_a_3801><s_b_5106><s_c_4647>", "<\|video_begin\|><s_a_532><s_b_5349><s_c_1034>", "<\|video_begin\|><s_a_1351><s_b_7246><s_c_1939>", "<\|video_begin\|><s_a_3801><s_b_1747><s_c_2658>", "<\|video_begin\|><s_a_3801><s_b_5106><s_c_975>", "<\|video_begin\|><s_a_6140><s_b_7246><s_c_5349>", "<\|video_begin\|><s_a_3801><s_b_7246><s_c_6361>", "<\|video_begin\|><s_a_3001><s_b_3614><s_c_1076>", "<\|video_begin\|><s_a_4487><s_b_1395><s_c_5349>", "<\|video_begin\|><s_a_3001><s_b_5349><s_c_7106>", "<\|video_begi
- `5b2e7e603229aeeae399ec52a7c0e9f0f22f4e4f7a3f1e126a4642c57bf96a49` candidate 3, reward=0.904762, violations=['wrong_selection_sid', 'hallucinated_sid', 'wrong_selection_sid']: ["<\|video_begin\|><s_a_3801><s_b_5106><s_c_8085>", "<\|video_begin\|><s_a_6140><s_b_4932><s_c_1004>", "<\|video_begin\|><s_a_3801><s_b_5106><s_c_4647>", "<\|video_begin\|><s_a_532><s_b_5349><s_c_1034>", "<\|video_begin\|><s_a_1351><s_b_7246><s_c_1939>", "<\|video_begin\|><s_a_3801><s_b_1747><s_c_2658>", "<\|video_begin\|><s_a_3801><s_b_5106><s_c_975>", "<\|video_begin\|><s_a_6140><s_b_7246><s_c_5349>", "<\|video_begin\|><s_a_3801><s_b_7246><s_c_6361>", "<\|video_begin\|><s_a_3001><s_b_3614><s_c_1076>", "<\|video_begin\|><s_a_4487><s_b_1395><s_c_5349>", "<\|video_begin\|><s_a_3001><s_b_5349><s_c_7106>", "<\|video_begi
- `aa423d72352a0459479688460a1e9a19614c29477fa5db34930a3748cb1eee09` candidate 2, reward=0.742857, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'hallucinated_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|prod_begin\|><s_a_5494><s_b_2976><s_c_5004>", "<\|prod_begin\|><s_a_7794><s_b_850><s_c_3055>", "<\|prod_begin\|><s_a_4994><s_b_4538><s_c_4877>", "<\|prod_begin\|><s_a_7526><s_b_2711><s_c_6571>", "<\|living_begin\|><s_a_1229><s_b_7878><s_c_6278>", "<\|prod_begin\|><s_a_4053><s_b_5442><s_c_1012>", "<\|living_begin\|><s_a_773><s_b_6234><s_c_932>", "<\|prod_begin\|><s_a_3850><s_b_4240><s_c_5925>", "<\|prod_begin\|><s_a_6141><s_b_3827><s_c_5845>", "<\|living_begin\|><s_a_4372><s_b_8008><s_c_5036>", "<\|prod_begin\|><s_a_4994><s_b_4782><s_c_5104>", "<\|prod_begin\|><s_a_3457><s_b_4782><s_c_334>", "<\|ad_begin\|><s_a_689
- `5c2fabad619f04fe1c1d6e6a0713f9046c03cbc3646ae366e1c8ba4c036cbaf4` candidate 3, reward=0.740741, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'hallucinated_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'duplicate_sid']: ["<\|prod_begin\|><s_a_4263><s_b_5369><s_c_7588>", "<\|prod_begin\|><s_a_4556><s_b_2916><s_c_5345>", "<\|living_begin\|><s_a_901><s_b_8049><s_c_1834>", "<\|prod_begin\|><s_a_4263><s_b_2848><s_c_1541>", "<\|living_begin\|><s_a_901><s_b_8022><s_c_6557>", "<\|prod_begin\|><s_a_4727><s_b_526><s_c_7055>", "<\|prod_begin\|><s_a_4727><s_b_526><s_c_539>", "<\|prod_begin\|><s_a_5516><s_b_4106><s_c_6542>", "<\|living_begin\|><s_a_901><s_b_765><s_c_2957>", "<\|prod_begin\|><s_a_526><s_b_8087><s_c_2215>", "<\|prod_begin\|><s_a_526><s_b_1029><s_c_705>", "<\|ad_begin\|><s_a_5018><s_b_2863><s_c_1834>", "<\|prod_begin\|><s_a_526><s_b_
- `9fba78f62bde96a64ddd10849ffa3aa02b53ff91707cd0f8aeac03c5ade576e2` candidate 3, reward=0.711864, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'hallucinated_sid']: ["<\|living_begin\|><s_a_726><s_b_7200><s_c_6278>", "<\|prod_begin\|><s_a_513><s_b_1259><s_c_3297>", "<\|prod_begin\|><s_a_513><s_b_1206><s_c_619>", "<\|living_begin\|><s_a_5659><s_b_7342><s_c_3882>", "<\|prod_begin\|><s_a_1805><s_b_5399><s_c_2294>", "<\|prod_begin\|><s_a_1805><s_b_4782><s_c_7368>", "<\|prod_begin\|><s_a_1805><s_b_2003><s_c_1901>", "<\|prod_begin\|><s_a_1805><s_b_4516><s_c_6298>", "<\|living_begin\|><s_a_7074><s_b_451><s_c_4454>", "<\|living_begin\|><s_a_1659><s_b_2990><s_c_3026>", "<\|prod_begin\|><s_a_513><s_b_7953><s_c_3297>", "<\|ad_begin\|><s_a_5632><s_b_3727><s_c_3171>", "<\|prod_begin\|><s_a_513

### Action High Reward Duplicate

- `37400eceb1dc973fe5cf2112de3046aee1af34cc4339a512534ff8a43eb53baf` candidate 0, reward=0.943396, violations=['duplicate_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|ad_begin\|><s_a_7246><s_b_7246><s_c_7566>", "<\|ad_begin\|><s_a_4487><s_b_1656><s_c_1070>", "<\|ad_begin\|><s_a_3001><s_b_6047><s_c_3995>", "<\|ad_begin\|><s_a_7417><s_b_7246><s_c_7676>", "<\|ad_begin\|><s_a_7417><s_b_397><s_c_6357>", "<\|ad_begin\|><s_a_7417><s_b_7246><s_c_6583>", "<\|ad_begin\|><s_a_4684><s_b_7551><s_c_6056>", "<\|video_begin\|><s_a_4487><s_b_3148><s_c_5795>", "<\|ad_begin\|><s_a_6474><s_b_7246><s_c_7003>", "<\|video_begin\|><s_a_3001><s_b_2130><s_c_5004>", "<\|ad_begin\|><s_a_7417><s_b_7261><s_c_1792>", "<\|ad_begin\|><s_a_7417><s_b_7246><s_c_7676>", "<\|ad_begin\|><s_a_7417><s_b_7342><s_c_2325
- `832085ebc58978064455eae4f29e0b7f58adebf8ad7a9a8e3412f497e1da224f` candidate 1, reward=0.893617, violations=['wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|prod_begin\|><s_a_3201><s_b_5172><s_c_7477>", "<\|prod_begin\|><s_a_7126><s_b_6599><s_c_806>", "<\|prod_begin\|><s_a_2852><s_b_4914><s_c_4079>", "<\|prod_begin\|><s_a_1908><s_b_3335><s_c_5069>", "<\|prod_begin\|><s_a_1908><s_b_7912><s_c_3979>", "<\|prod_begin\|><s_a_647><s_b_3854><s_c_2199>", "<\|prod_begin\|><s_a_7126><s_b_2365><s_c_2852>", "<\|prod_begin\|><s_a_2852><s_b_3335><s_c_2852>", "<\|prod_begin\|><s_a_1609><s_b_2365><s_c_2852>", "<\|prod_begin\|><s_a_2774><s_b_2539><s_c_4659>", "<\|video_begin\|><s_a_5295><s_b_2539><s_c_6583>", "<\|video_begin\|><s_a_2153><s_b_2539><s_c_229>", "<\|video_begin\|><s_a_136
- `5c2fabad619f04fe1c1d6e6a0713f9046c03cbc3646ae366e1c8ba4c036cbaf4` candidate 3, reward=0.740741, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'hallucinated_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'duplicate_sid']: ["<\|prod_begin\|><s_a_4263><s_b_5369><s_c_7588>", "<\|prod_begin\|><s_a_4556><s_b_2916><s_c_5345>", "<\|living_begin\|><s_a_901><s_b_8049><s_c_1834>", "<\|prod_begin\|><s_a_4263><s_b_2848><s_c_1541>", "<\|living_begin\|><s_a_901><s_b_8022><s_c_6557>", "<\|prod_begin\|><s_a_4727><s_b_526><s_c_7055>", "<\|prod_begin\|><s_a_4727><s_b_526><s_c_539>", "<\|prod_begin\|><s_a_5516><s_b_4106><s_c_6542>", "<\|living_begin\|><s_a_901><s_b_765><s_c_2957>", "<\|prod_begin\|><s_a_526><s_b_8087><s_c_2215>", "<\|prod_begin\|><s_a_526><s_b_1029><s_c_705>", "<\|ad_begin\|><s_a_5018><s_b_2863><s_c_1834>", "<\|prod_begin\|><s_a_526><s_b_
- `9fba78f62bde96a64ddd10849ffa3aa02b53ff91707cd0f8aeac03c5ade576e2` candidate 3, reward=0.711864, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'hallucinated_sid']: ["<\|living_begin\|><s_a_726><s_b_7200><s_c_6278>", "<\|prod_begin\|><s_a_513><s_b_1259><s_c_3297>", "<\|prod_begin\|><s_a_513><s_b_1206><s_c_619>", "<\|living_begin\|><s_a_5659><s_b_7342><s_c_3882>", "<\|prod_begin\|><s_a_1805><s_b_5399><s_c_2294>", "<\|prod_begin\|><s_a_1805><s_b_4782><s_c_7368>", "<\|prod_begin\|><s_a_1805><s_b_2003><s_c_1901>", "<\|prod_begin\|><s_a_1805><s_b_4516><s_c_6298>", "<\|living_begin\|><s_a_7074><s_b_451><s_c_4454>", "<\|living_begin\|><s_a_1659><s_b_2990><s_c_3026>", "<\|prod_begin\|><s_a_513><s_b_7953><s_c_3297>", "<\|ad_begin\|><s_a_5632><s_b_3727><s_c_3171>", "<\|prod_begin\|><s_a_513
- `5c2fabad619f04fe1c1d6e6a0713f9046c03cbc3646ae366e1c8ba4c036cbaf4` candidate 0, reward=0.583333, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'duplicate_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|living_begin\|><s_a_4372><s_b_3795><s_c_5563>", "<\|prod_begin\|><s_a_4263><s_b_5369><s_c_7588>", "<\|prod_begin\|><s_a_4556><s_b_2916><s_c_5345>", "<\|living_begin\|><s_a_901><s_b_8049><s_c_1834>", "<\|prod_begin\|><s_a_4263><s_b_2848><s_c_1541>", "<\|living_begin\|><s_a_901><s_b_8022><s_c_6557>", "<\|living_begin\|><s_a_901><s_b_765><s_c_2957>", "<\|prod_begin\|><s_a_526><s_b_8087><s_c_2215>", "<\|prod_begin\|><s_a_526><s_b_1029><s_c_705>", "<\|ad_begin\|><s_a_5018><s_b_2863><s_c_1834>", "<\|prod_begin\|><s_a_526><s_b_3569><s_c_819>", "<\|prod_begin\|><s_a_1834><s_b_729><s_c_7185>", "<\|prod_begin\|><s_a_526><s_

### Chain High Reward Date Or Action Mismatch

- `9c8f9b5fccb7ba236aa8d1797ae15818fce67028c4935f11b79e6b945082abdd` candidate 0, reward=0.726604, violations=['date_mismatch', 'date_mismatch']: {"logic_chain": {"name": "从季节性鼻部不适到针对性护理方案的探索", "events": [{"date": "2025-12-24", "action": "[商品-购买] <\|prod_begin\|><s_a_2950><s_b_7749><s_c_5879>", "logic": "问题触发：首次购买针对“鼻炎”、“过敏”、“鼻塞”症状的成套护理产品，表明用户遇到了具体的鼻部健康问题。"}, {"date": "2025-12-27", "action": "[商品-购买] <\|prod_begin\|><s_a_2950><s_b_1638><s_c_6361>", "logic": "认知增量与方案升级：在购买基础护理产品后，紧接着购买了包含“苍耳子油”这一明确药用成分的产品。这表明用户可能意识到基础护理效果有限，开始寻求更具有特定功效的解决方案，认知从“缓解不适”深化至“针对特定成分的治疗”。"}]}}
- `9c8f9b5fccb7ba236aa8d1797ae15818fce67028c4935f11b79e6b945082abdd` candidate 2, reward=0.718718, violations=['date_mismatch', 'date_mismatch']: {"logic_chain": {"name": "从季节性鼻部不适到针对性护理方案的探索", "events": [{"date": "2025-12-24", "action": "[商品-购买] <\|prod_begin\|><s_a_2950><s_b_7749><s_c_5879>", "logic": "问题触发：购买针对“鼻塞、鼻痒”的特定鼻舒贴，表明用户正遭受季节性或非特异性鼻部不适，并采取了初步的、对症的解决方案。"}, {"date": "2025-12-27", "action": "[商品-购买] <\|prod_begin\|><s_a_2950><s_b_1638><s_c_6361>", "logic": "认知跃迁：在短时间内从购买单一功能产品（鼻舒贴）转变为购买成分更专业、功能更强的“舒鼻膏”（包含苍耳子油、黄芪、蜂胶等），体现了对鼻部护理的认知升级，从缓解表层症状深入到对鼻腔深层进行养护。"}]}}
- `56ddd09a41db2129518992f6e6b4dc5d5334b72196d595266391c1fc3b8c5c2f` candidate 3, reward=0.689425, violations=['date_mismatch']: {"logic_chain": {"name": "从被动接收到主动实践的DIY兴趣激发", "events": [{"date": "2026-01-30", "action": "[视频-点赞/长播] <\|video_begin\|><s_a_2585><s_b_3408><s_c_2468>", "logic": "兴趣触发：通过观看视频，首次接触到“自制生根粉”这一具体DIY项目，形成初步认知。"}, {"date": "2026-01-30", "action": "[商品-点击] <\|prod_begin\|><s_a_7433><s_b_4771><s_c_396>", "logic": "行为转化与深化：在观看视频的同一天，立即产生了将兴趣转化为实践的意图，开始主动搜索并点击与“生根粉制作”强相关的工具“电动打碎机”，体现了从认知到行动的快速收敛。"}]}}
- `61022eaae6ece8f7411832802ac34fe1ee60017c2a074fce4ca66b8550f4388b` candidate 1, reward=0.634520, violations=['action_mismatch', 'duplicate_event']: {"logic_chain": {"name": "从被动接收到主动决策的取暖设备选购", "events": [{"date": "2026-01-21", "action": "[视频-点赞/长播] <\|video_begin\|><s_a_5132><s_b_5224><s_c_2242>", "logic": "问题触发与方案认知：用户被动接收到关于“创维智能取暖器”的产品介绍视频，内容明确指向“语音操控”、“节能”、“石墨烯导热”等核心卖点，构成了后续决策的初始信息输入。"}, {"date": "2026-01-22", "action": "[商品-点击] <\|prod_begin\|><s_a_7433><s_b_5857><s_c_5218>", "logic": "决策收敛与行动准备：在接收到产品信息的次日，用户从被动观看转为主动点击具体商品，目标与初始视频内容高度一致（取暖器、语音操控、节能），表明用户已进入购买意向的评估阶段。"}, {"date": "2026-01-22", "action": "[商品-点击] <\|prod_begin\|><s_a_7433><s_b_5857><s_c_5218>", "logic": "对比与最终锁定：用户对同一商品进行了两次点击，这通常是购买决策过程中的犹豫和对比行为，进一步确认了其对“创维”品牌及“语音”功能的偏好。
- `88985fea200635a1d4df3ed541507e96a9c3a9c0bdc6ebda264b6ce9bedcbc38` candidate 0, reward=0.633024, violations=['date_mismatch']: {"logic_chain": {"name": "从被动接收到主动探索：对明朝历史的兴趣深化", "events": [{"date": "2025-12-20", "action": "[广告-点击] <\|ad_begin\|><s_a_655><s_b_6519><s_c_7404>", "logic": "初始触发：被动点击与“明朝”相关的泛娱乐化广告，表现出初步的、非特异性兴趣。"}, {"date": "2025-12-24", "action": "[广告-点击] <\|ad_begin\|><s_a_3738><s_b_1141><s_c_3026>", "logic": "兴趣聚焦：在短时间内，从泛历史题材转向对“明朝皇室家族斗争”这一特定历史事件的关注，兴趣点开始具象化。"}, {"date": "2025-12-29", "action": "[视频-收藏/点赞/长播] <\|video_begin\|><s_a_7038><s_b_7665><s_c_7925>", "logic": "认知跃迁与深度互动：从被动点击广告，演化为主动对包含“朱瞻基”（明确的明朝人物）和“宫廷斗争”情节的长视频进行高价值的“收藏/点赞”操作，表明兴趣已从浅层了解深化为对特定历史事件的深度探究。"}]}}

### Action Low Reward Valid Grounded No Duplicate

- `05de1ffe67a8a113b46fad8867ccd474267d1ec9b5fd5178036788bf3344ac7b` candidate 2, reward=0.000000, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|ad_begin\|><s_a_3641><s_b_978><s_c_6171>", "<\|living_begin\|><s_a_4372><s_b_7342><s_c_3676>", "<\|prod_begin\|><s_a_345><s_b_2769><s_c_535>", "<\|video_begin\|><s_a_5288><s_b_5189><s_c_3973>"]
- `56b1345350d5bc9bb83952a06e4e479cae77a5b80c451d5f5a13cc5a019dd26e` candidate 2, reward=0.000000, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|prod_begin\|><s_a_2568><s_b_7423><s_c_2368>", "<\|prod_begin\|><s_a_1480><s_b_7270><s_c_5755>", "<\|prod_begin\|><s_a_7160><s_b_152><s_c_7101>", "<\|prod_begin\|><s_a_7160><s_b_5974><s_c_5044>", "<\|prod_begin\|><s_a_4661><s_b_6318><s_c_4133>"]
- `703dba95995ed9533d9ee8112df6c97db78cec21bada4d6ef5a6ecdb4f585ce1` candidate 1, reward=0.000000, violations=['wrong_selection_sid', 'wrong_selection_sid']: ["<\|video_begin\|><s_a_2764><s_b_7894><s_c_6341>", "<\|prod_begin\|><s_a_5547><s_b_6751><s_c_4115>"]
- `c8622fd30302cfbe377a39c7fbb4921940480909d9cab0d15397f638fa1d944e` candidate 0, reward=0.000000, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|ad_begin\|><s_a_7038><s_b_6771><s_c_5141>", "<\|video_begin\|><s_a_1985><s_b_4181><s_c_47>", "<\|video_begin\|><s_a_4963><s_b_2906><s_c_497>"]
- `c8622fd30302cfbe377a39c7fbb4921940480909d9cab0d15397f638fa1d944e` candidate 1, reward=0.000000, violations=['wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid', 'wrong_selection_sid']: ["<\|prod_begin\|><s_a_513><s_b_2789><s_c_3297>", "<\|ad_begin\|><s_a_7038><s_b_6771><s_c_5141>", "<\|ad_begin\|><s_a_1985><s_b_2549><s_c_372>", "<\|video_begin\|><s_a_4963><s_b_2906><s_c_497>", "<\|video_begin\|><s_a_5555><s_b_6771><s_c_2814>", "<\|video_begin\|><s_a_1985><s_b_4932><s_c_2568>", "<\|video_begin\|><s_a_5417><s_b_6973><s_c_7840>", "<\|video_begin\|><s_a_5417><s_b_3589><s_c_6162>"]

### Chain Low Reward Valid Grounded No Duplicate

- `51b7568efe276fbefd3d37169c49ec2a2e4b86fb0c788105bf75dc37b3d4a39a` candidate 0, reward=0.196975, violations=[]: {"logic_chain": {"name": "从被动接收到主动探索的游戏策略深化", "events": [{"date": "2026-02-02", "action": "[视频-长播] <\|video_begin\|><s_a_6988><s_b_6170><s_c_2067>", "logic": "问题触发：通过观看视频，首次接触到“金铲铲之战”中“海克斯大乱斗”这一具体游戏模式，并理解了其中的“海克斯”机制，产生了初步认知。"}, {"date": "2026-02-02", "action": "[视频-评论/长播] <\|video_begin\|><s_a_3062><s_b_4932><s_c_6919>", "logic": "认知深化与主动求证：从被动观看转为主动进行深度探索。用户开始关注“海克斯大乱斗”在不同赛季（S16）的演变，特别是新推出的“双龙引珠”海克斯机制，并开始评论和分析该机制的实际效果，体现了从“是什么”到“为什么”的认知增量。"}, {"date": "2026-02-02", "action": "[视频-长播] <\|video_begin\|><s_a_3062><s_b_4491><s_c_2067>", "logic": "策略收敛与应用：在理解了“双龙引珠”机制后，用户的兴趣迅速聚焦于该机制如何影响具体的游戏策略，开始观看关于“海克
- `044a0a231f27e97aa9650427590d1974058513ba798984f9cfb568549923a93a` candidate 2, reward=0.202741, violations=[]: {"logic_chain": {"name": "从泛化农业关注到具体农产品消费的转化", "events": [{"date": "2025-08-28", "action": "[商品-点击] <\|prod_begin\|><s_a_177><s_b_4246><s_c_1278>", "logic": "兴趣起点：对农业、农业机械和农耕过程产生兴趣，属于内容消费层面。"}, {"date": "2026-01-04", "action": "[商品-点击] <\|prod_begin\|><s_a_826><s_b_5101><s_c_4495>", "logic": "兴趣深化：将对农业的抽象兴趣转化为对具体农产品（牛肉）的关注，并首次出现“清真”这一关键限定词。"}, {"date": "2026-01-08", "action": "[商品-点击] <\|prod_begin\|><s_a_5422><s_b_6642><s_c_1023>", "logic": "需求确认与聚焦：在关注“清真”牛杂后，进一步将兴趣收敛到更细分的品类“牛油”，完成了从宽泛农产品到特定加工品类的认知跃迁。"}]}}
- `bd7c7c14481e5f860b65c189a2f817ec8d9f47bcc420c604b19e9f4a547b3190` candidate 0, reward=0.204924, violations=[]: {"logic_chain": {"name": "从关注心血管健康泛化科普到具体化用药知识的深化过程", "events": [{"date": "2025-07-06", "action": "[商品-购买] <\|prod_begin\|><s_a_6655><s_b_5><s_c_6524>", "logic": "初始触发：基于对健康问题的初步认知，购买具有心脑血管、抗氧化功能的保健品，问题指向较为模糊。"}, {"date": "2026-01-29", "action": "[视频-长播] <\|video_begin\|><s_a_7149><s_b_7016><s_c_557>", "logic": "认知深化：从泛化的保健品转向对特定心血管药物（丹参滴丸、稳心颗粒）的深入了解，关注点从“预防保健”升级为“疾病管理”和“药物作用机制对比”。"}, {"date": "2026-02-05", "action": "[商品-点击] <\|prod_begin\|><s_a_2950><s_b_6818><s_c_3692>", "logic": "决策收敛：在了解多种丹参类药物后，用户的兴趣点收敛到更具体的、具有“双倍浓缩”、“高倍”等强功能指向的丹参产品上，完成了从宽泛概念到具体解决方案的认知跃迁。"}]}}
- `56ddd09a41db2129518992f6e6b4dc5d5334b72196d595266391c1fc3b8c5c2f` candidate 0, reward=0.219641, violations=[]: {"logic_chain": {"name": "从被动接收到主动实践的DIY兴趣激发", "events": [{"date": "2026-02-01", "action": "[商品-点击] <\|prod_begin\|><s_a_3763><s_b_6454><s_c_1533>", "logic": "兴趣起点：被动浏览新奇宠物（鹦鹉），触发了对新事物的好奇心和潜在的饲养兴趣。"}, {"date": "2026-02-05", "action": "[视频-长播] <\|video_begin\|><s_a_4615><s_b_1740><s_c_4032>", "logic": "认知深化：从饲养兴趣延伸至对婴儿用品的关注，通过观看视频获取实用知识（如材质选择），体现了从被动消费到主动信息搜集的认知转变。"}]}}
- `e3d2b306a5be6d7ee73aedb76d3487431ed6905fd5de7244fb929618db085946` candidate 0, reward=0.280033, violations=[]: {"logic_chain": {"name": "从泛化生理焦虑到具体化健康调理的认知收敛", "events": [{"date": "2026-02-04", "action": "[广告-点击] <\|ad_begin\|><s_a_4030><s_b_8151><s_c_2915>", "logic": "问题触发：用户被动接收到关于中老年健康与疾病预防的泛化信息，可能触发了对自身或家人的健康关注。"}, {"date": "2026-02-04", "action": "[广告-点击] <\|ad_begin\|><s_a_1351><s_b_6379><s_c_1572>", "logic": "认知深化与聚焦：基于初步的健康焦虑，用户从泛化的预防概念转向了更具针对性的“调理”和“辨证”的中医养生方法，体现了从被动接收信息到主动寻求具体解决方案的认知跃迁。"}]}}

## Integrity

- LoRA checksum unchanged: True
- Frozen data SHA unchanged: True
- `requires_grad` parameters after load: 0
