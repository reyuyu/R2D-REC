import re


PROMPT_TEMPLATE_VERSION = "gr_user_eval_aligned_v1"

ACTION_DEMO = """输出示例为（注意：以下案例来自其他用户，仅供参考输出格式，与上述用户交互历史无关）： ["<|prod_begin|><s_a_750><s_b_2525><s_c_3393>",
"<|living_begin|><s_a_6354><s_b_6678><s_c_4429>",
"<|ad_begin|><s_a_7852><s_b_3625><s_c_5951>",
"<|video_begin|><s_a_528><s_b_1682><s_c_7986>",
"大静儿在北京"]"""

CHAIN_DEMO = """有效逻辑链案例（注意：以下案例来自其他用户，仅供参考输出格式和逻辑标准，与上述用户交互历史无关）：
{
  "logic_chain": {
    "name": "熬夜健康焦虑驱动需求演化链",
    "events": [
      {
        "date": "2026-02-10",
        "action": "[搜索] 长期熬夜心慌怎么办；[搜索] 熬夜后心慌胸闷",
        "logic": "初始触发：基于生理不适的泛化求助。"
      },
      {
        "date": "2026-02-11",
        "action": "[视频-长播] <|video_begin|><s_a_6705><s_b_713><s_c_5747>",
        "logic": "认知深化：意识到生理症状可能与心理/神经因素有关，开始进行症状鉴别。"
      },
      {
        "date": "2026-02-12",
        "action": "[广告-点击] <|ad_begin|><s_a_6638><s_b_1822><s_c_7257>",
        "logic": "需求闭环：由认知深化转向针对性营养补剂购买。"
      }
    ]
  }
}"""

CHAIN_RULES = """核心提取逻辑：
按时间顺序提取 5 步以内的行为链，要求后项交互 B 必须与前项交互 A 构成场景需求补全、兴趣因果递进或需求深度细化的深度演化关系。严禁提取缺乏实质性逻辑关联的浅层并列行为，如品类平级罗列、相似内容重复、无关联交互等。
1. 场景需求补全：交互 B 与交互 A 必须在同一生活场景、主题类目下具有极强的需求补全关系。有效案例：购买“专业登山鞋” → 搜索“高原紫外线防护” → 点击“户外硬壳冲锋衣”。原因：户外极限场景的装备补齐。无效案例：点击“男鞋” → 购买“棉拖” → 购买“羽绒服”。原因：仅为冬季杂物凑单，应予剔除。
2. 兴趣因果递进：交互 B 是因交互 A 产生的副作用或新需求，或 A 是 B 的诱因。有效案例：搜索“全屋定制” → 观看“甲醛危害科普” → 搜索“工业级空气净化器”。原因：装修行为触发健康焦虑。
3. 需求深度细化：在同一主题类目下，交互 B 的需求比交互 A 更具体详细。有效案例：泛化搜索“新手露营” → 观看“黑胶帐篷测评” → 点击“三峰出征服者帐篷”。

要求与约束：
1. 节点精炼：每个 event 节点对应一个核心交互步骤，严禁重复交互内容；同一日期内属于同一演进步骤的同类型交互可合并为一个节点，用“；”分隔，但不同演进步骤的交互不得合并；
2. 逻辑溯源：logic 字段必须严格以 action 交互内容为依据，避免缺乏依据的过度推断；"""


def split_source_prompt(source_prompt: str) -> tuple[str, str]:
    marker = "角色任务："
    if marker not in source_prompt:
        raise ValueError("source prompt has no role-task marker")
    history = source_prompt.split(marker, 1)[0].rstrip()
    topics = re.findall(r"(?:^|\n)主题：([^\n]+)", source_prompt)
    if not topics:
        raise ValueError("source prompt has no topic")
    return history, topics[-1].strip()


def action_prompt(history: str, topic: str) -> str:
    return f"""{history}

角色任务：你是一个极端严苛的用户行为数据挖掘与数据格式化专家。请基于以上用户交互历史，围绕给定主题，提取出所有相关的历史交互。
主题：{topic}
输出格式要求：请直接输出JSON数组，不要输出任何额外解释。
{ACTION_DEMO}/no_think"""


def chain_prompt(history: str, topic: str) -> str:
    return f"""{history}

角色任务：你是一名严格遵循标准的用户行为数据挖掘与数据格式化专家。请根据以上用户交互历史，针对给定主题，提取具有高阶逻辑演进与深度意图关联的交互行为链路。

{CHAIN_RULES}

请针对以下主题提取行为逻辑链：
主题：{topic}

输出格式：请以 JSON 对象形式返回。其中 action 字段须严格对应 Timeline 中交互条目，不允许省略；若同一节点合并了多条交互，用“；”分隔。结构如下：
{{
  "logic_chain": {{
    "name": "{topic}",
    "events": [
      {{
        "date": "YYYY-MM-DD",
        "action": "[交互行为类型] 交互内容",
        "logic": "逻辑关键词：逻辑说明"
      }}
    ]
  }}
}}
{CHAIN_DEMO}
/no_think"""


def adapt_prompt(route: str, source_prompt: str) -> str:
    history, topic = split_source_prompt(source_prompt)
    if route == "action":
        prompt = action_prompt(history, topic)
    elif route == "chain":
        prompt = chain_prompt(history, topic)
    else:
        raise ValueError(f"unknown route: {route}")
    if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
        raise ValueError("raw prompt contains chat wrapper tokens")
    if not prompt.endswith("/no_think"):
        raise ValueError("adapted prompt lacks /no_think suffix")
    return prompt
