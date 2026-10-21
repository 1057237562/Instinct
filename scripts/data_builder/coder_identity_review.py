"""Contextual review gate for pretraining identity candidates.

Triage hits never delete rows. Definite assistant-persona phrases and model-card
genres require an explicit, text-fingerprint-bound keep/drop decision. Other
hits are retained under the reviewed factual/code preservation policy, with
row-level audit records. Raw data is never rewritten.
"""

import re


BRANDS = r"(?:chat\s*gpt|qwen[\w.\-]*|openai|claude|gemini|bard|llama|deepseek|mistral|grok|minimax|alexa|siri|copilot|通义千问|通义|千问|文心一言|豆包|智谱|混元|小冰|小爱|百川)"
PERSONA = re.compile(
    rf"(?:\b(?:I\s+am|I['’]m|my\s+name\s+is|my\s+name['’]s|you\s+are)\s+(?:an?\s+)?{BRANDS}(?![\w])|"
    rf"(?:我是|我叫|你是|我的名字(?:叫|是))\s*{BRANDS}(?![妈爸姐哥弟妹叔姨])|"
    rf"\bI\s+am\s+(?:an?\s+)?(?:AI|large\s+language|language)\s*(?:model|assistant)?[^.\n]{{0,60}}(?:trained|created|developed|designed)\s+by\s+{BRANDS}|"
    rf"\bas\s+(?:an?\s+)?(?:AI\s+)?(?:large\s+)?(?:language\s+)?(?:model|AI)\b[^.\n]{{0,60}}(?:by\s+{BRANDS})[^.\n]{{0,40}}\bI\b|"
    rf"(?:我由|我是由)[^。\n]{{0,30}}(?:智谱|阿里云|阿里巴巴|通义|OpenAI)|"
    r"I\s+am\s+\w+Bot,\s+an?\s+AI\s+assistant)", re.IGNORECASE,
)
LLM = re.compile(r"chat\s*gpt|qwen|通义千问|大语言模型|语言模型|\blarge.language.model\b|\bllm\b", re.I)
CARD = re.compile(r"(?:\bmodel[ -]card\b|模型卡)|(?:\bwe\s+(?:are excited to\s+)?(?:release|introduce)\b|我们(?:发布|推出))[^.。\n]{0,120}(?:qwen|chat\s*gpt|通义千问|大语言模型)", re.I)


def review_reasons(text, component):
    reasons = []
    normalized = " ".join(text.lower().split())
    persona_cues = ("i am", "i'm", "i’m", "my name", "you are", "as a", "as an", "我是", "我叫", "你是", "我的名字", "我由")
    if any(cue in normalized for cue in persona_cues):
        for match in PERSONA.finditer(text):
            reasons.append({"rule": "assistant_persona_phrase", "evidence": text[max(0, match.start()-100):match.end()+160]})
    if component != "code" and any(cue in normalized for cue in ("model card", "model-card", "模型卡", "we release", "we introduce", "we are excited to", "我们发布", "我们推出")) and LLM.search(text):
        for match in CARD.finditer(text):
            reasons.append({"rule": "possible_model_card_promotion", "evidence": text[max(0, match.start()-100):match.end()+160]})
    return reasons


def reviewed_decision(text_id, text, component, decisions):
    reasons = review_reasons(text, component)
    if text_id in decisions:
        decision = decisions[text_id]
        if decision["decision"] not in ("keep", "drop") or not decision.get("reason"):
            raise ValueError(f"Invalid contextual decision: {text_id}")
        return decision["decision"], decision["reason"], reasons
    if reasons:
        raise ValueError(f"Unreviewed persona/model-card candidate: {text_id}")
    return "keep", "Factual third-party mention, author/character name, code identifier or ordinary code TODO; no targeted assistant-persona/model-card pattern after contextual triage.", reasons
