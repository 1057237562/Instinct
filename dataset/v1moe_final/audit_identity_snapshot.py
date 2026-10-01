"""Audit datasets for assistant-side model-identity contamination.

``filter_anomaly_candidates.py`` casts a wide net for review candidates. This
audit answers one narrower question: how many rows teach the assistant to
present itself as a different model, provider, or owner.

It inspects both assistant ``content`` and ``reasoning_content`` because the
SFT chat template renders the reasoning trace inside the assistant turn, so
``_generate_sft_labels`` trains it. A foreign-model name that is only discussed
as third-party knowledge is reported separately and is not contamination.

Target identity: Instinct, developed by L1bra, affiliated with no commercial
organization.

Run from the repository root::

    python dataset/scripts/audit_identity_contamination.py dataset/sft_t2t_mini.jsonl \
        --output dataset/review_candidates/sft_t2t_mini.identity_candidates.jsonl

The input is never modified; candidates are written with row-level provenance.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.source_format import classify, source_files  # noqa: E402


TARGET_IDENTITY = {
    "name": "Instinct",
    "developer": "L1bra",
    "affiliation": "none; personal project, not part of any commercial organization",
}

# Qwen-family and Alibaba terms.  The bare "阿里" form is kept out of the
# hard-claim patterns on purpose: it also matches transliterated names such as
# "阿里" for a person, so it is only used to raise review candidates.
QWEN_TERMS_HARD = (
    "qwen", "通义千问", "通义万相", "通义听悟", "通义实验室", "通义", "千问",
    "alibaba cloud", "alibaba group", "alibaba", "aliyun", "阿里云", "阿里巴巴",
    "达摩院", "damo",
)
QWEN_TERMS_WEAK = ("阿里",)
QWEN_HARD_ALT = (
    r"qwen[a-z0-9\-_]*(?:\.[a-z0-9\-_]+)*|通义千问|通义万相|通义听悟|通义实验室|通义|千问|"
    r"alibaba(?:\s+(?:cloud|group))?|aliyun|阿里云|阿里巴巴|达摩院|damo"
)
QWEN_TERM_RE = r"(?:" + QWEN_HARD_ALT + r"|阿里)"

QWEN_HARD_RE = re.compile("(?:" + QWEN_HARD_ALT + ")", re.IGNORECASE)
QWEN_WEAK_RE = re.compile(r"阿里", re.IGNORECASE)

# Non-Qwen model families, reported as a secondary signal.
OTHER_FOREIGN_TERMS = (
    "chatgpt", "openai", "gpt", "claude", "anthropic", "gemini", "deepmind",
    "bard", "llama", "deepseek", "深度求索", "mistral", "grok", "xai", "文心一言", "文心",
    "百度", "豆包", "字节跳动", "kimi", "moonshot", "智谱", "glm", "混元",
    "hunyuan", "腾讯", "元宝", "星火", "讯飞", "盘古", "华为", "internlm", "书生",
    "minimax", "阶跃星辰", "商汤", "sensenova", "百川", "baichuan", "零一万物",
    "面壁", "天工", "昆仑万维", "悟道", "ernie", "modelarts", "edas",
)
OTHER_FOREIGN_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(OTHER_FOREIGN_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)

COMMERCIAL_ORG_PATTERN = (
    r"(?:阿里云|阿里巴巴|腾讯|百度|字节跳动|华为|谷歌|微软|google|microsoft|meta|"
    r"openai|anthropic|deepseek|商汤|智谱|moonshot|科大讯飞|"
    r"商业公司|科技有限公司)"
)
COMMERCIAL_ORG_RE = re.compile(COMMERCIAL_ORG_PATTERN, re.IGNORECASE)

# Sentence-ish segmentation: Chinese sentence enders, newlines, and Latin
# sentence ends.  Splitting keeps a claim in one sentence from leaking into a
# neighbouring factual sentence, which is what makes the tiers precise.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=[.])[\s]+|[\r\n]+")

# A negation just before the foreign-model term turns a surface claim into a
# disavowal ("我是 Instinct，不是 Qwen").
NEGATION_RE = re.compile(
    r"(?:不是|并非|不属于|而不是|而非|不是由|不叫|不由|不受|并没有|没有|未曾|"
    r"no|not|never|isn't|aren't|wasn't|unlike|instead of|rather than)\s*$",
    re.IGNORECASE,
)

# Self-identification cues and explicit claim patterns.
CN_SELF_PREFIX = (
    r"我\s*(?:是|就是|乃是|叫|名叫|名为|的名字\s*是|的名称\s*是|的全名\s*是|"
    r"的模型\s*是|的模型名(?:称)?\s*是|的身份\s*是)"
)
CN_SELF_CUE_RE = re.compile(
    r"我(?:是|就是|乃是|叫|名叫|名为|全名|的名字|的名称|的模型|的模型名|的身份|由|基于|来自|属于|"
    r"的开发者|的创建者|的开发|的研发|的创造者|的团队|的所属|背后的模型|的底层|的底座|"
    r"使用|采用|之前是|原先是|其实是|也是一个)|本模型|本助手|本AI|我的全名"
)
EN_SELF_CUE_RE = re.compile(
    r"\b(?:i\s*(?:'m|’m|am)|my\s+(?:name|model|creator|developer|maker|company|team|backend|base)|"
    r"i\s+was\s+(?:created|developed|trained|built|made|designed)|i\s+(?:belong|come)|"
    r"created\s+by|developed\s+by|trained\s+by|built\s+by)\b",
    re.IGNORECASE,
)

def _build_claim_patterns(term_re: str) -> tuple[re.Pattern[str], ...]:
    """Build the self-identity claim patterns for one vendor term regex.

    The same shapes apply to every vendor in the list, so the Qwen-family pass
    and the all-vendor pass share them.
    """
    return (
    # 我是 <由 X 开发/训练/... 的> Qwen/通义千问/阿里云...
    re.compile(
        CN_SELF_PREFIX + r"\s*[：:]?\s*(?:一个|一款|一名|一位|个)?\s*"
        r"(?:由[^，。；\n]{0,24}?(?:开发|训练|研发|推出|发布|创造)的)?\s*" + term_re,
        re.IGNORECASE),
    # 我是由/来自/属于 <...> Qwen —— "我是由通义实验室研发的"
    re.compile(
        CN_SELF_PREFIX + r"\s*(?:由|来自|属于|隶属于)\s*[^。！？；\n]{0,30}?" + term_re,
        re.IGNORECASE),
    # 同位语: 我是 X，<由|来自|属于> <...> Qwen —— "我是 Instinct，由通义实验室研发的..."
    re.compile(
        CN_SELF_PREFIX + r"\s*[^。！？；\n]{0,24}?[，,]\s*(?:由|来自|属于|隶属于|是|为)?\s*"
        r"[^。！？；\n]{0,10}?" + term_re,
        re.IGNORECASE),
    # 我的 <名字|模型|身份|开发者|底座|...> 是/为/由/来自 Qwen...
    re.compile(
        r"我的\s*(?:名字|名称|全名|模型名(?:称)?|模型|大模型|身份|开发者|创建者|开发团队|研发团队|开发公司|"
        r"所属公司|母公司|底座(?:模型)?|底层模型|基础模型|背后(?:的模型)?|前身|定位)\s*"
        r"[^，。；\n]{0,8}?(?:是|为|叫|由|来自|属于)\s*[^，。；\n]{0,16}?" + term_re,
        re.IGNORECASE),
    # 我由/基于/来自/属于/诞生于 Qwen...
    re.compile(
        r"我\s*(?:由|基于|来自|属于|诞生于|出自|脱胎于|继承自)\s*[^，。；\n]{0,20}?" + term_re,
        re.IGNORECASE),
    # 我使用的模型/底座是 Qwen
    re.compile(
        r"我\s*(?:使用|采用|调用|运行在)的\s*(?:底层)?\s*(?:模型|大模型|底座|基础模型|框架)\s*"
        r"[^，。；\n]{0,8}?(?:是|为)\s*[^，。；\n]{0,16}?" + term_re,
        re.IGNORECASE),
    # 我被 阿里云/Qwen 开发/训练
    re.compile(
        r"我\s*(?:被|受)\s*[^，。；\n]{0,16}?" + term_re + r"[^，。；\n]{0,12}?"
        r"(?:开发|训练|研发|创造|打造|推出|发布)",
        re.IGNORECASE),
    # 作为 <由 X 开发/研发/训练 的> 模型/助手/AI —— "作为由OpenAI开发的AI语言模型，我…"
    re.compile(
        r"作为\s*(?:一个|一名|一款|一位|一款)?\s*(?:由\s*)?[^。！？；\n]{0,28}?"
        r"(?:开发|研发|训练|创建|推出|设计|微调|提供)的\s*[^。！？；\n]{0,10}?"
        r"(?:AI|人工智能)?\s*(?:模型|助手|语言模型|大模型|聊天机器人|智能体)",
        re.IGNORECASE),
)

def _build_en_claim_patterns(term_re: str) -> tuple[re.Pattern[str], ...]:
    return (
    re.compile(
        r"\bI\s*(?:'m|’m|am)\s+(?:an?\s+|the\s+)?(?:[A-Za-z0-9][\w.\-]*\s+){0,4}?" + term_re,
        re.IGNORECASE),
    re.compile(
        r"\bI\s*(?:was|am|have\s+been|'ve\s+been)\s+(?:created|developed|trained|built|made|designed|released|fine-?tuned)\s+by\s+"
        r"[^.!?\n]{0,40}?" + term_re,
        re.IGNORECASE),
    re.compile(
        r"\bmy\s+(?:name|model|creator|developer|creator's|company|maker|backend|base\s+model)\b[^.!?\n]{0,40}?" + term_re,
        re.IGNORECASE),
    re.compile(
        r"\bI\s+(?:belong|come)\s+(?:to|from)\s+[^.!?\n]{0,40}?" + term_re,
        re.IGNORECASE),
    # "As an AI language model developed by <vendor>, I ..."
    re.compile(
        r"\bas\s+an?\s+[^.!?\n]{0,40}?(?:created|developed|trained|built|made|designed|released|provided)\s+by\s+"
        r"[^.!?\n]{0,30}?" + term_re,
        re.IGNORECASE),
)

QWEN_CLAIM_PATTERNS = _build_claim_patterns(QWEN_TERM_RE)
QWEN_EN_CLAIM_PATTERNS = _build_en_claim_patterns(QWEN_TERM_RE)

# The same claim shapes against every vendor term, so a boilerplate answer
# naming any provider ("作为由OpenAI开发的AI语言模型，我…") is not missed just
# because it is not the Qwen family.
OTHER_FOREIGN_ALT = "|".join(
    re.escape(term) for term in sorted(OTHER_FOREIGN_TERMS, key=len, reverse=True)
)
VENDOR_ANY_TERM_RE = "(?:" + QWEN_HARD_ALT + "|" + OTHER_FOREIGN_ALT + ")"
VENDOR_ANY_RE = re.compile(VENDOR_ANY_TERM_RE, re.IGNORECASE)
ANY_CLAIM_PATTERNS = _build_claim_patterns(VENDOR_ANY_TERM_RE)
ANY_EN_CLAIM_PATTERNS = _build_en_claim_patterns(VENDOR_ANY_TERM_RE)
CN_ORG_CLAIM_PATTERNS = (
    # 我是 <org> 开发/训练/推出 的 模型/助手/AI
    re.compile(
        r"我\s*(?:是|就是)\s*[^，。；\n]{0,12}?" + COMMERCIAL_ORG_PATTERN +
        r"[^，。；\n]{0,12}?(?:开发|训练|研发|推出|发布|创造|打造)的?[^，。；\n]{0,6}?"
        r"(?:模型|助手|大模型|AI|智能体|产品)",
        re.IGNORECASE),
    # 我由/来自/属于 <org>
    re.compile(r"我\s*(?:由|来自|属于|诞生于|出自)\s*[^，。；\n]{0,20}?" + COMMERCIAL_ORG_PATTERN, re.IGNORECASE),
    # 我的开发者/所属公司/母公司 是 <org>
    re.compile(
        r"我的\s*(?:开发者|创建者|开发团队|研发团队|开发公司|所属公司|母公司|公司)\s*"
        r"[^，。；\n]{0,8}?(?:是|为|由|来自|属于)\s*[^，。；\n]{0,16}?" + COMMERCIAL_ORG_PATTERN,
        re.IGNORECASE),
)

# ASCII-only boundaries: "\b" fails next to CJK because CJK characters count as
# word characters, so "L1bra的ECS" has no word boundary before ECS.
def _ascii_token(token: str) -> str:
    return rf"(?<![a-z0-9]){token}(?![a-z0-9])"


PRE_FILTER_EXTRA = tuple(
    _ascii_token(token) for token in ("ecs", "rds", "oss", "slb", "cdn", "odps", "maxcompute", "polardb")
) + (r"alibaba\.com", "飞天", "专有云", "容器镜像服务")
PRE_FILTER_RE = re.compile(
    "|".join(
        [re.escape(term) for term in sorted(set(QWEN_TERMS_HARD + QWEN_TERMS_WEAK + OTHER_FOREIGN_TERMS), key=len, reverse=True)]
        + list(PRE_FILTER_EXTRA)
    ),
    re.IGNORECASE,
)

# A corpus built by renaming a foreign model's brand name can leave the model's
# own name attached to that vendor ("L1bra 是阿里巴巴集团旗下的云计算服务品牌",
# "阿里巴巴推出 Instinct"). Such rows contain no first-person claim, so the
# identity patterns cannot see them, yet they teach the wrong affiliation.
SELF_BRAND_RE = re.compile(r"(?:l1bra|instinct)", re.IGNORECASE)
VENDOR_ORG_RE = re.compile(
    r"(?:阿里巴巴|阿里云|阿里集团|通义|达摩院|alibaba|aliyun|damo)", re.IGNORECASE,
)
VENDOR_RELATION_RE = re.compile(
    r"(?:旗下|属于|集团|推出|发布|研发|开发|品牌|云服务|官网|子公司|业务|团队|Inc\.|Group|damo|达摩院|"
    r"\b(?:brand|company|group|subsidiary|product|service|team|owned|provides|provider|launched|released|developed)\b)",
    re.IGNORECASE,
)
# The rename also produced parenthetical brand equations ("阿里巴巴云（L1bra）")
# and vendor-owned-product sentences ("通义万相是L1bra的AI作画工具").
BRAND_EQUATION_RE = re.compile(
    r"(?:阿里巴巴|阿里云|阿里集团|通义|达摩院|alibaba|aliyun)[^。！？；\n]{0,4}?[（(]\s*(?:l1bra|instinct)\s*[)）]"
    r"|(?:l1bra|instinct)[^。！？；\n]{0,4}?[（(][^)）\n]{0,24}?(?:阿里巴巴|阿里云|aliyun|通义)",
    re.IGNORECASE,
)
VENDOR_OWNS_BRAND_RE = re.compile(
    r"(?:阿里巴巴|阿里云|通义|达摩院|alibaba|aliyun)[^。！？；\n]{0,6}?是[^。！？；\n]{0,6}?(?:l1bra|instinct)[^。！？；\n]{0,4}?的",
    re.IGNORECASE,
)
# A renamed corpus also leaves vendor-specific product names attached to the
# model's own brand ("L1bra 容器镜像服务（Aliyun Container Registry）"), where no
# generic relation word appears. Those artifacts carry the vendor's own name, so
# they identify ownership without further context.
CLOUD_ARTIFACT_RE = re.compile(
    r"(?:aliyun|alibaba\.com|阿里云|容器镜像服务|云服务器|云数据库|对象存储|云盘|飞天|专有云|数据中台|"
    + "|".join(
        _ascii_token(token)
        for token in ("ecs", "rds", "oss", "odps", "maxcompute", "polardb", "slb", "cdn")
    ) + ")",
    re.IGNORECASE,
)


# Disavowals also follow the term ("与阿里巴巴无关", "阿里巴巴集团是另一家公司").
POST_NEGATION_RE = re.compile(
    r"^[^，。！？；\n]{0,8}?(?:无关|无任何关系|没有关系|没有任何关系|无关联|没有关联|不隶属|不属于|不是由|而不是|并非|不是|"
    r"是另一家|是另一款|是另一个|是其他的?|是不同的|是别的|"
    r"\b(?:is\s+not|are\s+not|isn't|aren't|has\s+nothing|is\s+unrelated|is\s+different)\b)",
    re.IGNORECASE,
)
# Speculation, expectation, or hypothetical framing right before the claim
# ("用户可能期待我是 Qwen", "if I were Qwen") is not an assertion of identity;
# such rows are reported for review rather than counted as contamination.
NON_ASSERTION_RE = re.compile(
    r"(?:可能|也许|或许|大概|似乎|好像|恐怕|如果|假如|假设|若是|万一|期待|以为|认为|觉得|假装|冒充|误以为|"
    r"\b(?:if|maybe|perhaps|might|may|could|would|suppose|assuming|expect|think)\b)[^。！？.!?\n]{0,10}$",
    re.IGNORECASE,
)
# A user asserting the assistant's identity ("你是通义实验室研发的吧？") that the
# assistant accepts without repeating the term is contamination the term-based
# patterns cannot see, so the pair is captured separately.
IDENTITY_QUESTION_RE = re.compile(
    r"你是(?:不是|否|不|由|哪|什么|谁|通义|qwen|阿里)|你是谁|你叫(?:什么|做)?|你的(?:名字|名称|身份|开发者|模型|来源|创建者)|"
    r"你背后的模型|你属于|你是由谁|\bwho are you\b|\bwhat model are you\b|\bare you\b|\byour (?:name|model|developer|creator)\b",
    re.IGNORECASE,
)
AFFIRM_RE = re.compile(
    r"是的|对的|没错|说得对|正确|正是|当然|确实|的确|不假|如您所说|你说得对|\byes\b|\bcorrect\b|\bindeed\b|\bexactly\b|\bthat's right\b",
    re.IGNORECASE,
)
ROLE_SUFFIX_AFTER_RE = re.compile(
    r"^[\s的]*(?:模型|大模型|系列|产品|平台|服务)?[\s的]*"
    r"(?:用户|粉丝|开发者|员工|爱好者|使用者|会员|客户|支持者|工程师|研究员|实习生|竞品|对手|竞争对手)",
)
EN_ROLE_WORD_RE = re.compile(
    r"\b(?:user|fan|developer|employee|member|customer|supporter|client|advocate|engineer|researcher|critic)\b",
    re.IGNORECASE,
)


def _iter_sentences(text: str) -> Iterator[str]:
    for part in SENTENCE_SPLIT_RE.split(text):
        part = part.strip()
        if part:
            yield part


def _snippet(text: str, start: int, end: int, radius: int = 60) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    fragment = re.sub(r"\s+", " ", text[left:right]).strip()
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{fragment}{suffix}"


def _is_negated(sentence: str, start: int, end: int | None = None, window: int = 16) -> bool:
    """True when a negation precedes or immediately follows the term.

    Negation must be tested at the foreign-model term's own position: ``我是
    Qwen`` and ``我不是 Qwen`` share the leading ``我是``, and a disavowal such as
    ``与阿里巴巴无关`` puts the negation after the term.
    """
    if NEGATION_RE.search(sentence[max(0, start - window):start]):
        return True
    if end is not None and POST_NEGATION_RE.search(sentence[end:end + 12]):
        return True
    return False


def _is_non_assertion(sentence: str, position: int, window: int = 22) -> bool:
    """True when speculative framing precedes the claim position."""
    return bool(NON_ASSERTION_RE.search(sentence[max(0, position - window):position]))


def _term_match(sentence: str, match: re.Match[str], term_re: re.Pattern[str]) -> re.Match[str] | None:
    """Locate the foreign-model term inside a claim match."""
    found = term_re.search(sentence, match.start(), match.end())
    if found:
        return found
    return term_re.search(sentence[:match.end()])


def _is_role_mention(sentence: str, match: re.Match[str], term: re.Match[str]) -> bool:
    """True for "我是 Qwen 的用户" / "I am a fan of Qwen" style role statements."""
    if ROLE_SUFFIX_AFTER_RE.search(sentence[term.end():term.end() + 12]):
        return True
    return bool(EN_ROLE_WORD_RE.search(match.group(0)))


def _claim_hits(text: str) -> list[dict[str, str]]:
    """Return self-identity claims about a foreign model in one message text.

    ``tier`` is ``claim`` for an explicit self-identification and ``review`` for
    a sentence that only places a self-cue next to a foreign-model name, which
    needs a human or model decision.
    """
    hits: list[dict[str, str]] = []
    for sentence in _iter_sentences(text):
        term_match = VENDOR_ANY_RE.search(sentence)
        if not term_match:
            continue
        claimed = False
        for pattern in ANY_CLAIM_PATTERNS + ANY_EN_CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            term = _term_match(sentence, match, VENDOR_ANY_RE)
            anchor = term.start() if term else match.start()
            if term and _is_negated(sentence, term.start(), term.end()):
                continue
            if term and _is_role_mention(sentence, match, term):
                continue
            hits.append({
                "tier": "review" if _is_non_assertion(sentence, anchor) else "claim",
                "match": match.group(0).strip()[:200],
                "snippet": _snippet(sentence, match.start(), match.end()),
            })
            claimed = True
            break
        if claimed:
            continue
        # No explicit claim: a self-cue in the same sentence as the term still
        # needs a decision rather than an automatic verdict.
        cue = CN_SELF_CUE_RE.search(sentence) or EN_SELF_CUE_RE.search(sentence)
        if cue and not _is_negated(sentence, term_match.start(), term_match.end()):
            hits.append({
                "tier": "review",
                "match": term_match.group(0),
                "snippet": _snippet(sentence, term_match.start(), term_match.end()),
            })
    return hits


def _org_claim_hits(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for sentence in _iter_sentences(text):
        org_match = COMMERCIAL_ORG_RE.search(sentence)
        if not org_match:
            continue
        for pattern in CN_ORG_CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            term = _term_match(sentence, match, COMMERCIAL_ORG_RE)
            anchor = term.start() if term else match.start()
            if _is_negated(sentence, anchor, term.end() if term else None):
                continue
            if term and _is_role_mention(sentence, match, term):
                continue
            hits.append({
                "match": match.group(0).strip()[:200],
                "snippet": _snippet(sentence, match.start(), match.end()),
            })
            break
    return hits


def count_terms(text: str, counter: collections.Counter[str]) -> None:
    """Count Qwen-family term occurrences without double counting 阿里云/阿里."""
    hard_spans = [match.span() for match in QWEN_HARD_RE.finditer(text)]
    for match in QWEN_HARD_RE.finditer(text):
        counter[match.group(0).lower()] += 1
    for match in QWEN_WEAK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in hard_spans):
            continue
        counter[match.group(0).lower()] += 1


def _message_fields(message: Any) -> tuple[str, str, str]:
    if not isinstance(message, dict):
        return "", "", ""
    role = str(message.get("role", message.get("from", ""))).strip().lower()
    alias = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    content = message.get("content", message.get("value", message.get("text")))
    reasoning = message.get("reasoning_content")
    return alias.get(role, role), content if isinstance(content, str) else "", reasoning if isinstance(reasoning, str) else ""


def iter_turns(row: Any) -> Iterator[tuple[str, str, str]]:
    """Yield (role, content, reasoning_content) for chat-style rows."""
    if not isinstance(row, dict):
        return
    for key in ("conversations", "messages", "chosen", "rejected"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for message in value:
            role, content, reasoning = _message_fields(message)
            if role or content or reasoning:
                yield role, content, reasoning
        return
    prompt = row.get("prompt", row.get("question", row.get("instruction")))
    response = row.get("response", row.get("answer", row.get("output", row.get("completion"))))
    if isinstance(prompt, str):
        yield "user", prompt, ""
    if isinstance(response, str):
        yield "assistant", response, ""


def _brand_affiliation_hits(text: str) -> list[dict[str, str]]:
    """Find sentences that tie the model's own brand name to a foreign vendor."""
    hits: list[dict[str, str]] = []
    for sentence in _iter_sentences(text):
        brand = SELF_BRAND_RE.search(sentence)
        if not brand:
            continue
        org = VENDOR_ORG_RE.search(sentence)
        artifact = CLOUD_ARTIFACT_RE.search(sentence)
        equation = BRAND_EQUATION_RE.search(sentence)
        owns_brand = VENDOR_OWNS_BRAND_RE.search(sentence)
        # "L1bra 的 ModelArts": the brand owns a product of another vendor.
        cross_vendor = None
        for other in OTHER_FOREIGN_RE.finditer(sentence):
            if SELF_BRAND_RE.search(other.group(0)):
                continue
            gap = sentence[brand.end():other.start()] if other.start() > brand.end() else sentence[other.end():brand.start()]
            if len(gap) <= 3 and any(mark in gap for mark in ("的", "研发", "开发", "推出", "", " ")):
                cross_vendor = other
                break
        if not any((org, artifact, equation, owns_brand, cross_vendor)):
            continue
        if org and not VENDOR_RELATION_RE.search(sentence) and not any((artifact, equation, owns_brand)):
            continue
        if _is_negated(sentence, brand.start(), brand.end()):
            continue
        for other in (org, artifact):
            if other and _is_negated(sentence, other.start(), other.end()):
                break
        else:
            witness = org or artifact or equation or owns_brand or cross_vendor
            hits.append({
                "brand": brand.group(0),
                "vendor": witness.group(0).strip()[:40] if witness else "",
                "snippet": _snippet(sentence, brand.start(), brand.end()),
            })
    return hits


def _agreement_hits(turns: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Flag (user asserts foreign identity, assistant agrees) pairs.

    Only short accepting replies are caught; a long answer that elaborates on a
    false user assertion without repeating the term still needs review, so the
    row is reported rather than counted as confirmed contamination.
    """
    hits: list[dict[str, str]] = []
    for (user_role, user_text, user_reason), (answer_role, answer_text, _) in zip(turns, turns[1:]):
        if user_role != "user" or answer_role != "assistant":
            continue
        term = QWEN_HARD_RE.search(user_text) or QWEN_WEAK_RE.search(user_text)
        if not term or not IDENTITY_QUESTION_RE.search(user_text):
            continue
        answer = answer_text.strip()
        if QWEN_HARD_RE.search(answer) or QWEN_WEAK_RE.search(answer) or len(answer) > 150:
            continue
        if AFFIRM_RE.search(answer) and not NEGATION_RE.search(answer[:40]):
            hits.append({
                "user_assertion": _snippet(user_text, term.start(), term.end()),
                "assistant_reply": answer[:200],
            })
    return hits


def classify_row(row: Any) -> dict[str, Any] | None:
    turns = list(iter_turns(row))
    if not turns:
        return None
    joined = " ".join(text for _, content, reasoning in turns for text in (content, reasoning))
    if not PRE_FILTER_RE.search(joined):
        return None

    result: dict[str, list[dict[str, str]]] = {
        "assistant_content_identity_claims": [],
        "assistant_reasoning_identity_claims": [],
        "assistant_content_identity_review": [],
        "assistant_reasoning_identity_review": [],
        "assistant_commercial_affiliation_claims": [],
        "user_side_mentions": [],
        "assistant_agreement_with_user_claim": [],
        "assistant_brand_affiliation_facts": [],
    }
    terms: collections.Counter[str] = collections.Counter()
    for role, content, reasoning in turns:
        if role != "assistant":
            if PRE_FILTER_RE.search(content) or PRE_FILTER_RE.search(reasoning):
                result["user_side_mentions"].append({
                    "role": role or "non_assistant",
                    "snippet": _snippet(content or reasoning, 0, 0, radius=80),
                })
            count_terms(content, terms)
            count_terms(reasoning, terms)
            continue
        count_terms(content, terms)
        count_terms(reasoning, terms)
        for source, text in (("content", content), ("reasoning", reasoning)):
            for hit in _claim_hits(text):
                key = f"assistant_{source}_identity_claims" if hit["tier"] == "claim" else f"assistant_{source}_identity_review"
                result[key].append(hit)
        result["assistant_commercial_affiliation_claims"].extend(
            _org_claim_hits(content) + _org_claim_hits(reasoning)
        )
        result["assistant_brand_affiliation_facts"].extend(
            _brand_affiliation_hits(content) + _brand_affiliation_hits(reasoning)
        )
    if not any(result.values()) and not terms:
        return None
    result["assistant_agreement_with_user_claim"].extend(_agreement_hits(turns))
    hard = bool(result["assistant_content_identity_claims"] or result["assistant_reasoning_identity_claims"])
    review = result["assistant_content_identity_review"] + result["assistant_reasoning_identity_review"]
    result["qwen_terms"] = dict(terms)  # type: ignore[assignment]
    result["hard_claim"] = hard  # type: ignore[assignment]
    result["reasoning_only"] = bool(  # type: ignore[assignment]
        result["assistant_reasoning_identity_claims"] and not result["assistant_content_identity_claims"]
    )
    result["mention_only"] = bool(not hard and not review and terms)  # type: ignore[assignment]
    return dict(result)


def iter_rows(path: Path) -> Iterator[tuple[int, Any]]:
    if classify(path) == "parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, row
        return
    opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield row_number, json.loads(line)
            except json.JSONDecodeError:
                yield row_number, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(inputs: list[Path], output: Path, report_path: Path, max_candidates: int | None,
        overwrite: bool, emit: str = "review") -> dict[str, Any]:
    resolved: list[Path] = []
    for input_path in inputs:
        resolved.extend(Path(item) for item in source_files(input_path))
    resolved = list(dict.fromkeys(item.resolve() for item in resolved))
    output = output.resolve()
    report_path = report_path.resolve()
    if output == report_path:
        raise ValueError("Candidate output and report must use different paths.")
    if output in resolved or report_path in resolved:
        raise ValueError("Output and report paths must not overwrite an input dataset.")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError("Output/report exists; choose new paths or pass --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts: collections.Counter[str] = collections.Counter()
    terms: collections.Counter[str] = collections.Counter()
    source_reports = []
    written = 0
    output_hash = hashlib.sha256()
    temp_output = output.with_name(output.name + ".tmp")
    try:
        with temp_output.open("wb") as destination:
            for source_path in resolved:
                rows = contaminated = review_rows = mention_rows = 0
                for row_number, row in iter_rows(source_path):
                    rows += 1
                    counts["rows_scanned"] += 1
                    if row is None:
                        counts["invalid_json"] += 1
                        continue
                    verdict = classify_row(row)
                    if not verdict:
                        continue
                    counts["rows_with_foreign_model_mention"] += 1
                    terms.update(verdict["qwen_terms"])
                    needs_review = bool(
                        verdict["assistant_content_identity_review"]
                        or verdict["assistant_reasoning_identity_review"]
                        or verdict["assistant_agreement_with_user_claim"]
                        or verdict["assistant_brand_affiliation_facts"]
                    )
                    if verdict["hard_claim"]:
                        contaminated += 1
                        counts["contaminated_rows"] += 1
                        if verdict["assistant_content_identity_claims"]:
                            counts["contaminated_in_content"] += 1
                        if verdict["assistant_reasoning_identity_claims"]:
                            counts["contaminated_in_reasoning"] += 1
                            if not verdict["assistant_content_identity_claims"]:
                                counts["contaminated_reasoning_only"] += 1
                    elif needs_review:
                        review_rows += 1
                        counts["review_rows"] += 1
                    else:
                        mention_rows += 1
                        counts["mention_only_rows"] += 1
                    if verdict["assistant_commercial_affiliation_claims"]:
                        counts["commercial_affiliation_claim_rows"] += 1
                    if verdict["user_side_mentions"]:
                        counts["user_side_mention_rows"] += 1
                    if verdict["assistant_agreement_with_user_claim"]:
                        counts["assistant_agreement_rows"] += 1
                    if verdict["assistant_brand_affiliation_facts"]:
                        counts["brand_affiliation_fact_rows"] += 1
                    verdict_class = "contaminated" if verdict["hard_claim"] else "review" if needs_review else "mention_only"
                    if emit == "contaminated" and verdict_class != "contaminated":
                        continue
                    if emit == "review" and verdict_class == "mention_only":
                        continue
                    if max_candidates is not None and written >= max_candidates:
                        counts["candidate_limit_reached"] = 1
                        continue
                    candidate = {
                        "_review": {
                            "source": str(source_path),
                            "row_number": row_number,
                            "verdict": verdict_class,
                            "target_identity": TARGET_IDENTITY,
                            "evidence": {key: value for key, value in verdict.items() if key not in {"qwen_terms", "hard_claim", "reasoning_only", "mention_only"}},
                            "qwen_terms": verdict["qwen_terms"],
                            "instruction": "Decide keep, drop, or rewrite. A claim that the assistant is another model/provider must be removed or rewritten to the target identity; factual third-party mentions may be preserved.",
                        },
                        "record": row,
                    }
                    encoded = (json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    destination.write(encoded)
                    output_hash.update(encoded)
                    written += 1
                    counts["candidates_written"] += 1
                source_reports.append({
                    "path": str(source_path),
                    "rows_scanned": rows,
                    "contaminated_rows": contaminated,
                    "review_rows": review_rows,
                    "mention_only_rows": mention_rows,
                    "sha256": sha256_file(source_path),
                })
        os.replace(temp_output, output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise

    report = {
        "tool": "audit_identity_contamination",
        "version": 1,
        "target_identity": TARGET_IDENTITY,
        "emit": emit,
        "inputs": source_reports,
        "counts": dict(sorted(counts.items())),
        "qwen_term_hits": dict(sorted(terms.items(), key=lambda item: -item[1])),
        "output": str(output),
        "output_sha256": output_hash.hexdigest(),
        "decision": "Counts are report-only; the source data was not modified.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL/.jsonl.gz/.parquet file or shard directory")
    parser.add_argument("--output", required=True, type=Path, help="Candidate JSONL output path")
    parser.add_argument("--report", type=Path, help="Audit report path (default: <output>.report.json)")
    parser.add_argument("--max-candidates", type=int, help="Optional cap on written candidates")
    parser.add_argument("--emit", choices=("all", "review", "contaminated"), default="review",
                        help="Which verdicts to write to the candidate file (default: review)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and report files")
    args = parser.parse_args()
    report_path = args.report or Path(str(args.output) + ".report.json")
    try:
        report = run(args.inputs, args.output, report_path, args.max_candidates, args.overwrite, args.emit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
