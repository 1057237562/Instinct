"""CPU-only equivalent of the reviewed bounded CPT identity scanner.

Keep this worker independent of transformers/torch; corpus process pools do
not need model runtimes. The decisions match the existing L3 review policy.
"""
from collections import Counter
import re
from scripts.data_builder.audit_identity_contamination import (
    CN_SELF_CUE_RE, EN_SELF_CUE_RE, OTHER_FOREIGN_ALT, OTHER_FOREIGN_TERMS,
    QWEN_HARD_ALT, QWEN_TERMS_HARD, QWEN_TERMS_WEAK,
    _build_claim_patterns, _build_en_claim_patterns, _is_negated,
    _is_non_assertion, _is_role_mention, _iter_sentences, _snippet,
)

EXTRA_VENDORS=r'minicpm[a-z0-9\-]*|openbmb|modelbest|面壁智能|面壁'
VENDOR_ALT=f'(?:{QWEN_HARD_ALT}|{OTHER_FOREIGN_ALT}|{EXTRA_VENDORS})'
CLAIM_PATTERNS=_build_claim_patterns(VENDOR_ALT)+_build_en_claim_patterns(VENDOR_ALT)
TERMS=sorted(set(QWEN_TERMS_HARD)|set(QWEN_TERMS_WEAK)|set(OTHER_FOREIGN_TERMS)
             |{'minicpm','openbmb','modelbest','面壁智能','面壁'},key=len,reverse=True)

def bounded(term):
    if re.fullmatch(r'[a-z0-9][a-z0-9 .+#\-]*',term,re.I):
        return rf'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])'
    return re.escape(term)

VENDOR_COUNT_RE=re.compile('|'.join(bounded(x) for x in TERMS),re.I)
VENDOR_RE=re.compile(VENDOR_COUNT_RE.pattern+r'|(?<![a-z0-9_])(?:qwen[a-z0-9.\-]*|minicpm[a-z0-9.\-]*)(?![a-z0-9_])',re.I)
ASCII=tuple(x.lower() for x in TERMS if x.isascii() and re.fullmatch(r'[a-z0-9][a-z0-9 .+#\-]*',x,re.I))
CJK=re.compile('|'.join(re.escape(x) for x in TERMS if not x.isascii()))

def identity_scan(text):
    hits=[];keywords=Counter();lowered=text.lower()
    if not(any(x in lowered for x in ASCII) or CJK.search(text)):return 'clean',hits,keywords
    for match in VENDOR_COUNT_RE.finditer(text):keywords[match.group(0).lower()]+=1
    for sentence in _iter_sentences(text):
        term_match=VENDOR_RE.search(sentence)
        if not term_match:continue
        claimed=False
        for pattern in CLAIM_PATTERNS:
            match=pattern.search(sentence)
            if not match:continue
            term=VENDOR_RE.search(sentence,match.start(),match.end())
            if term is None:term=VENDOR_RE.search(sentence[:match.end()])
            if term and _is_negated(sentence,term.start(),term.end()):continue
            if term and _is_role_mention(sentence,match,term):continue
            anchor=term.start() if term else match.start()
            tier='review' if _is_non_assertion(sentence,anchor) else 'claim'
            hits.append({'tier':tier,'match':match.group(0).strip()[:200],'snippet':_snippet(sentence,match.start(),match.end())})
            claimed=True;break
        if claimed:continue
        cue=CN_SELF_CUE_RE.search(sentence) or EN_SELF_CUE_RE.search(sentence)
        if cue and not _is_negated(sentence,term_match.start(),term_match.end()):
            hits.append({'tier':'review','match':term_match.group(0),'snippet':_snippet(sentence,term_match.start(),term_match.end())})
    if any(x['tier']=='claim' for x in hits):return 'claim',hits,keywords
    return ('review',hits,keywords) if hits else ('clean',hits,keywords)
