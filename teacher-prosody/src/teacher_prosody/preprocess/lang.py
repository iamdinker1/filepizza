"""Token-level language tagging for Hindi / English / Hinglish / maths.

Heuristic tagger (no model download needed): Devanagari -> hi; a curated romanised-Hindi
lexicon -> hi; English frequency list (wordfreq) -> en; symbols/numbers/units/variables -> math.
Ambiguous short tokens ("to", "do", "hi", "me", "the"...) are resolved from neighbours.
Accuracy must be measured on the gold set; swap in a trained tagger if it falls short.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..schema import LangSpan

# Frequent romanised Hindi tokens in Hinglish physics teaching (extend from real transcripts).
ROMAN_HI = set("""
ab abhi accha achha agar aisa aise aur aap apna apne apni aata aati aate baad bahut bata batao bhai bhi bilkul
bolo chalo chahiye dekho dekhiye dekh diya dijiye do dono ek ekdam fir phir gaya gayi gaye hai hain hota hoti
hote hoga hogi honge hua hui hue hum humne hamara hamare haan isko isse iska iski iske isliye inka jab jaisa jaise
jaata jaati jaate jaana jayega jaayega jitna jo kab kabhi kaha kahan kaise kam kar karo karna karke karenge
karte kardo kardiya kitna kitni kitne kis kisi ko koi kuch kya kyun kyunki kyuki lekin lo liya liye le lete
magar main mai matlab mein mera mere meri mujhe na nahi nahin neeche niche pata pehle par poora pura raha rahi
rahe sab sabse sahi samajh samjho samjhe samjha sath saath se sirf socho suno tab tak tarah theek thik tha thi
the toh tum tumhe tumhara upar usko usse uska uski uske wala wale wali waha wahan woh wo yaad yaani yahan yaha
yahi yeh ye zara zyada jyada ho hi ka ki ke hona hoke raho bachcho bacchon beta doston
""".split())

# Tokens valid in both languages; decided by context.
AMBIGUOUS = {"to", "do", "hi", "me", "the", "a", "is", "par", "may", "man", "log", "sab", "hai", "he", "ho", "ki", "na",
             "so", "use", "us", "pe", "main", "bus", "bhi"}
# Of those, the reading preferred when neighbours are Hindi.
AMBIG_HI_READING = {"to", "do", "hi", "me", "par", "hai", "ho", "ki", "na", "pe", "main", "bhi", "sab", "log"}

UNITS = {"m", "s", "kg", "g", "n", "j", "w", "cm", "mm", "km", "hz", "pa", "v", "a", "c", "k", "ms", "kmph", "rad"}
MATH_RE = re.compile(r"^([=+\-−×÷*/^<>≤≥≈∝√∆Δ∑∫πθαβγωλμσρφ°%]+|\d+([.,]\d+)?(e[-+]?\d+)?|[a-zA-Z](_?\d+|')?|"
                     r"[a-zA-Z]+\^\d+|[a-zA-Z]/[a-zA-Z0-9^]+|\d+/\d+)$")
TOKEN_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?|[^\sA-Za-z\dऀ-ॿ]")


@lru_cache(maxsize=50000)
def _en_zipf(w: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(w, "en")


def token_lang(tok: str, formula_context: bool = False) -> str:
    t = tok.strip()
    if not t:
        return "other"
    if any("ऀ" <= ch <= "ॿ" for ch in t):
        return "hi"
    low = t.lower()
    if MATH_RE.match(t) and (not t.isalpha() or len(t) == 1 and formula_context):
        return "math"
    if re.fullmatch(r"\d+([.,]\d+)?", t):
        return "math"
    if not re.search(r"[A-Za-z]", t):
        return "other"
    if low in AMBIGUOUS:
        return "amb"
    if low in ROMAN_HI:
        return "hi"
    if _en_zipf(low) >= 2.5:
        return "en"
    # unknown Latin word: Hindi-looking orthography (aa, ee, oo, kh, bh, dh, jh...) -> hi
    if re.search(r"(aa|ee|oo|kh|gh|bh|dh|jh|th|ph|sh)", low) and _en_zipf(low) < 1.5:
        return "hi"
    return "en"


def tag_tokens(text: str) -> list[tuple[str, int, int, str]]:
    """Return (token, start_char, end_char, lang) with ambiguous tokens resolved by context."""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    has_formula = bool(re.search(r"[=^/×]", text))
    langs = [token_lang(t, has_formula) for t, _, _ in toks]
    for i, l in enumerate(langs):
        if l != "amb":
            continue
        ctx = [x for x in langs[max(0, i - 3):i] + langs[i + 1:i + 4] if x in ("hi", "en")]
        hi_votes = ctx.count("hi")
        en_votes = ctx.count("en")
        low = toks[i][0].lower()
        if hi_votes > en_votes and low in AMBIG_HI_READING:
            langs[i] = "hi"
        elif en_votes > hi_votes:
            langs[i] = "en"
        else:
            langs[i] = "hi" if low in AMBIG_HI_READING else "en"
    return [(t, a, b, l) for (t, a, b), l in zip(toks, langs)]


def lang_spans(text: str) -> list[LangSpan]:
    """Merge consecutive same-language tokens into spans (punctuation absorbed into neighbours)."""
    spans: list[LangSpan] = []
    for tok, a, b, l in tag_tokens(text):
        if l == "other":
            if spans:
                spans[-1].end_char = b
            continue
        script = "deva" if l == "hi" and any("ऀ" <= c <= "ॿ" for c in tok) else "sym" if l == "math" else "latn"
        if spans and spans[-1].lang == l:
            spans[-1].end_char = b
            if spans[-1].script != script:
                spans[-1].script = "mixed"
        else:
            spans.append(LangSpan(a, b, l, script))
    return spans


def switch_points(text: str) -> list[int]:
    """Token indices where the language switches hi<->en (math tokens are transparent)."""
    tags = [l for *_, l in tag_tokens(text) if l in ("hi", "en")]
    return [i for i in range(1, len(tags)) if tags[i] != tags[i - 1]]


def cmi(text: str) -> float:
    """Code-Mixing Index (Das & Gambäck 2014): 100 * (1 - max_lang / (n - n_other)); 0 = monolingual."""
    tags = [l for *_, l in tag_tokens(text)]
    n = len(tags)
    u = sum(1 for l in tags if l not in ("hi", "en"))
    if n - u == 0:
        return 0.0
    mx = max(tags.count("hi"), tags.count("en"))
    return 100.0 * (1 - mx / (n - u))


def word_langs(words: list[str]) -> list[str]:
    """Language per word (context-resolved), robust to punctuation attached to words."""
    text, spans, pos = "", [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        text += w + " "
        pos += len(w) + 1
    toks = tag_tokens(text)
    out = []
    for a, b in spans:
        ls = [l for _, s, e, l in toks if s >= a and e <= b and l != "other"]
        out.append(ls[0] if ls else "other")
    return out
