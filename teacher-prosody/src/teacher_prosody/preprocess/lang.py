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
yahi yeh ye zara zyada jyada ho hi ka ki ke hona hoke raho bachcho bacchon beta doston matra maatra disha
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
        return "en" if english_in_devanagari(t) else "hi"
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


# ---------------------------------------------------------------- Devanagari -> Latin (for cue matching)
_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
         "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
         "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
         "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y"}
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
_VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
        "औ": "au", "ऍ": "e", "ऑ": "o"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ॅ": "e", "ॉ": "o"}


def deva_to_latin(word: str) -> str:
    """Simplified Hindi romanisation with word-final and VC_CV schwa deletion
    (e.g. समझते -> samajhte, क्या -> kya, होगा -> hogaa). Good enough for cue matching."""
    units: list[list[str]] = []  # [consonant, vowel] ; vowel "" = virama, None = inherent schwa
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _CONS:
            c = _CONS[ch]
            if i + 1 < len(word) and word[i + 1] == "़":
                c = _NUKTA.get(ch, c)
                i += 1
            units.append([c, None])
        elif ch in _MATRA and units:
            units[-1][1] = _MATRA[ch]
        elif ch == "्" and units:
            units[-1][1] = ""
        elif ch in _VOW:
            units.append(["", _VOW[ch]])
        elif ch in "ंँ":
            units.append(["n", ""])
        elif ch == "ः":
            units.append(["h", ""])
        else:
            units.append([ch, ""])
        i += 1
    # schwa deletion: word-final, then VC_CV right-to-left
    if len(units) > 1 and units[-1][1] is None and units[-1][0]:
        units[-1][1] = ""
    for k in range(len(units) - 2, 0, -1):
        prev_has_vowel = units[k - 1][1] != ""
        nxt = units[k + 1]
        if units[k][1] is None and units[k][0] and prev_has_vowel and nxt[0] and nxt[1] != "":
            units[k][1] = ""
    return "".join(c + ("a" if v is None else v) for c, v in units)


def phonetic_key(text: str) -> str:
    """Loose key shared by romanised and transliterated Hindi: 'hogaa'/'hoga' -> 'hoga', 'dhyaan'/'dhyan' -> 'dhyan'."""
    out = []
    for tok in re.findall(r"[ऀ-ॿ]+|[A-Za-z']+|[?]", text):
        t = deva_to_latin(tok) if any("ऀ" <= c <= "ॿ" for c in tok) else tok.lower()
        t = t.replace("aa", "a").replace("ee", "i").replace("oo", "u").replace("w", "v")
        out.append(t)
    return " ".join(out)


QUESTION_TAILS = {"kya", "na", "n", "hai na", "haina", "right", "ok", "samjhe", "clear", "theek"}


def is_question_text(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    key = phonetic_key(t)
    toks = key.split()
    return bool(toks) and (toks[-1] in QUESTION_TAILS or " ".join(toks[-2:]) in QUESTION_TAILS)


# ---------------------------------------------------------------- English loanwords written in Devanagari
PHYSICS_TERMS = """physics physical quantity quantities scalar vector force mass time length velocity acceleration
displacement distance speed current voltage meter metre temperature unit units dimension magnitude direction energy work
power momentum friction gravity gravitational charge field formula equation graph slope area volume density pressure
example question answer chapter concept problem numerical value point zero total net resultant component angle rule law
theory system object particle motion rest uniform constant average instantaneous relative frame kilogram second minute
hour newton joule watt ampere kelvin candela mole volt ohm weight number addition subtraction multiplication division
fundamental derived standard measurement physically basically actually simple important definition""".split()


def _skeleton(latin: str) -> str:
    s = latin.lower()
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("ch", "C").replace("c", "k")
    for a, b in (("tion", "Sn"), ("sion", "Sn"), ("ture", "Cr"), ("ngth", "nt"), ("ph", "f"), ("qu", "kv"), ("q", "k"),
                 ("x", "ks"), ("w", "v"), ("sh", "S"), ("gh", "g")):
        s = s.replace(a, b)
    s = re.sub(r"g(?=[eiy])|ge$", "j", s)
    s = re.sub(r"[aeiouyh]", "", s)
    s = re.sub(r"n(?=[pb])", "m", s)  # anusvara assimilation (टेंप्रेचर ~ temperature)
    s = re.sub(r"[szjSC]", "s", s)  # sibilant/affricate class: फिजिक्स ~ physics
    return re.sub(r"(.)\1+", r"\1", s)


@lru_cache(maxsize=1)
def _english_skeletons() -> tuple[dict[str, str], dict[str, str]]:
    from wordfreq import top_n_list, zipf_frequency

    general: dict[str, str] = {}
    for w in top_n_list("en", 40000):
        if len(w) < 4 or not w.isalpha() or zipf_frequency(w, "en") < 3.0:
            continue
        k = _skeleton(w)
        if len(k) >= 3 and k not in general:
            general[k] = w
    physics = {}
    for w in PHYSICS_TERMS:
        k = _skeleton(w)
        if len(k) >= 2:
            physics.setdefault(k, w)
    return general, physics


@lru_cache(maxsize=1)
def _roman_hi_keys() -> set[str]:
    return {phonetic_key(w) for w in ROMAN_HI}


@lru_cache(maxsize=50000)
def english_in_devanagari(tok: str) -> str | None:
    """English word a Devanagari token spells (वोल्टेज -> voltage, फिजिक्स -> physics), else None.

    Guards: known romanised-Hindi words stay Hindi; physics terms match up to Hindi zipf 5.5
    (common loans such as मीटर, टाइम); other English words only when the token is not a common
    Hindi word (zipf < 4). The returned English word may be a same-skeleton neighbour - use it
    for language tagging, not as a translation.
    """
    from wordfreq import zipf_frequency

    if not any("\u0900" <= c <= "\u097f" for c in tok):
        return None
    if phonetic_key(tok) in _roman_hi_keys():
        return None
    sk = _skeleton(deva_to_latin(tok))
    general, physics = _english_skeletons()
    zh = zipf_frequency(tok, "hi")
    if sk in physics and len(sk) >= 2 and zh < 5.5:
        return physics[sk]
    if len(sk) < 3 or zh >= 4.0:
        return None
    return general.get(sk)
