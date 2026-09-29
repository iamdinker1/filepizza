"""Spoken-form normalisation for Hinglish physics scripts: numbers, units, symbols, equations.

The goal is that the TTS never has to guess how "F = ma", "9.8 m/s^2" or "3 × 10^8" is read.
Every formula span is returned with its spoken form so pronunciation can be QC'd separately
from expressiveness (an expressive but wrong formula must be rejected).

Readings default to common Indian-classroom English ("into" for ×, "by" for /, "square",
"v naught"). Override per teacher with a lexicon learned from their transcripts:
    normalise(text, lexicon={"=": "equal to", "v0": "v zero"})
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

UNIT_WORDS = {
    "m/s^2": "metre per second square", "m/s²": "metre per second square", "ms^-2": "metre per second square",
    "m/s": "metre per second", "km/h": "kilometre per hour", "kmph": "kilometre per hour", "km/hr": "kilometre per hour",
    "kg": "kg", "g": "gram", "mg": "milligram", "N": "newton", "J": "joule", "kJ": "kilojoule", "W": "watt",
    "kW": "kilowatt", "Hz": "hertz", "Pa": "pascal", "m": "metre", "cm": "centimetre", "mm": "millimetre",
    "km": "kilometre", "s": "second", "sec": "second", "ms": "millisecond", "min": "minute", "h": "hour",
    "V": "volt", "A": "ampere", "C": "coulomb", "K": "kelvin", "°C": "degree celsius", "°": "degree",
    "rad": "radian", "rad/s": "radian per second", "N/m": "newton per metre", "kg/m^3": "kg per metre cube",
    "J/s": "joule per second", "Nm": "newton metre", "eV": "electron volt", "Ω": "ohm", "T": "tesla",
}
SYMBOL_WORDS = {
    "=": "equals", "+": "plus", "-": "minus", "−": "minus", "×": "into", "*": "into", "·": "into", "÷": "divided by",
    "/": "by", "∝": "is proportional to", "≈": "is approximately", "≠": "is not equal to", "<": "is less than",
    ">": "is greater than", "≤": "is less than or equal to", "≥": "is greater than or equal to", "√": "root",
    "Δ": "delta", "∆": "delta", "π": "pi", "θ": "theta", "α": "alpha", "β": "beta", "γ": "gamma", "ω": "omega",
    "λ": "lambda", "μ": "mu", "σ": "sigma", "ρ": "rho", "φ": "phi", "τ": "tau", "∑": "summation", "∫": "integral",
    "%": "percent", "→": "gives",
}
FUNCS = {"sin": "sine", "cos": "cos", "tan": "tan", "log": "log", "ln": "l n", "exp": "exponential", "max": "max",
         "min": "min", "KE": "K E", "PE": "P E", "net": "net"}
FRACTIONS = {"1/2": "half", "1/3": "one third", "2/3": "two third", "1/4": "one fourth", "3/4": "three fourth"}


def int_words(n: int) -> str:
    if n < 0:
        return "minus " + int_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_words(n % 100))
    for div, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand")):
        if n >= div:
            return int_words(n // div) + " " + name + ("" if n % div == 0 else " " + int_words(n % div))
    return str(n)


def num_words(s: str) -> str:
    s = s.replace(",", "")
    neg = s.startswith(("-", "−"))
    s = s.lstrip("-−")
    if "." in s:
        a, b = s.split(".", 1)
        out = (int_words(int(a)) if a else "zero") + " point " + " ".join(ONES[int(d)] for d in b if d.isdigit())
    else:
        out = int_words(int(s))
    return ("minus " + out) if neg else out


@dataclass
class FormulaSpan:
    start: int
    end: int
    written: str
    spoken: str


@dataclass
class Normalised:
    text: str
    spoken: str
    formulas: list[FormulaSpan] = field(default_factory=list)


_UNIT_ALT = "|".join(sorted((re.escape(u) for u in UNIT_WORDS), key=len, reverse=True))
_GREEK = "ΔΔθωπλμαβγσρφτ"
_EXP = r"\^\s*\(?\s*[-−]?\s*\w+\s*\)?"
_TOKEN = re.compile(
    rf"(?P<numunit>(?:\d+(?:\.\d+)?\s*[×x*]\s*10\s*{_EXP}|10\s*{_EXP}|\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_ALT})(?![\w/^²³]))"
    rf"|(?P<sci>\d+(?:\.\d+)?\s*[×x*]\s*10\s*{_EXP})"
    rf"|(?P<pow>(?:\d+(?:\.\d+)?|[A-Za-z{_GREEK}][A-Za-z0-9_']*)\s*(?:{_EXP}|[²³]))"
    r"|(?P<num>\d+(?:[.,]\d+)?)"
    rf"|(?P<ident>[A-Za-z{_GREEK}][A-Za-z0-9_']*)"
    r"|(?P<op>[=+\-−×*/÷∝≈≠<>≤≥√·→%])"
    r"|(?P<deva>[\u0900-\u097f]+)"
    r"|(?P<space>\s+)"
    r"|(?P<punct>.)")
_MATH_KINDS = {"numunit", "sci", "pow", "num", "op"}


def _is_word(tok: str) -> bool:
    from ..preprocess.lang import ROMAN_HI, _en_zipf

    low = tok.lower()
    return low in ROMAN_HI or _en_zipf(low) >= 3.5


def _speak_number_expr(s: str) -> str:
    """'3 × 10^8', '10^-3', '9.8' -> words."""
    m = re.fullmatch(rf"\s*(\d+(?:\.\d+)?)\s*[×x*]\s*10\s*\^\s*\(?\s*([-−]?)\s*(\w+)\s*\)?\s*", s)
    if m:
        exp = ("minus " if m.group(2) else "") + (num_words(m.group(3)) if m.group(3).isdigit() else m.group(3))
        return f"{num_words(m.group(1))} into ten to the power {exp}"
    m = re.fullmatch(r"\s*10\s*\^\s*\(?\s*([-−]?)\s*(\w+)\s*\)?\s*", s)
    if m:
        return "ten to the power " + ("minus " if m.group(1) else "") + (num_words(m.group(2)) if m.group(2).isdigit() else m.group(2))
    return num_words(s.strip())


def _speak_power(tok: str, lexicon: dict) -> str:
    m = re.fullmatch(r"(.+?)\s*(?:\^\s*\(?\s*([-−]?)\s*(\w+)\s*\)?|([²³]))", tok)
    base_raw = m.group(1)
    base = num_words(base_raw) if re.fullmatch(r"\d+(?:\.\d+)?", base_raw) else _speak_ident(base_raw, lexicon, split=True)
    if m.group(4):
        return base + (" square" if m.group(4) == "²" else " cube")
    neg, e = m.group(2), m.group(3)
    if not neg and e == "2":
        return base + " square"
    if not neg and e == "3":
        return base + " cube"
    ew = num_words(e) if e.isdigit() else _speak_ident(e, lexicon, split=True)
    return f"{base} to the power {'minus ' if neg else ''}{ew}"


def _speak_ident(tok: str, lexicon: dict, split: bool = True) -> str:
    if tok in lexicon:
        return lexicon[tok]
    if tok in FUNCS:
        return FUNCS[tok]
    m = re.fullmatch(rf"([A-Za-z{_GREEK}])_?(0|\d+|[a-z])?('*)", tok)
    if m:
        base = SYMBOL_WORDS.get(m.group(1), m.group(1).upper())
        sub = m.group(2)
        out = base
        if sub is not None:
            out += " naught" if sub == "0" else " " + (num_words(sub) if sub.isdigit() else sub.upper())
        return out + " dash" * len(m.group(3))
    if split and re.fullmatch(r"[A-Za-z]{2,3}", tok) or split and re.fullmatch(r"[A-Za-z]{4}", tok) and not _is_word(tok):
        return " ".join(ch.upper() for ch in tok)  # product of variables: ma -> M A, mgh -> M G H
    return tok


def _tokens(text: str):
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup if m.lastgroup != "unit" else "numunit"
        yield kind, m.group(), m.start(), m.end()


def _speak_run(run: list[tuple[str, str, int, int]], lexicon: dict, sym: dict) -> str:
    out: list[str] = []
    toks = [t for t in run if t[0] != "space"]
    i = 0
    while i < len(toks):
        kind, s, _, _ = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        nxt2 = toks[i + 2] if i + 2 < len(toks) else None
        if kind == "num" and nxt and nxt[1] == "/" and nxt2 and nxt2[0] in ("num", "numunit") and f"{s}/{nxt2[1]}" in FRACTIONS:
            out.append(FRACTIONS[f"{s}/{nxt2[1]}"])
            i += 3
            continue
        if kind == "numunit":
            m = re.fullmatch(rf"(.*?)\s*({_UNIT_ALT})", s)
            unit = lexicon.get(m.group(2), UNIT_WORDS[m.group(2)])
            out.append(f"{_speak_number_expr(m.group(1))} {unit}")
        elif kind == "sci":
            out.append(_speak_number_expr(s))
        elif kind == "pow":
            out.append(_speak_power(s, lexicon))
        elif kind == "num":
            out.append(num_words(s))
        elif kind == "op":
            if s == "/" and out and re.fullmatch(r"d [A-Z]", out[-1] or ""):
                out.append("by")
            elif s in ("-", "−") and (not out or toks[i - 1][0] == "op"):
                out.append("minus")
            else:
                out.append(lexicon.get(s, sym.get(s, s)))
        elif kind == "ident":
            if re.fullmatch(r"d[a-z]", s) and ((nxt and nxt[1] == "/") or (i > 0 and toks[i - 1][1] == "/")):
                out.append("d " + s[1].upper())
            else:
                out.append(_speak_ident(s, lexicon, split=len(s) <= 4))
        i += 1
    return " ".join(out)


def normalise(text: str, lexicon: dict | None = None) -> Normalised:
    lexicon = dict(lexicon or {})
    sym = {**SYMBOL_WORDS, **{k: v for k, v in lexicon.items() if k in SYMBOL_WORDS}}
    toks = list(_tokens(text))
    n = len(toks)

    def unary_minus(j):
        """'-' directly attached to a following number and detached from the previous token."""
        if toks[j][1] not in ("-", "−") or j + 1 >= n or toks[j + 1][0] not in ("num", "numunit", "sci"):
            return False
        return j == 0 or toks[j - 1][0] == "space"

    def binary_op(tok_idx):
        return tok_idx is not None and toks[tok_idx][0] == "op" and not unary_minus(tok_idx)

    def neighbour_idx(i, step):
        j = i + step
        while 0 <= j < n and toks[j][0] == "space":
            j += step
        return j if 0 <= j < n else None

    mathish = []
    for i, (kind, s, _, _) in enumerate(toks):
        if kind in _MATH_KINDS:
            mathish.append(True)
        elif kind == "ident":
            li, ri = neighbour_idx(i, -1), neighbour_idx(i, +1)
            near_op = binary_op(li) or binary_op(ri)
            glued_to_num = i > 0 and toks[i - 1][0] in ("num", "pow")  # "2as", "4pi"
            has_mark = bool(re.search(rf"[\d_'{_GREEK}]", s))
            mathish.append(bool(near_op or has_mark or glued_to_num))
        else:
            mathish.append(False)
    # maximal runs of math-ish tokens joined only by spaces
    formulas: list[FormulaSpan] = []
    out: list[str] = []
    i = 0
    while i < n:
        if not mathish[i]:
            out.append(toks[i][1])
            i += 1
            continue
        j = i
        run = [toks[i]]
        while True:
            k = j + 1
            while k < n and toks[k][0] == "space":
                k += 1
            if k < n and mathish[k]:
                run.extend(toks[j + 1:k + 1])
                j = k
            else:
                break
        kinds = {t[0] for t in run}
        speak = bool(kinds & {"op", "numunit", "sci", "pow", "num"}) or any(re.search(r"[\d_']", t[1]) for t in run if t[0] == "ident")
        if speak and not (kinds == {"op"} and run[0][1] in "-/"):
            spoken = _speak_run(run, lexicon, sym)
            formulas.append(FormulaSpan(run[0][2], run[-1][3], text[run[0][2]:run[-1][3]], spoken))
            out.append(spoken)
        else:
            out.append(text[run[0][2]:run[-1][3]])
        i = j + 1
    spoken = re.sub(r"\s+", " ", "".join(out)).strip()
    spoken = re.sub(r"\s+([,.;?!।])", r"\1", spoken)
    return Normalised(text=text, spoken=spoken, formulas=formulas)
