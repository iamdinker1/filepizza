"""Blinded listening tests: package stimuli, then analyse ratings with honest uncertainty.

Design (see README): MUSHRA-style multi-stimulus pages per item (real teacher reference as a
hidden reference + systems), plus pairwise CMOS for the final two systems. Raters never see
system names; file names are random ids; the key is kept separately.

Questions (1-100 sliders, or 1-5 for ACR):
  teacher_explaining  "Does this feel like a teacher personally explaining this concept to you?"
  clarity, engagement, emphasis_pacing, hinglish_naturalness, voice_resemblance, fatigue (long-form only)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

QUESTIONS = ["teacher_explaining", "clarity", "engagement", "emphasis_pacing", "hinglish_naturalness",
             "voice_resemblance", "fatigue"]


def package(stimuli: list[dict], out_dir: str | Path, seed: int = 0) -> dict:
    """stimuli: [{"item": "nlm_03", "system": "baseline_elevenlabs", "path": "..."}].
    Writes blinded audio + trial list (randomised order per item) + a separate key file."""
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    key, trials = {}, defaultdict(list)
    for s in stimuli:
        bid = hashlib.sha1(f"{seed}:{s['item']}:{s['system']}".encode()).hexdigest()[:10]
        ext = Path(s["path"]).suffix or ".wav"
        shutil.copy(s["path"], out / "audio" / f"{bid}{ext}")
        key[bid] = {"item": s["item"], "system": s["system"]}
        trials[s["item"]].append(f"{bid}{ext}")
    pages = []
    for item, files in trials.items():
        rng.shuffle(files)
        pages.append({"item": item, "stimuli": files})
    rng.shuffle(pages)
    (out / "trials.json").write_text(json.dumps({"questions": QUESTIONS, "pages": pages}, indent=2))
    (out / "KEY_DO_NOT_SHARE.json").write_text(json.dumps(key, indent=2))
    with open(out / "ratings_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "stimulus", *QUESTIONS])
        for p in pages:
            for s in p["stimuli"]:
                w.writerow(["", s, *[""] * len(QUESTIONS)])
    return {"n_items": len(pages), "n_stimuli": len(key), "dir": str(out)}


def load_ratings(csv_path: str | Path, key_path: str | Path) -> list[dict]:
    key = json.loads(Path(key_path).read_text())
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            bid = Path(r["stimulus"]).stem
            if bid not in key or not r.get("rater"):
                continue
            for q in QUESTIONS:
                if r.get(q, "").strip():
                    rows.append({"rater": r["rater"], "item": key[bid]["item"], "system": key[bid]["system"],
                                 "question": q, "score": float(r[q])})
    return rows


def _cluster_bootstrap(rows: list[dict], stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Resample raters AND items (two-way cluster bootstrap) - ratings are not independent."""
    rng = np.random.default_rng(seed)
    raters = sorted({r["rater"] for r in rows})
    items = sorted({r["item"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["rater"], r["item"])].append(r)
    point = stat(rows)
    boots = []
    for _ in range(n_boot):
        rs = rng.choice(raters, len(raters))
        its = rng.choice(items, len(items))
        sample = [x for a in rs for b in its for x in by.get((a, b), [])]
        if sample:
            v = stat(sample)
            if np.isfinite(v):
                boots.append(v)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def summarise(rows: list[dict], question: str = "teacher_explaining", n_boot: int = 2000) -> dict:
    """Per-system mean with 95% two-way cluster bootstrap CI, pairwise differences, and rater agreement."""
    q = [r for r in rows if r["question"] == question]
    systems = sorted({r["system"] for r in q})
    out = {"question": question, "systems": {}, "pairwise": {}, "n_raters": len({r["rater"] for r in q}),
           "n_items": len({r["item"] for r in q})}
    for s in systems:
        out["systems"][s] = dict(zip(("mean", "ci_lo", "ci_hi"),
                                     _cluster_bootstrap([r for r in q if r["system"] == s],
                                                        lambda rr: np.mean([x["score"] for x in rr]), n_boot)))
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            def diff(rr, a=a, b=b):
                da = [x["score"] for x in rr if x["system"] == a]
                db = [x["score"] for x in rr if x["system"] == b]
                return np.mean(da) - np.mean(db) if da and db else np.nan
            out["pairwise"][f"{a} - {b}"] = dict(zip(("diff", "ci_lo", "ci_hi"), _cluster_bootstrap(q, diff, n_boot)))
    out["rater_agreement_icc"] = icc_raters(q)
    out["disagreements"] = disagreements(q)
    return out


def icc_raters(rows: list[dict]) -> float:
    """ICC(2,1)-style consistency across raters over (item, system) stimuli with complete ratings."""
    stim = defaultdict(dict)
    for r in rows:
        stim[(r["item"], r["system"])][r["rater"]] = r["score"]
    raters = sorted({r["rater"] for r in rows})
    M = np.array([[d[x] for x in raters] for d in stim.values() if all(x in d for x in raters)], float)
    if M.shape[0] < 3 or M.shape[1] < 2:
        return float("nan")
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def disagreements(rows: list[dict], spread: float = 40.0) -> list[dict]:
    """Stimuli where raters disagree by more than `spread` points (report them, listen to them)."""
    stim = defaultdict(list)
    for r in rows:
        stim[(r["item"], r["system"])].append(r["score"])
    return [{"item": i, "system": s, "min": min(v), "max": max(v)} for (i, s), v in stim.items()
            if len(v) >= 2 and max(v) - min(v) > spread]
