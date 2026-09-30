"""Static HTML acoustic dashboard (self-contained: plots embedded as PNG data URIs)."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..features.f0 import to_semitones  # noqa: E402

CH_COLORS = {"pitch": "#2a6fdb", "duration": "#e08a1e", "energy": "#2f9e6e", "pause": "#8a5cc2"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def emphasis_trace(audio, f0, energy, words, rows, ref_hz, title: str = "") -> str:
    """Waveform + F0 (st re speaker median) + energy + per-word prominence split by channel."""
    t0 = max(0.0, words[0].start - 0.2)
    t1 = min(audio.duration, words[-1].end + 0.2)
    fig, ax = plt.subplots(4, 1, figsize=(10, 7.2), sharex=False, gridspec_kw={"height_ratios": [1, 1.4, 1, 1.3]})
    tt = np.arange(len(audio.y)) / audio.sr
    m = (tt >= t0) & (tt <= t1)
    ax[0].plot(tt[m], audio.y[m], lw=0.4, color="#555")
    ax[0].set_ylabel("wave")
    st = to_semitones(f0.hz, ref_hz)
    fm = (f0.times >= t0) & (f0.times <= t1)
    ax[1].plot(f0.times[fm], st[fm], ".", ms=2.5, color=CH_COLORS["pitch"])
    ax[1].axhline(0, color="#aaa", lw=0.6)
    ax[1].set_ylabel("F0 (st re median)")
    em = (energy.times >= t0) & (energy.times <= t1)
    ax[2].plot(energy.times[em], energy.db_norm[em], lw=0.8, color=CH_COLORS["energy"])
    ax[2].set_ylabel("energy (dB re speech)")
    for a in ax[:3]:
        a.set_xlim(t0, t1)
        for w in words:
            a.axvspan(w.start, w.end, color="#f2f2f2", zorder=0)
    for w in words:
        ax[1].text((w.start + w.end) / 2, ax[1].get_ylim()[1], w.w, ha="center", va="bottom", fontsize=8)
    x = np.arange(len(rows))
    bottom_pos = np.zeros(len(rows))
    from ..features.words import DEFAULT_WEIGHTS

    for ch, attr in (("pitch", "z_pitch"), ("duration", "z_duration"), ("energy", "z_energy"), ("pause", "z_pause")):
        vals = np.array([max(0.0, getattr(r, attr) * DEFAULT_WEIGHTS[ch]) for r in rows])
        ax[3].bar(x, vals, bottom=bottom_pos, color=CH_COLORS[ch], label=ch, width=0.7)
        bottom_pos += vals
    ax[3].plot(x, [r.prominence for r in rows], "k_", ms=18, label="prominence (net)")
    ax[3].set_xticks(x, [r.w for r in rows], fontsize=8, rotation=0)
    ax[3].set_ylabel("prominence parts")
    ax[3].legend(fontsize=7, ncol=5, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def overview_plots(rf) -> str:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    v = rf.f0.hz[rf.f0.voiced]
    if v.size:
        ax[0].hist(to_semitones(v, rf.ref_hz), bins=60, color=CH_COLORS["pitch"])
    ax[0].set_xlabel("F0 (st re median)")
    ax[0].set_title(f"median {rf.ref_hz:.0f} Hz", fontsize=9)
    ax[1].plot(rf.rate_t / 60, rf.rate, lw=0.6, color=CH_COLORS["duration"])
    ax[1].set_xlabel("time (min)")
    ax[1].set_ylabel("syll/s (articulation)")
    ax[2].plot(rf.energy.times / 60, rf.energy.db_norm, lw=0.3, color=CH_COLORS["energy"])
    ax[2].set_xlabel("time (min)")
    ax[2].set_ylabel("dB re speech")
    fig.tight_layout()
    return _png(fig)


def pause_plot(pauses_by_kind: dict[str, list[float]]) -> str:
    kinds = [k for k, v in pauses_by_kind.items() if v]
    fig, ax = plt.subplots(figsize=(8, 0.45 * max(3, len(kinds)) + 0.8))
    if kinds:
        ax.boxplot([pauses_by_kind[k] for k in kinds], orientation="horizontal", tick_labels=kinds, showfliers=True)
    ax.set_xlabel("pause duration (s)")
    fig.tight_layout()
    return _png(fig)


def _table(rows: list[list], head: list[str]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _fmt(x) -> str:
    if isinstance(x, float):
        return "–" if not np.isfinite(x) else f"{x:.2f}"
    return str(x)


def build(out_path: str | Path, title: str, sections: list[tuple[str, str]], note: str = "") -> Path:
    css = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d}}
:root[data-theme=dark]{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid var(--line);padding-bottom:4px}
.note{color:var(--mut);font-size:13px} img{max-width:100%;background:#fff;border-radius:6px}
table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto} td,th{border:1px solid var(--line);padding:3px 7px;text-align:right}
th{background:var(--card)} td:first-child,th:first-child{text-align:left} pre{white-space:pre-wrap;font-size:12px}
"""
    body = "".join(f"<h2>{html.escape(h)}</h2>{content}" for h, content in sections)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{css}</style></head><body><h1>{html.escape(title)}</h1>"
           f"<p class='note'>{html.escape(note)}</p>{body}</body></html>")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def img(src: str) -> str:
    return f"<img src='{src}'>"


def profile_tables(profile: dict) -> str:
    pb = profile.get("per_beat", {})
    rows = []
    for b, d in pb.items():
        rows.append([b, d["n"], d["rate_rel"].get("median", np.nan), d["f0_median_rel_st"].get("median", np.nan),
                     d["f0_range_st"].get("median", np.nan), d["energy_rel_db"].get("median", np.nan),
                     json.dumps(d["final_types"]), d["emph_density"].get("median", np.nan)])
    t1 = _table(rows, ["beat", "n", "rate ×", "F0 Δst", "F0 range st", "energy ΔdB", "finals", "emph density"])
    conf = profile.get("confounds", {})
    t2 = _table([[k, v.get("beat", np.nan), v.get("session", np.nan), v.get("condition", np.nan), v.get("topic", np.nan), v["verdict"]]
                 for k, v in conf.items()], ["feature", "η² beat", "η² session", "η² condition", "η² topic", "verdict"])
    other = {k: profile.get(k) for k in ("pre_reveal", "questions", "transitions", "emphasis_channels", "code_switch")}
    return (t1 + "<p class='note'>Rates are multipliers of the session median; F0/energy are offsets from it.</p>"
            + "<h3>What drives each feature (η² = share of variance)</h3>" + t2
            + "<h3>Dynamic patterns</h3><pre>" + html.escape(json.dumps(other, indent=1, default=float)) + "</pre>")
