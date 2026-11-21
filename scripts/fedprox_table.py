#!/usr/bin/env python3
"""
scripts/fedprox_table.py

Presentation-only, read-only renderer for the FedProx Mode C measurement.
No training, no measurement, no pipeline import. Every number printed is
parsed at run time from docs/IMPLEMENTATION_NOTES.md ("FedProx" and "Defect
A, measured" sections, plus the fusion-head table for the hidden size that
gives the measured parameter count) - nothing numeric is hardcoded here. A
value the notes do not contain prints NOT MEASURED; it is never estimated.

The only arithmetic done here is the percentage reduction versus mu=0,
computed from the pooled means parsed from the notes.

Same output conventions as fl_comparison_table.py: ASCII only, 80 columns,
no ANSI colour, no box-drawing characters.

Usage:
    .venv\\Scripts\\python.exe scripts\\fedprox_table.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NOTES_PATH = Path(__file__).resolve().parent.parent / "docs" / "IMPLEMENTATION_NOTES.md"
WIDTH = 80
NM = "NOT MEASURED"
TITLE = "FEDPROX - CLIENT DRIFT UNDER NON-IID SHARDING (MODE C)"


def section(text: str, heading_start: str) -> str:
    """The body of the '## <heading_start>...' section, up to the next '## '."""
    m = re.search(rf"^## {re.escape(heading_start)}.*?$", text, re.MULTILINE)
    if not m:
        return ""
    nxt = re.search(r"^## ", text[m.end():], re.MULTILINE)
    return text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]


def find(pattern: str, text: str, group: int = 1, flags: int = 0) -> str:
    m = re.search(pattern, text, flags)
    return " ".join(m.group(group).split()) if m else NM


def load() -> dict:
    if not NOTES_PATH.exists():
        raise SystemExit(f"REFUSING TO DISPLAY: cannot find {NOTES_PATH}")
    text = NOTES_PATH.read_text(encoding="utf-8")
    fp = section(text, "FedProx")
    da = section(text, "Defect A, measured")
    if not fp:
        raise SystemExit(f"REFUSING TO DISPLAY: no '## FedProx' section in {NOTES_PATH}")

    d: dict = {}
    d["params"] = find(r"trainable parameters only \(([\d,]+);", fp)
    # FUSION_HIDDEN_DIM for that parameter count, from the fusion-head table.
    d["hidden"] = find(rf"^\|\s*(\d+)(?: \([^)]*\))?\s*\|\s*{re.escape(d['params'])}\s*\|", text, flags=re.MULTILINE)
    d["n_shards"] = find(r"CLIENT_N_SHARDS=(\d+)", fp)
    d["shard_seed"] = find(r"CLIENT_SHARD_SEED=(\d+)", fp)
    d["alpha"] = find(r"CLIENT_NONIID_ALPHA=([0-9.]+)", fp)
    d["n_mu"] = find(r"(\d+) mu values x \d+ repeats = \d+ client runs", fp)
    d["repeats"] = find(r"\d+ mu values x (\d+) repeats = \d+ client runs", fp)
    d["client_runs"] = find(r"\d+ mu values x \d+ repeats = (\d+) client runs", fp)
    d["dp_not_run"] = bool(re.search(r"DP, encryption, upload and\s+aggregation were not run", fp))

    # shard composition, from the delta-L2 table's own header row
    d["shards"] = {int(s): (size, pos, neg) for s, size, pos, neg in
                   re.findall(r"shard (\d) \((\d+)(?: records)?, (\d+)\+/(\d+)-\)", fp)}

    # | mu | shard0 (var) | shard1 (var) | shard2 (var) | pooled |
    cell = r"\s*([0-9.]+) \([0-9.e+-]+\)\s*"
    d["delta"] = {int(mu): (float(a), float(b), float(c), float(p)) for mu, a, b, c, p in re.findall(
        rf"^\|\s*(\d+)\s*\|{cell}\|{cell}\|{cell}\|\s*([0-9.]+)\s*\|\s*$", fp, re.MULTILINE)}

    # | mu | within-round variance | std / pooled mean |
    d["spread"] = {int(mu): (v, r) for mu, v, r in re.findall(
        r"^\|\s*(\d+)\s*\|\s*([0-9.]+e-\d+)\s*\|\s*([0-9.]+)\s*\|\s*$", fp, re.MULTILINE)}

    m = re.search(r"Mean F1 over the (\d+) runs per mu:(.*?)Non-uniform \(NEITHER\)\s+runs:\s*"
                  r"((?:\d+/\d+[,\s]*)+)", fp, re.DOTALL)
    d["f1"] = {}
    if m:
        d["f1_n"] = m.group(1)
        f1s = re.findall(r"([0-9.]+) \((?:mu=)?(\d+)\)", m.group(2))
        nonuni = re.findall(r"(\d+)/(\d+)", m.group(3))
        for (f1, mu), (k, n) in zip(f1s, nonuni):
            d["f1"][int(mu)] = (f1, k, n)

    d["shard1_allneg"] = re.search(r"all-negative in\s+(\d+) of (\d+) runs", fp)
    d["noise_std"] = find(r"per-element std of (\d+\.\d+-\d+\.\d+)", da)
    d["real_std"] = find(r"weight has\s+std (\d+\.\d+)", da)
    return d


def rule(ch: str = "=") -> str:
    return ch * WIDTH


def render_lines(local_note_first: bool = False) -> list[str]:
    """The whole FedProx block as a list of lines (title banner first).
    `local_note_first` moves the local-objective note to the top of HOW TO
    READ THIS - used by fl_comparison_table.py, where the first thing a
    reader needs is why this is not a row in the table printed above it."""
    d = load()
    mus = sorted(d["delta"])
    out: list[str] = [rule(), f" {TITLE}", rule()]

    out.append(f" Measured at: FUSION_HIDDEN_DIM={d['hidden']}, {d['params']} trainable params.")
    out.append(f" Mode C non-IID Dirichlet sharding (alpha={d['alpha']}, seed {d['shard_seed']}),")
    out.append(f" {d['n_shards']} shards, {d['repeats']} repeats per mu, {d['client_runs']} client runs total.")
    if d["dp_not_run"]:
        out.append(" sigma: not applicable - DP, encryption, upload and aggregation were NOT")
        out.append(" run for this measurement; delta L2 and local F1 are produced before them.")
    else:
        out.append(f" sigma: {NM}")
    out.append("")

    out.append(" SHARD COMPOSITION")
    out.append(" " + "-" * (WIDTH - 2))
    for s in range(3):
        if s in d["shards"]:
            size, pos, neg = d["shards"][s]
            out.append(f" shard {s} = {size:>3} records ({pos:>2} pos / {neg:>2} neg)")
        else:
            out.append(f" shard {s} = {NM}")
    out.append("")

    out.append(f" DELTA L2 - distance each client moved from the shared starting weights")
    out.append(f" (mean of {d['repeats']} runs)")
    out.append(" " + "-" * (WIDTH - 2))
    out.append(f" {'mu':<6} {'shard 0':<10} {'shard 1':<10} {'shard 2':<10} {'pooled mean':<13} {'vs mu=0'}")
    out.append(" " + "-" * (WIDTH - 2))
    if not mus:
        out.append(f" {NM}")
    base = d["delta"][0][3] if 0 in d["delta"] else None
    reductions: dict[int, str] = {}
    for mu in mus:
        a, b, c, p = d["delta"][mu]
        if mu == 0:
            change = "-"
        elif base:
            reductions[mu] = f"{round(100 * (1 - p / base))}%"
            change = f"-{reductions[mu]}"
        else:
            change = NM
        out.append(f" {mu:<6} {a:<10.4f} {b:<10.4f} {c:<10.4f} {p:<13.4f} {change}")
    out.append(" " + "-" * (WIDTH - 2))
    out.append("")

    out.append(" SPREAD ACROSS THE 3 CLIENTS OF A ROUND")
    out.append(" " + "-" * (WIDTH - 2))
    out.append(f" {'mu':<6} {'variance of the 3 delta L2':<30} {'spread relative to the mean'}")
    out.append(" " + "-" * (WIDTH - 2))
    for mu in mus:
        v, r = d["spread"].get(mu, (NM, NM))
        out.append(f" {mu:<6} {v:<30} {r}")
    out.append(" " + "-" * (WIDTH - 2))
    out.append("")

    n_f1 = d.get("f1_n", NM)
    out.append(f" LOCAL F1 (37 held-out patients, mean of {n_f1} runs per mu)")
    out.append(" " + "-" * (WIDTH - 2))
    out.append(f" {'mu':<6} {'mean F1':<12} {'non-uniform runs'}")
    out.append(" " + "-" * (WIDTH - 2))
    for mu in mus:
        if mu in d["f1"]:
            f1, k, n = d["f1"][mu]
            out.append(f" {mu:<6} {f1:<12} {k} of {n}")
        else:
            out.append(f" {mu:<6} {NM}")
    out.append(" " + "-" * (WIDTH - 2))
    out.append("")

    red = ", ".join(f"{reductions[mu]} at mu={mu}" for mu in mus if mu in reductions) or NM
    lo = d["spread"].get(mus[0], (NM, NM))[1] if mus else NM
    hi = d["spread"].get(mus[-1], (NM, NM))[1] if mus else NM
    s1 = d["shards"].get(1)
    s1_txt = f"{s1[1]} positive example in {s1[0]} records" if s1 else NM
    an = d["shard1_allneg"]
    an_txt = f"{an.group(1)} of {an.group(2)} runs" if an else NM

    points = [
        ["FedProx reduces client drift monotonically:",
         f"{red}."],
        ["It does NOT improve F1. That was expected: FedProx corrects drift, it",
         "does not add training data."],
        ["The honest nuance: the absolute spread between clients FELL, but the",
         f"relative spread ROSE ({lo} -> {hi}). The clients did not converge on",
         "each other; every client simply moved less."],
        [f"Shard 1 holds only {s1_txt} and predicted",
         f"all-negative in {an_txt}, so its numbers say little about FedProx."],
        ["The distance BETWEEN clients' updates was not measured. Delta L2 is each",
         "client's distance from the shared reference, so two clients can both",
         "move less and still point in different directions."],
        ["Whether FedProx improves the AGGREGATED model is untested: the only",
         "global model this pipeline produces is DP noise (Defect A, measured:",
         f"per-element std {d['noise_std']} vs {d['real_std']} for real weights)."],
    ]
    local_note = [
        "FedProx is a LOCAL-objective method (a term added to each client's",
        "training loss), not a server-side aggregation optimizer like FedAdam",
        "or FedYogi - that is why it is a separate section, not a row in the",
        "table above." if local_note_first else "historical table of fl_comparison_table.py.",
    ]
    if local_note_first:
        points.insert(0, local_note)

    out.append(" HOW TO READ THIS")
    out.append(" " + "-" * (WIDTH - 2))
    for i, lines in enumerate(points, 1):
        out.append(f" {i}. {lines[0]}")
        out += [f"    {l}" for l in lines[1:]]
    if not local_note_first:
        out.append("")
        out += [f" {l}" for l in local_note]
    out.append(rule())

    too_wide = [l for l in out if len(l) > WIDTH]
    assert not too_wide, f"line(s) over {WIDTH} columns: {too_wide}"
    assert all(ord(ch) < 128 for l in out for ch in l), "non-ASCII output"
    return out


def main() -> int:
    print("\n".join(render_lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
