"""Build the Phase 2 review artifact (HTML) from the recorded results JSON.

Mirrors the Phase 1 review artifact, with one deliberate difference: Phase 1's HTML
was hand-authored, so its numbers could drift from the artifact they describe. This
phase generates the chart data *from* `phase2_baseline_results.json`, so the review
cannot disagree with the run that produced it -- the same drift that put stale alpha
values in the results doc is structurally impossible here.

Emits, next to the results JSON:
  - `phase2_chartdata.json`          the injected data (inspectable on its own)
  - `phase2_review.html`             body fragment (matches Phase 1's .html)
  - `phase2_review_standalone.html`  full document, double-click to open

Run: `PYTHONPATH=src python -m dfa.build_phase2_review`
"""

from __future__ import annotations

import json

from . import config

TEMPLATE = config.ARTIFACTS_DIR / "phase2_review.template.html"
RESULTS = config.ARTIFACTS_DIR / "phase2_baseline_results.json"
CHARTDATA = config.ARTIFACTS_DIR / "phase2_chartdata.json"
OUT_FRAGMENT = config.ARTIFACTS_DIR / "phase2_review.html"
OUT_STANDALONE = config.ARTIFACTS_DIR / "phase2_review_standalone.html"

TITLE = "Phase 2 Review — Baseline & the Bar"

# The zero forecast scores exactly this under sum-then-divide WMAPE; see
# `baseline._zero_predictions`. Named here because it is the artifact's spine:
# every WMAPE chart draws it as the reference rule.
ZERO_BAR = 1.0

DATASET_LABEL = {"A": "A — dense", "B": "B — intermittent", "C": "C — slow/sparse"}
SB_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]


def build_chartdata(results: dict) -> dict:
    """Reshape the results JSON into exactly what the template's script consumes."""
    datasets = []
    for key in ("A", "B", "C"):
        r = results[key]
        td = r.get("transform_diagnostics", {})
        datasets.append({
            "key": key,
            "label": DATASET_LABEL[key],
            "cell": r["cell"],
            "n": r["n_series"],
            "coverage": r["sample_or_full"],
            "baseline": r["wmape_overall"],
            "l2": r["wmape_l2_subset"],
            "naive": r["wmape_naive_modelable"],
            "zero": r["wmape_zero"],
            "bar": r["phase3_bar"],
            "beats_zero": r["baseline_beats_zero"],
            "indicative": r["l2_indicative"],
            "routing": r["routing"],
            "alpha": r["config"]["alpha"],
            "transform": r["config"]["transform"],
            "sb": [{"cls": c, "wmape": r["wmape_by_sb_class"].get(c)}
                   for c in SB_ORDER if c in r["wmape_by_sb_class"]],
            "folds": [{"fold": i, "wmape": v}
                      for i, (_, v) in enumerate(sorted(r["wmape_by_fold"].items()))],
            "terciles": ([{"band": b, "wmape": r["wmape_by_volume_tercile"][b]}
                          for b in ("low", "mid", "high")]
                         if "wmape_by_volume_tercile" in r else None),
            "transforms": [{"name": t, "wmape": td[t]["wmape"], "bias": td[t]["bias"],
                            "alpha": td[t]["best_alpha"], "spread": td[t]["alpha_spread"]}
                           for t in ("raw", "log1p") if t in td],
            "leakage": r["leakage_check_pass"],
        })
    return {
        "zeroBar": ZERO_BAR,
        "datasets": datasets,
        "nBarsCleared": sum(1 for d in datasets if d["beats_zero"]),
    }


def render(template: str, chartdata: dict) -> str:
    payload = json.dumps(chartdata, separators=(",", ":"))
    if "/*CHARTDATA*/" not in template:
        raise ValueError("template is missing the /*CHARTDATA*/ injection point")
    return template.replace("/*CHARTDATA*/", payload)


def wrap_standalone(fragment: str, title: str) -> str:
    """Wrap the body fragment in a minimal document -- the Phase 1 standalone shape."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n<style>\n  *{{margin:0;padding:0}}\n"
        "  html{-webkit-text-size-adjust:100%}\n  img,svg{max-width:100%}\n"
        "  a{color:inherit}\n</style>\n</head>\n<body>\n"
        f"{fragment}\n</body>\n</html>\n"
    )


def main() -> None:
    results = json.loads(RESULTS.read_text())
    data = build_chartdata(results)
    CHARTDATA.write_text(json.dumps(data, indent=2))

    fragment = render(TEMPLATE.read_text(), data)
    OUT_FRAGMENT.write_text(fragment)
    OUT_STANDALONE.write_text(wrap_standalone(fragment, TITLE))

    for p in (CHARTDATA, OUT_FRAGMENT, OUT_STANDALONE):
        print(f"wrote {p}  ({p.stat().st_size:,} bytes)")
    cleared = data["nBarsCleared"]
    print(f"\n{cleared}/3 datasets clear the zero bar "
          f"({', '.join(d['key'] for d in data['datasets'] if d['beats_zero']) or 'none'})")


if __name__ == "__main__":
    main()
