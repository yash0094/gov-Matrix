"""
pipeline.py — run the engine once, cache JSON artifacts for the API.

The API does NOT recompute on every request. Scoring 4,000 tenders takes a few
seconds; a reviewer hitting refresh should not pay for that. Run this whenever
new data lands (or on a cron), serve the cache.

    python -m src.engine.pipeline              # synthetic data
    python -m src.engine.pipeline --real path/to/tenders.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.engine import core

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
MODEL_VERSION = "0.3.1"


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"not serialisable: {type(o)}")


def analyse(tenders: pd.DataFrame, bids: pd.DataFrame, alpha: float = 0.10) -> dict:
    """Full pipeline. Returns every artifact the dashboard needs."""
    df = core.assign_peer_groups(tenders)

    raw = pd.DataFrame({
        "price":       core.sig_price(df),
        "price_hed":   core.sig_price_hedonic(df),
        "dispersion":  core.sig_bid_dispersion(df, bids),
        "margin":      core.sig_margin(df),
        "competition": core.sig_competition(df),
        "rotation":    core.sig_rotation(df),
        "network":     core.sig_network(df, bids),
    }).fillna(0.0)

    tender_signals = [c for c in raw.columns if c != "rotation"]
    w = {k: core.WEIGHTS[k] for k in tender_signals}
    pv = pd.DataFrame({c: core.empirical_p(raw[c], df["peer_group"]) for c in tender_signals})
    Z, contrib, denom = core.stouffer(pv, w)
    Z = Z * df["peer_confidence"].to_numpy()
    df["Z"] = Z.to_numpy()
    df["p_tender"] = core.permutation_null_p(pv, df["peer_group"], w, denom, Z)

    cases = core.build_cases(df, df["Z"], raw["rotation"])
    rej, cutoff = core.benjamini_hochberg(cases["p_case"], alpha=alpha)
    cases["flagged"] = rej.to_numpy()

    # ---- validation metrics (only meaningful on red-teamed data) -----------
    q = df.sort_values("Z", ascending=False)
    base = df["is_injected"].mean() if "is_injected" in df else np.nan
    metrics = {"model_version": MODEL_VERSION, "alpha": alpha,
               "n_tenders": int(len(df)), "n_cases": int(len(cases)),
               "base_rate": float(base) if base == base else None}
    if "is_injected" in df and df["is_injected"].any():
        metrics["tender_level"] = [
            {"k": k,
             "precision": float(q.head(k).is_injected.mean()),
             "recall": float(q.head(k).is_injected.sum() / df.is_injected.sum()),
             "lift": float(q.head(k).is_injected.mean() / base)}
            for k in (10, 25, 50, 100, 200)
        ]
        auc = stats.mannwhitneyu(q.loc[q.is_injected, "Z"], q.loc[~q.is_injected, "Z"],
                                 alternative="greater").statistic
        metrics["roc_auc"] = float(auc / (q.is_injected.sum() * (~q.is_injected).sum()))
        n_cc = int(cases.is_cartel_cell.sum())
        metrics["n_cartel_cells"] = n_cc
        metrics["case_level"] = [
            {"k": k, "precision": float(cases.head(k).is_cartel_cell.mean()),
             "recall": float(cases.head(k).is_cartel_cell.sum() / max(n_cc, 1))}
            for k in (3, 5, 10, 20)
        ]
    metrics["specialised_markets"] = [
        {"category": c,
         "mean_bidders": float(df.loc[df.category == c, "n_bidders"].mean()),
         "cases": int((cases.category == c).sum()),
         "flagged": int(cases.loc[cases.category == c, "flagged"].sum())}
        for c in sorted(df.category.unique())
    ]

    return {"tenders": df, "bids": bids, "cases": cases, "contrib": contrib,
            "raw": raw, "pvalues": pv, "metrics": metrics, "bh_cutoff": float(cutoff)}


def to_cache(res: dict, cache: Path = CACHE) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    df, cases, contrib = res["tenders"], res["cases"], res["contrib"]

    cases = cases.copy()
    cases["case_id"] = [f"{b}::{c}" for b, c in zip(cases.buyer, cases.category)]
    cases.to_json(cache / "cases.json", orient="records")

    keep = ["ocid", "buyer", "category", "region", "year", "quantity", "n_bidders",
            "bid_window_days", "winner", "winning_unit_price", "award_value",
            "published_value", "runner_up_margin", "director", "peer_group",
            "peer_level", "peer_n", "peer_confidence", "Z", "p_tender", "is_injected"]
    df[keep].to_json(cache / "tenders.json", orient="records")
    contrib.assign(ocid=df["ocid"].values).to_json(cache / "contributions.json", orient="records")
    res["bids"].to_json(cache / "bids.json", orient="records")

    peers = (df.groupby("peer_group")
               .agg(n=("ocid", "size"),
                    median_unit_price=("winning_unit_price", "median"),
                    median_bidders=("n_bidders", "median"))
               .reset_index())
    peers.to_json(cache / "peer_groups.json", orient="records")

    meta = dict(res["metrics"])
    meta["bh_cutoff"] = res["bh_cutoff"]
    meta["built_at"] = datetime.now(timezone.utc).isoformat()
    (cache / "metrics.json").write_text(json.dumps(meta, indent=2, default=_json_safe))
    print(f"cache written -> {cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-tenders", type=int, default=4000)
    ap.add_argument("--real", type=str, default=None,
                    help="parquet/csv of real tenders; expects the tender schema")
    ap.add_argument("--real-bids", type=str, default=None)
    args = ap.parse_args()

    if args.real:
        rd = pd.read_parquet if args.real.endswith(".parquet") else pd.read_csv
        tenders = rd(args.real)
        bids = rd(args.real_bids) if args.real_bids else pd.DataFrame(
            columns=["ocid", "vendor", "unit_price", "won", "director"])
        if "is_injected" not in tenders:
            tenders["is_injected"] = False
        print(f"loaded {len(tenders):,} real tenders")
    else:
        tenders, bids, injected = core.make_universe(n_tenders=args.n_tenders)
        print(f"synthetic universe: {len(tenders):,} tenders, {len(injected)} injected")

    res = analyse(tenders, bids, alpha=args.alpha)
    to_cache(res)
    m = res["metrics"]
    print(f"cases={m['n_cases']}  flagged={int(res['cases'].flagged.sum())}  "
          f"auc={m.get('roc_auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
