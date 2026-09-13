"""
FastAPI service. Serves the cached engine output plus the static dashboard.

    uvicorn src.api.main:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
STATIC = ROOT / "static"

app = FastAPI(
    title="Procurement case-prioritisation API",
    description="Ranks contracting processes by how much evidence justifies human "
                "review. Never labels a vendor corrupt.",
    version="0.3.1",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_STORE: dict = {}
FEEDBACK: list[dict] = []


def store() -> dict:
    """Lazy-load the cache. Raises a helpful error if the pipeline never ran."""
    if _STORE:
        return _STORE
    need = ["cases", "tenders", "contributions", "bids", "peer_groups"]
    missing = [n for n in need if not (CACHE / f"{n}.json").exists()]
    if missing:
        raise HTTPException(
            503,
            "Cache not built. Run:  python -m src.engine.pipeline")
    for n in need:
        _STORE[n] = pd.read_json(CACHE / f"{n}.json")
    _STORE["metrics"] = json.loads((CACHE / "metrics.json").read_text())
    return _STORE


def _clean(recs):
    """JSON has no NaN. Replace before serialising or the browser chokes."""
    return json.loads(pd.DataFrame(recs).replace({np.nan: None}).to_json(orient="records"))


class Feedback(BaseModel):
    verdict: str = Field(pattern="^(reviewed|explained|escalated)$")
    reviewer_id: str = "anonymous"
    notes: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "cache_built": (CACHE / "metrics.json").exists()}


@app.get("/api/cases")
def list_cases(q_max: float = 1.0, region: Optional[str] = None,
               category: Optional[str] = None, limit: int = 100, offset: int = 0):
    """The queue. Filter by q-value tolerance, not a fixed alpha."""
    s = store()
    c = s["cases"].copy()
    t = s["tenders"]
    region_of = t.groupby("buyer")["region"].first()
    c["region"] = c["buyer"].map(region_of)

    if region:
        c = c[c.region == region]
    if category:
        c = c[c.category == category]
    c = c[c.q_value <= q_max].sort_values("q_value")

    total = len(c)
    page = c.iloc[offset:offset + limit]
    return {
        "stated_fdr": q_max,
        "n_cases": total,
        "expected_false_discoveries": round(q_max * total, 1),
        "model_version": s["metrics"].get("model_version"),
        "cases": _clean(page.to_dict("records")),
    }


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str):
    """Evidence card. Contributions are exact, not approximated."""
    s = store()
    c = s["cases"]
    row = c[c.case_id == case_id]
    if row.empty:
        raise HTTPException(404, f"no case {case_id}")
    row = row.iloc[0]

    t = s["tenders"]
    sub = t[(t.buyer == row.buyer) & (t.category == row.category)].sort_values(
        "Z", ascending=False)
    contrib = s["contributions"].set_index("ocid")
    signals = [c for c in contrib.columns]

    top = sub.iloc[0]
    tc = contrib.loc[top.ocid]
    pos = tc[tc > 0].sum() or 1.0
    breakdown = sorted(
        [{"signal": k, "contribution": float(v),
          "share": float(max(v, 0) / pos),
          "counter_evidence": COUNTER.get(k)} for k, v in tc.items()],
        key=lambda d: -d["contribution"])

    peers = s["peer_groups"].set_index("peer_group")
    pg = peers.loc[top.peer_group] if top.peer_group in peers.index else None

    return {
        "case_id": case_id,
        "buyer": row.buyer, "category": row.category,
        "n_tenders": int(row.n_tenders), "case_value": float(row.case_value),
        "z_case": float(row.Z_case), "p_case": float(row.p_case),
        "q_value": float(row.q_value), "flagged": bool(row.flagged),
        "peer_confidence": float(row.peer_confidence),
        "exemplar_ocid": top.ocid,
        "signal_breakdown": breakdown,
        "peer_comparison": {
            "peer_group": top.peer_group,
            "peer_level": top.peer_level,
            "peer_n": int(top.peer_n),
            "peer_median_unit_price": float(pg.median_unit_price) if pg is not None else None,
            "this_unit_price": float(top.winning_unit_price),
            "peer_median_bidders": float(pg.median_bidders) if pg is not None else None,
            "this_n_bidders": int(top.n_bidders),
        },
        "tenders": _clean(sub.head(25).to_dict("records")),
        "disclaimer": "Not an accusation. This is a review-priority score, not a "
                      "finding of wrongdoing.",
    }


COUNTER = {
    "price": "Price premium may reflect urgency, spec differences, or a small order.",
    "price_hed": "Unmodelled quality or specification differences drive the residual.",
    "dispersion": "Tight bid clustering is normal in commoditised, price-transparent markets.",
    "margin": "A thin winning margin is expected where a reference price is published.",
    "competition": "Few bidders can be structural in a thin supplier market.",
    "network": "Shared directors occur legitimately within declared business groups.",
}


@app.get("/api/cases/{case_id}/peer-comparison")
def peer_comparison(case_id: str):
    """Distribution of the peer group, for the before/after chart."""
    s = store()
    c = s["cases"]
    row = c[c.case_id == case_id]
    if row.empty:
        raise HTTPException(404, f"no case {case_id}")
    row = row.iloc[0]
    t = s["tenders"]
    sub = t[(t.buyer == row.buyer) & (t.category == row.category)]
    top = sub.sort_values("Z", ascending=False).iloc[0]
    peer = t[t.peer_group == top.peer_group]
    return {
        "peer_group": top.peer_group,
        "this_value": float(top.winning_unit_price),
        "peer_prices": [float(x) for x in peer.winning_unit_price],
        # the same tender against everything, where it looks unremarkable.
        # this toggle IS the slide-5 argument.
        "national_prices": [float(x) for x in t[t.category == row.category].winning_unit_price],
    }


@app.get("/api/cases/{case_id}/network")
def network(case_id: str):
    """Entity subgraph: bidders linked by shared director."""
    s = store()
    c = s["cases"]
    row = c[c.case_id == case_id]
    if row.empty:
        raise HTTPException(404, f"no case {case_id}")
    row = row.iloc[0]
    t = s["tenders"]
    b = s["bids"]
    ocids = set(t[(t.buyer == row.buyer) & (t.category == row.category)].ocid)
    sub = b[b.ocid.isin(ocids)]
    if sub.empty:
        return {"nodes": [], "edges": []}

    vend = sub.groupby("vendor").agg(bids=("ocid", "size"), wins=("won", "sum"),
                                     director=("director", "first")).reset_index()
    nodes = [{"id": r.vendor, "label": r.vendor, "type": "vendor",
              "bids": int(r.bids), "wins": int(r.wins)} for r in vend.itertuples()]
    dirs = vend.director.dropna().unique()
    nodes += [{"id": d, "label": d, "type": "director"} for d in dirs]
    edges = [{"source": r.vendor, "target": r.director, "type": "shared_director"}
             for r in vend.itertuples() if pd.notna(r.director)]
    return {"nodes": nodes, "edges": edges}


@app.get("/api/cases/{case_id}/ripple")
def ripple(case_id: str):
    """Award value connected to this cluster through shared entities."""
    s = store()
    c = s["cases"]
    row = c[c.case_id == case_id]
    if row.empty:
        raise HTTPException(404, f"no case {case_id}")
    row = row.iloc[0]
    t = s["tenders"]
    sub = t[(t.buyer == row.buyer) & (t.category == row.category)]
    dirs = set(sub.director.dropna())
    conn = t[t.director.isin(dirs)]
    return {
        "clusters": len(dirs),
        "direct_value_inr": float(sub.award_value.sum()),
        "connected_contracts": int(len(conn)),
        "connected_value_inr": float(conn.award_value.sum()),
        "multiplier": float(conn.award_value.sum() / max(sub.award_value.sum(), 1)),
    }


@app.post("/api/cases/{case_id}/feedback", status_code=201)
def feedback(case_id: str, fb: Feedback):
    """Step 7 of the loop. In production this re-weights signals; here it is
    persisted so the demo can show the loop closing."""
    rec = {"case_id": case_id, **fb.model_dump()}
    FEEDBACK.append(rec)
    return {"stored": rec, "n_feedback": len(FEEDBACK)}


@app.get("/api/validation")
def validation():
    """Red-team results. Keep this endpoint public and link it from the deck."""
    return store()["metrics"]


@app.get("/api/public/contestability")
def contestability(ocid: str):
    """Free MSME tier. No auth. 'Was this tender contestable?'"""
    s = store()
    t = s["tenders"]
    row = t[t.ocid == ocid]
    if row.empty:
        raise HTTPException(404, f"no tender {ocid}")
    row = row.iloc[0]
    peers = s["peer_groups"].set_index("peer_group")
    med = float(peers.loc[row.peer_group].median_bidders) if row.peer_group in peers.index else None
    reasons = []
    if med and row.n_bidders < med:
        reasons.append(f"{int(row.n_bidders)} bidders vs a peer median of {med:.0f}")
    if row.bid_window_days < 14:
        reasons.append(f"bid window of {int(row.bid_window_days)} days")
    return {
        "ocid": ocid,
        "was_contestable": not reasons,
        "this_n_bidders": int(row.n_bidders),
        "peer_median_bidders": med,
        "peer_group": row.peer_group,
        "reasons": reasons,
    }


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")
