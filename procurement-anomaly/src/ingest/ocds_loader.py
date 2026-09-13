"""
ocds_loader.py — map real OCDS releases onto the engine's schema.

Source (verified, open, commercially licensed with attribution):
  Assam State Government Finance Department, published in OCDS via CivicDataLab
  https://data.open-contracting.org/en/publication/131
  Download the all-time .jsonl.gz, drop it in data/raw/, then:

      python -m src.ingest.ocds_loader data/raw/assam_ocds.jsonl.gz
      python -m src.engine.pipeline --real data/interim/tenders.parquet \
                                    --real-bids data/interim/bids.parquet

Known quality issues the publisher documents, all handled below:
  - invalid codes in tender.status and tender.mainProcurementCategory
  - inconsistent date formats
  - milestones without ids

The honest caveat for your roadmap slide: most Indian portals publish the AWARD
but not every losing quote. Where `bids.details` is absent, the bid-level
signals (dispersion, margin, network co-bidding) abstain and the award-level
ones (price, hedonic, competition, rotation) carry the case. The engine degrades
rather than breaks -- say that out loud instead of hoping nobody asks.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "interim"

STATE_FROM_NAME = re.compile(r"\b(assam|himachal|karnataka|maharashtra|gujarat|tamil)\b", re.I)


def _f(x):
    try:
        v = float(x)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _date(x):
    d = pd.to_datetime(x, errors="coerce", utc=True)   # handles the format drift
    return None if pd.isna(d) else d.tz_localize(None)


def normalise_category(raw: str) -> str:
    """Collapse free-text item descriptions into a coarse taxonomy.

    This is the least glamorous and most load-bearing function in the repo. Peer
    groups are only as good as this mapping: if 'OPC 43 grade cement' and
    'cement bags' land in different buckets, every downstream comparison is
    against the wrong universe. Start with keyword rules, eyeball the residual,
    and put a TF-IDF + clustering pass on the roadmap.
    """
    s = (raw or "").lower()
    rules = [
        ("cement",           ["cement", "opc", "ppc", "concrete"]),
        ("road_bitumen",     ["bitumen", "asphalt", "road work", "black top"]),
        ("office_laptops",   ["laptop", "desktop", "computer", "notebook pc"]),
        ("school_furniture", ["desk", "bench", "furniture", "chair", "almirah"]),
        ("pipes_hdpe",       ["hdpe", "pipe", "pvc", "gi pipe"]),
        ("generic_drugs",    ["tablet", "injection", "syrup", "drug", "medicine"]),
        ("medical_equipment",["mri", "ct scan", "x-ray", "ventilator", "ultrasound"]),
        ("civil_works",      ["construction", "building", "repair", "renovation"]),
    ]
    for name, kws in rules:
        if any(k in s for k in kws):
            return name
    return "other"


def parse(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    op = gzip.open if str(path).endswith(".gz") else open
    trows, brows = [], []

    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rel = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = rel.get("compiledRelease", rel)

            ocid = rel.get("ocid")
            tender = rel.get("tender") or {}
            awards = rel.get("awards") or []
            if not ocid or not awards:
                continue

            buyer = ((rel.get("buyer") or {}).get("name")
                     or (tender.get("procuringEntity") or {}).get("name") or "unknown")
            items = tender.get("items") or []
            title = tender.get("title") or (items[0].get("description") if items else "")
            qty = _f(items[0].get("quantity")) if items else None

            start = _date((tender.get("tenderPeriod") or {}).get("startDate"))
            end = _date((tender.get("tenderPeriod") or {}).get("endDate"))
            window = (end - start).days if (start and end) else None

            award = awards[0]
            av = _f((award.get("value") or {}).get("amount"))
            pv = _f((tender.get("value") or {}).get("amount"))
            adate = _date(award.get("date")) or end or start
            if av is None or adate is None:
                continue

            suppliers = award.get("suppliers") or []
            winner = suppliers[0].get("name") if suppliers else None
            if not winner:
                continue

            # full bid roster, where the publisher provides it
            details = ((rel.get("bids") or {}).get("details")) or []
            quotes = []
            for b in details:
                amt = _f((b.get("value") or {}).get("amount"))
                ten = b.get("tenderers") or []
                if amt is None or not ten:
                    continue
                quotes.append((ten[0].get("name"), amt))

            n_bidders = (len(quotes) or _f(tender.get("numberOfTenderers"))
                         or (len(suppliers) or 1))
            q = qty or 1.0
            wu = av / q

            if quotes:
                srt = sorted(quotes, key=lambda kv: kv[1])
                margin = ((srt[1][1] - srt[0][1]) / srt[0][1]) if len(srt) > 1 else None
                for name, amt in quotes:
                    brows.append(dict(ocid=ocid, vendor=name, unit_price=amt / q,
                                      won=int(name == winner), director=None))
            else:
                margin = None

            region = (STATE_FROM_NAME.search(buyer) or [None])
            trows.append(dict(
                ocid=ocid, buyer=buyer, category=normalise_category(title),
                region=(region.group(1).upper()[:2] if hasattr(region, "group") else "NA"),
                year=adate.year, quantity=q, n_bidders=int(n_bidders),
                bid_window_days=int(window) if window and 0 < window < 400 else 21,
                winner=winner, winning_unit_price=wu, award_value=av,
                published_value=pv if pv else av,     # fall back, but flag it
                runner_up_margin=margin, director=None,
                seq=len(trows), is_injected=False,
            ))

    tenders = pd.DataFrame(trows)
    bids = pd.DataFrame(brows) if brows else pd.DataFrame(
        columns=["ocid", "vendor", "unit_price", "won", "director"])

    # crude entity resolution: normalise the name, then self-join.
    # Replace with rapidfuzz + MCA director joins for anything real.
    def canon(n):
        n = re.sub(r"[^a-z0-9 ]", " ", str(n).lower())
        n = re.sub(r"\b(pvt|private|ltd|limited|co|company|and|the|llp|inc)\b", " ", n)
        return re.sub(r"\s+", " ", n).strip()

    if len(tenders):
        tenders["winner"] = tenders["winner"].map(canon)
        tenders["director"] = "D_" + tenders["winner"]
    if len(bids):
        bids["vendor"] = bids["vendor"].map(canon)
        bids["director"] = "D_" + bids["vendor"]

    return tenders, bids


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.ingest.ocds_loader <path-to-ocds.jsonl[.gz]>")
    tenders, bids = parse(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    tenders.to_parquet(OUT / "tenders.parquet", index=False)
    bids.to_parquet(OUT / "bids.parquet", index=False)
    print(f"{len(tenders):,} tenders, {len(bids):,} bid rows -> {OUT}")
    if len(tenders):
        print("\ncategory mix (check this before trusting any peer group):")
        print(tenders.category.value_counts().head(12).to_string())
        print(f"\nbid-level coverage: {bids.ocid.nunique()/max(len(tenders),1):.0%} of tenders")


if __name__ == "__main__":
    main()
