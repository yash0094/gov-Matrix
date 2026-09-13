# Procurement Anomaly Detection — Build Spec

Companion to `procurement_anomaly_engine.py`. Scoped for a 4-person team with ~48 hours
before the 15 September Round 1 deadline.

---

## 0. Scope discipline

You cannot build the full architecture in two days. You do not need to. Round 1 is judged
on a deck and a video; the prototype is bonus marks under Section 8.6. So build the thin
vertical slice that makes the deck's three load-bearing claims *demonstrable*:

| Claim | Slide | What must actually run |
|---|---|---|
| Peer baselines stop false positives on specialised markets | 5 | Peer bucketing + one before/after case |
| The queue has a stated false-discovery rate | 7 | p-values → Stouffer → BH → q-values |
| We measured it against injected cartels | 9 | Red-team generator + precision@k |

Everything else — live scraping, entity resolution at scale, the feedback loop — goes on
the roadmap slide as a dated phase. Judges reward stated limitations. Do not fake depth.

**Cut order if you run out of time:** live data ingest → network signal → auth → styling.
Never cut the red-team validation. It is the only thing on your slide that no competing
team will have.

---

## 1. Tech stack

```
Data          Python 3.11 · pandas · numpy · scipy · scikit-learn · networkx
              rapidfuzz (entity resolution) · duckdb (analytics) · pyarrow
Store         PostgreSQL 16 (or DuckDB single-file for the 48h build)
API           FastAPI + Pydantic v2 + uvicorn
Frontend      React 18 + Vite + TypeScript + TailwindCSS
              Recharts (peer-comparison charts) · Cytoscape.js (entity graph)
              TanStack Query + TanStack Table (server-side sort/filter)
Deploy        Docker Compose. Frontend → Vercel, API → Railway/Render, DB → Neon
Repo          Public GitHub for the whole evaluation window. Private = all bonus forfeited.
```

**Two-day substitution:** skip Postgres entirely. Use DuckDB over Parquet. It reads Parquet
natively, does window functions and joins at Postgres speed on a laptop, ships as a single
file with no server, and the SQL below runs on it essentially unchanged. Migrate later if
you make Round 2.

**Do not** reach for XGBoost, an LLM, or an autoencoder. You have no labels. An
unsupervised deep model gives you a number you cannot explain to a vigilance officer,
which fails the problem statement's own explainability requirement and hands the judges an
easy question you cannot answer. Classical statistics is the *stronger* choice here and you
should say so out loud on slide 6.

---

## 2. Datasets — named, verified, downloadable

Slide 9 asks you to name sources so feasibility reads as verified, not assumed. These are
real and open.

**Primary — use this one.** <cite index="14-1">Assam State Government Finance Department publishes contracting data in OCDS format, transformed and published in collaboration with CivicDataLab</cite>. <cite index="2-1">The dataset downloads as JSON, Excel or CSV, either for a specific year or for all time, with each contracting process as one line of JSON in a gzipped .jsonl file</cite>. Available from the OCP Data Registry at `data.open-contracting.org/en/publication/131`, and mirrored on `data.gov.in` under the Assam Public Procurement Data catalogue. Licence permits commercial use with attribution — which matters for your monetisation slide.

Be aware going in: <cite index="2-1">the publisher documents known quality issues including invalid codes in tender.status and tender.procurementMethod, incorrect date formats, and milestones without ids</cite>. Budget half a day for cleaning and put it on the roadmap slide as a named constraint.

**Secondary.** <cite index="12-1">CivicDataLab's Himachal Pradesh health procurement repository parses data from FY 2017-18 onward into OCDS, built by extracting awards data from the national eprocure.gov.in platform and joining tender details from hptenders.gov.in</cite>. Gives you a second state for the "each new state is data ingest, not new engineering" claim.

**Explorers worth citing on the problem slide.** `assam.open-contracting.in` (Assam Public Procurement Explorer) and the Himachal Health Procurement Performance Index. Useful for your "existing tools stop at descriptive dashboards; nobody quantifies surprise relative to peers" argument on slide 3.

**Entity resolution sources.** MCA21 company master data on data.gov.in for CIN, registered address and director DIN. GSTIN structure gives you state and PAN linkage for free.

**Reality check on bidder-level data.** Full losing-bid rosters are the single scarcest field in Indian procurement data — most portals publish the award, not every quote. <cite index="4-1">Georgia's Tender Monitor is the standard counterexample, publishing extensive bidder information that is typically sparse among contracting datasets</cite>. Consequences you should plan around:

- Signals that need full bid vectors (`dispersion`, `margin`, `network` co-bidding lift) degrade where only the winner is published.
- Signals that need only awards (`price`, `price_hed`, `competition`, `rotation`, HHI, CUSUM) work everywhere.
- **Say this on the roadmap slide.** Naming a data limitation and showing your architecture degrades gracefully reads as engineering maturity. Pretending it doesn't exist reads as a team that never opened the file.

**Synthetic backstop.** `make_universe()` in the engine generates a full 4,000-tender universe with bids, directors and injected cartels, so your demo runs with zero external dependencies. Use it for validation numbers; use real Assam data for the screenshots.

---

## 3. Database schema

```sql
-- ---------- canonical entities (post entity-resolution) ----------
CREATE TABLE entity (
    entity_id       BIGSERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    cin             TEXT,                    -- MCA company identifier
    gstin           TEXT,
    pan             TEXT,
    entity_type     TEXT CHECK (entity_type IN ('buyer','vendor','both')),
    state_code      TEXT,
    first_seen      DATE,
    last_seen       DATE
);

CREATE TABLE entity_alias (              -- "ABC Traders Pvt Ltd" / "A.B.C. Traders"
    alias_id        BIGSERIAL PRIMARY KEY,
    entity_id       BIGINT REFERENCES entity(entity_id),
    raw_name        TEXT NOT NULL,
    source_portal   TEXT,
    match_score     REAL,                    -- rapidfuzz token_set_ratio
    match_method    TEXT,                    -- 'exact_cin' | 'fuzzy_name' | 'addr+phone'
    reviewed        BOOLEAN DEFAULT FALSE
);
CREATE INDEX ON entity_alias (lower(raw_name));

CREATE TABLE entity_link (               -- the graph: shared directors/addresses/contacts
    link_id         BIGSERIAL PRIMARY KEY,
    entity_a        BIGINT REFERENCES entity(entity_id),
    entity_b        BIGINT REFERENCES entity(entity_id),
    link_type       TEXT,                    -- 'shared_director'|'shared_address'|'shared_phone'
    link_value      TEXT,                    -- the DIN / normalised address hash
    confidence      REAL,
    CHECK (entity_a < entity_b)              -- undirected, stored once
);
CREATE INDEX ON entity_link (entity_a);
CREATE INDEX ON entity_link (entity_b);

-- ---------- contracting processes ----------
CREATE TABLE tender (
    ocid                TEXT PRIMARY KEY,     -- OCDS contracting process id
    buyer_id            BIGINT REFERENCES entity(entity_id),
    title               TEXT,
    category_raw        TEXT,
    category_norm       TEXT NOT NULL,        -- your own taxonomy, drives peer groups
    region              TEXT NOT NULL,
    fiscal_year         INT  NOT NULL,
    procurement_method  TEXT,
    published_value     NUMERIC,              -- PRE-AWARD estimate. peer banding uses THIS
    quantity            NUMERIC,
    uom                 TEXT,
    bid_open_date       DATE,
    bid_close_date      DATE,
    bid_window_days     INT GENERATED ALWAYS AS (bid_close_date - bid_open_date) STORED,
    n_bidders           INT,
    source_portal       TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON tender (category_norm, region, fiscal_year);

CREATE TABLE bid (
    bid_id          BIGSERIAL PRIMARY KEY,
    ocid            TEXT REFERENCES tender(ocid),
    vendor_id       BIGINT REFERENCES entity(entity_id),
    bid_amount      NUMERIC,
    unit_price      NUMERIC,
    rank_in_tender  INT,
    is_winner       BOOLEAN,
    disqualified    BOOLEAN DEFAULT FALSE,
    UNIQUE (ocid, vendor_id)
);
CREATE INDEX ON bid (vendor_id);

CREATE TABLE award (
    award_id        BIGSERIAL PRIMARY KEY,
    ocid            TEXT REFERENCES tender(ocid),
    vendor_id       BIGINT REFERENCES entity(entity_id),
    award_value     NUMERIC,
    award_date      DATE,
    final_paid      NUMERIC,                  -- for Benford + variation analysis
    n_variations    INT DEFAULT 0             -- Round 2 carry-over hook
);

-- ---------- analytics layer ----------
CREATE TABLE peer_group (
    peer_group_id   TEXT PRIMARY KEY,         -- 'L0::cement|VB2|KA|2025'
    level           TEXT,                     -- L0..L3, widening hierarchy
    category_norm   TEXT,
    value_band      TEXT,
    region          TEXT,
    fiscal_year     INT,
    n_members       INT,
    confidence      REAL,                     -- 1.00 / 0.85 / 0.70 / 0.50
    median_unit_price NUMERIC,
    mad_unit_price    NUMERIC,
    median_n_bidders  REAL,
    computed_at     TIMESTAMPTZ
);

CREATE TABLE signal_score (
    ocid            TEXT REFERENCES tender(ocid),
    signal_name     TEXT,
    raw_value       DOUBLE PRECISION,
    peer_group_id   TEXT REFERENCES peer_group(peer_group_id),
    z_score         DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    model_version   TEXT,
    PRIMARY KEY (ocid, signal_name, model_version)
);

CREATE TABLE case_alert (
    case_id         BIGSERIAL PRIMARY KEY,
    buyer_id        BIGINT REFERENCES entity(entity_id),
    category_norm   TEXT,
    n_tenders       INT,
    case_value      NUMERIC,
    z_case          DOUBLE PRECISION,
    p_case          DOUBLE PRECISION,
    q_value         DOUBLE PRECISION,         -- BH-adjusted; ship this, not a hard cutoff
    ripple_value    NUMERIC,
    ripple_contracts INT,
    status          TEXT DEFAULT 'new'
                    CHECK (status IN ('new','reviewed','explained','escalated')),
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON case_alert (q_value);

CREATE TABLE reviewer_feedback (              -- closes the Step 7 loop
    feedback_id     BIGSERIAL PRIMARY KEY,
    case_id         BIGINT REFERENCES case_alert(case_id),
    reviewer_id     TEXT,
    verdict         TEXT CHECK (verdict IN ('reviewed','explained','escalated')),
    dominant_signal TEXT,                     -- which signal drove the alert
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

**The one schema decision worth defending live:** `published_value`, not `award_value`, in
peer banding. Award value is downstream of the manipulation you are hunting — an inflated
award gets pushed into a higher value band where the peer median is higher, so the
inflation partly cancels itself and your price signal goes quiet. You would be conditioning
on the treatment. Same discipline as never neutralising a factor against anything computed
from forward returns. I hit this bug building the engine; the fix moved tender-level
precision@50 from 28% to 56%.

**Signal weights belong in config, not code.** Round 2 swaps the problem while keeping the
domain. Keep signal definitions pluggable and the schema generic over *entity*,
*transaction* and *peer group* — the playbook is right that ~70% transfers, but only if you
write it that way now.

---

## 4. API contract

```
GET  /api/cases?q_max=0.2&region=KA&category=cement&sort=z_case&page=1
     -> paginated case queue + {stated_fdr, n_cases, expected_false_discoveries}

GET  /api/cases/{case_id}
     -> full evidence card: signal contributions (exact, additive),
        peer comparison series, entity subgraph, counter-evidence list

GET  /api/cases/{case_id}/peer-comparison
     -> {this_value, peer_median, peer_iqr, peer_distribution[], peer_group_meta}

GET  /api/cases/{case_id}/network
     -> {nodes:[{id,name,type,cin}], edges:[{source,target,type,confidence}]}

GET  /api/cases/{case_id}/ripple
     -> {clusters, connected_contracts, connected_value_inr, contracts[]}

POST /api/cases/{case_id}/feedback
     body {verdict, notes} -> 201, enqueues signal re-weighting

GET  /api/validation
     -> red-team results: precision@k, recall@k, ROC-AUC, injection config

GET  /api/public/contestability?tender_id=...     # free MSME tier, no auth
     -> {was_contestable: bool, peer_bidder_median, this_bidder_count, reasons[]}
```

Keep `/api/validation` public and linked from the deck. A live endpoint serving your
precision numbers is worth more than a screenshot of them.

---

## 5. Dashboard requirements

### 5.1 Case queue (landing)
- Server-side sorted table: case, buyer, category, n_tenders, value, Z, **q-value**, status.
- **A persistent banner stating the FDR at the current filter.** "34 cases shown at
  q ≤ 0.20 — expect ~7 to be false alarms." This single element is your differentiator made
  visible. Most teams will show a list of alerts with no error rate anywhere.
- A q-value slider, not a fixed alpha. Let the investigator set their own tolerance.
- Peer-confidence chip on every row (L0 full / L3 widened-and-weak). Never hide that an
  alert came from a thin bucket.

### 5.2 Evidence card (the demo centrepiece)
Four panels, in this order, because it mirrors how a reviewer actually reasons:

1. **Contribution bar chart.** Horizontal, signed, sorted by magnitude, labelled in plain
   language ("winning price 38% above comparable awards in Karnataka cement, 2025"), with
   the % of total evidence per signal. Exact, not approximated — say that on the slide.
2. **Peer comparison.** Distribution of the peer group with this tender marked. Include the
   peer group definition as a subtitle — the reviewer must be able to challenge the
   comparison set. Toggle to show the same tender against the *national* distribution,
   where it looks normal. That toggle is your slide 5 before/after in interactive form.
3. **Entity subgraph.** Cytoscape, nodes = vendors/directors/addresses, edges typed and
   confidence-weighted. Click a node to see its other contracts.
4. **Counter-evidence panel.** Deliberately surfaced innocent explanations, styled with
   equal visual weight to the evidence — not greyed out in a footnote. This is the honesty
   that the problem statement is actually testing for.

Plus a fixed non-accusation notice on every card, and Reviewed / Explained / Escalated
buttons that POST feedback.

### 5.3 Ripple view
Sankey or radial from the flagged cluster to connected contracts, with a running rupee
total. One headline number: *"reviewing this one cluster puts ₹246 crore across 422
contracts in scope."* Put that number in your video's closing line.

### 5.4 Public contestability check (free tier)
Single input, single verdict, no login. An MSME pastes a tender ID and gets "this tender
had 2 bidders; comparable tenders in this category averaged 7." Small to build, and it
makes inclusivity a named feature on slide 11 rather than a footnote.

### 5.5 Validation page
Red-team config, precision@k table, ROC curve, and the injection patterns you designed.
Publishing your own evaluation methodology is a credibility move.

**Accessibility, since SDG 16 judges notice:** never encode a verdict in colour alone —
pair every red/amber with an icon and a text label.

---

## 6. Repo layout

```
procurement-anomaly/
├── README.md                 # architecture diagram + 5-minute quickstart
├── docker-compose.yml
├── data/
│   ├── raw/                  # gitignored
│   ├── interim/              # parquet
│   └── external/mca_directors.csv
├── src/
│   ├── ingest/               # ocds_parser.py, portal_scraper.py, normalise.py
│   ├── entity/               # fuzzy_match.py, union_find.py, graph_build.py
│   ├── peers/                # bucketing.py, hierarchy.py
│   ├── signals/              # price.py bidding.py process.py network.py
│   │                         #   each exposes compute(df) -> Series. PLUGGABLE.
│   ├── combine/              # pvalues.py stouffer.py fdr.py
│   ├── validate/             # redteam.py metrics.py
│   └── api/                  # FastAPI app
├── frontend/
├── notebooks/                # 01_eda … 04_validation (judges do open these)
├── tests/
└── config/signals.yaml       # weights live here, not in code
```

---

## 7. Forty-eight hours, four people

| | Person A (data) | Person B (stats) | Person C (frontend) | Person D (deck/video) |
|---|---|---|---|---|
| **Sat AM** | Assam OCDS dump → Parquet, schema mapped | Run the engine on synthetic, tune weights | Vite scaffold, queue table on mock JSON | Deck skeleton in the official template |
| **Sat PM** | Name normalisation + director join | Engine on real Assam data | Evidence card: contribution chart + peer chart | Slides 1–4, script draft |
| **Sun AM** | Peer buckets persisted, export API JSON | Red-team run, final precision@k numbers | Network graph + ripple view | Slides 5–9 with real numbers |
| **Sun PM** | **Freeze.** Deploy, verify links | Validation page JSON | Deploy frontend, screenshot for deck | Record video, export PDF |
| **Mon AM** | Compliance sweep with D | Sanity-check every number on the deck | Repo public, README | **Submit with hours to spare** |

**Submission traps from Section 7 of the playbook, worth re-reading at 2am:** PDF only
(.pptx is rejected); the video link must open in an incognito window with no sign-in;
filename exactly `TeamID_TeamName_ProblemStatementID`; no institute name, logo or campus
background anywhere in the deck, the video, *or the GitHub repo*; repo public for the whole
evaluation window. Check the repo — a university email in a commit author field or a
college logo in a README is an easy own goal.

---

## 8. What to say when a judge pushes

**"How do you know it isn't noise?"** — Every signal is converted to an empirical p-value
against its own peer null, combined with a correlation-corrected Stouffer Z, and the queue
is Benjamini-Hochberg controlled. We publish a q-value per case. At q ≤ 0.10 our red-team
queue had a realised false-discovery rate of 0%.

**"Why not deep learning?"** — There are no labels, so any supervised model is untrainable
and any unsupervised one is unexplainable. A vigilance officer cannot act on a number they
cannot interrogate. Our combination is a weighted sum, so each signal's contribution is
exact arithmetic rather than a SHAP approximation.

**"Won't you flag every niche market?"** — That's the failure mode the problem statement
names, and it's why peer conditioning is Layer 1 rather than an afterthought. In our
validation, specialised categories averaging 1.6 bidders produced zero flagged cases, while
a naive "fewer than 3 bidders" rule would have flagged all of them.

**"What if the cartel is big enough to move the baseline?"** — It is, and that's the honest
limit. We use median/MAD rather than mean/std, which has a 50% breakdown point, so the
baseline survives contamination up to half the peer group. Beyond that we would need an
external reference price. It's on the roadmap.
