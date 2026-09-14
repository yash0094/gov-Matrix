# Procurement Anomaly Detection

Statistical screening for public procurement data. Ingests OCDS tenders, scores them against
peer-conditioned baselines across seven signals, and returns a **short, FDR-controlled queue of
buyerâ€“category cases** worth a human investigator's time.

No black box. No labels required. Every score decomposes into the signals that produced it.

---

## Why statistics, not machine learning

There is no labelled corpus of "confirmed cartel" tenders. Supervised ML on unlabelled data
means inventing a proxy target, and the proxy becomes the thing you detect.

So the engine takes the classical route used in competition-authority screening:

1. Build a **peer group** for each tender (buyer type Ã— category Ã— value band Ã— period).
2. Compute each signal as a **p-value against that peer group**, not a global distribution.
3. Combine correlated p-values with **Brown-corrected Stouffer's Z**.
4. Roll tenders up to **buyerâ€“category cases** and apply **Benjaminiâ€“Hochberg FDR control**
   at the case level.

The output is a ranked queue with an explicit false-discovery budget. An investigator working
the top 20 cases knows roughly what fraction are expected to be noise.

---

## The seven signals

| Family | Signal | Intuition |
|---|---|---|
| Price | Unit-price deviation | Award price sits far above peer-group distribution for comparable line items |
| Price | Estimate-to-award inflation | Award consistently lands above the published estimate in a way peers don't show |
| Bidding | Bid-spread compression | Losing bids cluster too tightly around the winner â€” cover bidding signature |
| Bidding | Win-share concentration | A small supplier set splits wins in a way random allocation wouldn't produce |
| Process | Single-bid / thin-competition rate | Repeated lots attracting one qualified bidder above peer base rate |
| Process | Submission-window compression | Tender open for an unusually short period relative to value and category |
| Network | Co-bidding affinity | The same suppliers appear together across lots more often than chance predicts |

Weights live in `config/weights.yaml`. Change the policy, not the code.

---

## Three bugs that mattered

Documenting these because they are the transferable part of the work.

**1. Leakage through peer-group construction.**
Peer bands were originally cut on *award* value. Award value is an outcome â€” banding on it
leaks the answer into the comparison set and quietly flattens the very signal you want.
Rebanding on **pre-award published/estimated value** moved precision@50 from ~28% â†’ ~56%.

**2. FDR applied at the wrong granularity.**
Benjaminiâ€“Hochberg was running over individual tenders. With thousands of tenders and diffuse
evidence, nothing survived correction and the queue came back empty. Cartels are a *relationship*
between a buyer and a supplier set over time, so the unit of inference is the
**buyerâ€“category case**. Correcting at case level produced a usable queue immediately.

**3. Thin-evidence inflation.**
Several signals were structurally loudest where denominators were smallest â€” three tenders in a
niche category could max out a ratio. Explicit small-sample handling (shrinkage toward the peer
prior + minimum-evidence gates) removed a whole class of false alarms.

---

## Validation

Synthetic generator produces realistic procurement structure with **three injected cartels**
across 4,000 tenders.

| Metric | Result |
|---|---|
| Case-level precision@3 | 100% |
| Tender-level ROC-AUC | 0.89 |
| False positives on specialised low-competition markets | 0 |

That last row is the one that matters operationally. Defence, medical equipment and specialist
civil works are *legitimately* low-competition. A screener that flags them is a screener nobody
uses twice.

Synthetic validation is a substitute for ground truth, not a claim of field accuracy.

---

## Quickstart

```bash
# one command
./run.sh

# or with docker
docker compose up --build
```

Then open `http://localhost:8000` for the dashboard and `http://localhost:8000/docs`
for the interactive API reference.

Local install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate          # build synthetic dataset + cache
python -m src.score             # run the scoring pipeline
uvicorn src.api:app --reload    # serve API + dashboard
```

---

## Repository layout

```
â”œâ”€â”€ config/
â”‚   â”œâ”€â”€ weights.yaml         # signal weights
â”‚   â””â”€â”€ peers.yaml           # peer-group policy (banding, min group size)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ generate.py          # synthetic data generator with injectable cartels
â”‚   â”œâ”€â”€ parse_ocds.py        # OCDS release â†’ internal tender schema
â”‚   â”œâ”€â”€ peers.py             # peer-group construction (pre-award value banding)
â”‚   â”œâ”€â”€ signals/             # the seven signal implementations
â”‚   â”œâ”€â”€ combine.py           # Brown-corrected Stouffer's Z
â”‚   â”œâ”€â”€ fdr.py               # Benjaminiâ€“Hochberg, case-level scope
â”‚   â”œâ”€â”€ score.py             # pipeline entrypoint
â”‚   â””â”€â”€ api.py               # FastAPI service
â”œâ”€â”€ dashboard/
â”‚   â””â”€â”€ index.html           # zero-build single file, Chart.js via CDN
â”œâ”€â”€ tests/                   # regression suite
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ docker-compose.yml
â””â”€â”€ run.sh
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/cases` | FDR-controlled case queue, ranked |
| GET | `/cases/{case_id}` | Case detail with signal decomposition |
| GET | `/tenders` | Paginated tender list with scores |
| GET | `/tenders/{tender_id}` | Single tender, peer group, per-signal p-values |
| GET | `/buyers/{buyer_id}` | Buyer profile and history |
| GET | `/suppliers/{supplier_id}` | Supplier profile, co-bidding neighbours |
| GET | `/stats` | Corpus-level summary for dashboard charts |

---

## Configuration

`config/peers.yaml` controls the comparison logic:

```yaml
peer_group:
  keys: [buyer_type, category_code, value_band, fiscal_period]
  value_band:
    source: published_value     # NEVER award_value â€” see bug #1
    edges: [0, 1e5, 1e6, 1e7, 1e8]
  min_group_size: 12            # below this, shrink toward category prior

fdr:
  alpha: 0.10
  scope: case                   # buyer Ã— category, NOT tender â€” see bug #2
```

`source: published_value` is load-bearing. Do not change it to an award-time field.

---

## Data

Built against the **Open Contracting Data Standard (OCDS)**. The parser targets the
Assam and Himachal Pradesh procurement releases; other OCDS publishers should work with
field-mapping adjustments in `parse_ocds.py`.

Point at your own data:

```bash
python -m src.parse_ocds --input data/releases/ --output data/cache/
python -m src.score --source cache
```

---

## Tests

```bash
pytest -q
```

Eight regression tests cover peer-group determinism, leakage guard (asserts award fields never
reach banding), FDR scope, Stouffer's correlation correction, and small-sample shrinkage.
They exist because rapid iteration on statistical code silently breaks things that still run.

---

## Limitations

- Validated on synthetic data. Real-world precision is unknown.
- Signals are **screening indicators, not evidence**. A flagged case means "look here", never
  "this is collusion". Output should never be published as an accusation.
- Network signals need reasonable supplier-identifier hygiene; fragmented entity names degrade
  the co-bidding signal.
- Peer groups thin out fast in specialised categories. Min-group gates reduce false alarms but
  also create blind spots.

## Roadmap

- Entity resolution for supplier names
- Temporal change-point detection on win shares
- Investigator feedback loop to calibrate weights
- Cross-buyer bidder-network view in the dashboard
