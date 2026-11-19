# Data directory

**Nothing under `data/raw/` or `data/processed/` is committed to git.**
All datasets are fetched by `uv run python scripts/download_data.py`, which
records provenance into `data/manifest.json` (source URL, license, download
date, size, sha256).

## Sources

| Dataset | Role | Source | License |
|---|---|---|---|
| Olist Brazilian E-commerce (100k real orders, 9 tables) | Business DB for the support agent's tools (orders / logistics / payments / reviews) | Kaggle `olistbr/brazilian-ecommerce` | CC BY-NC-SA 4.0 |
| Amazon QA (Electronics subset) | Real customer-question distribution: eval calibration, few-shot, FAQ corpus | UCSD McAuley Lab (public direct download) | public academic dataset (cite ICDM 2016 / WWW 2016) |
| GitLab Handbook (HR/IT pages) | Knowledge corpus for the internal assistant | gitlab.com open-source repo | CC BY-SA 4.0 |
| FastAPI docs (markdown source) | Developer-docs knowledge corpus | github.com/fastapi/fastapi | MIT |
| CN consumer-protection regulations | Chinese after-sales policy corpus | gov.cn public texts | official public documents |

Every processed document keeps a provenance triple (source URL, license,
fetched-at date) so any retrieved chunk can be traced back — this is part of the
project's "real data, verifiable claims" guarantee.

## Setup

```bash
# 1. Kaggle token (for Olist): put kaggle.json in ~/.kaggle/ or set
#    KAGGLE_USERNAME / KAGGLE_KEY in .env
# 2. Fetch everything:
uv run python scripts/download_data.py
```
