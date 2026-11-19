"""Build the business SQLite DB from the real Olist dataset (stdlib only).

The Support Agent's tools (order status / logistics / refund) query this DB.
Full 99k real orders — SQLite handles it locally without any extra dependency.

Usage:
    uv run python scripts/build_business_db.py              # full dataset
    uv run python scripts/build_business_db.py --max-orders 10000   # subset

Output: data/processed/business.db + business_db_report.json (row counts +
order-status distribution — the "quality report" the eval report cites).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "olist"
OUT_DIR = ROOT / "data" / "processed"
DB_PATH = OUT_DIR / "business.db"

SCHEMA = """
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS order_payments;
DROP TABLE IF EXISTS order_reviews;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS product_category_translation;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    zip_code_prefix TEXT,
    city TEXT,
    state TEXT
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT NOT NULL,
    order_purchase_timestamp TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE TABLE order_items (
    order_id TEXT NOT NULL,
    order_item_id INTEGER,
    product_id TEXT,
    seller_id TEXT,
    price REAL,
    freight_value REAL
);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE TABLE order_payments (
    order_id TEXT NOT NULL,
    payment_sequential INTEGER,
    payment_type TEXT,
    payment_installments INTEGER,
    payment_value REAL
);
CREATE INDEX idx_pay_order ON order_payments(order_id);
CREATE TABLE order_reviews (
    review_id TEXT,
    order_id TEXT NOT NULL,
    review_score INTEGER,
    review_comment_title TEXT,
    review_comment_message TEXT
);
CREATE INDEX idx_reviews_order ON order_reviews(order_id);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_weight_g REAL,
    product_photos_qty INTEGER
);
CREATE TABLE product_category_translation (
    category_pt TEXT PRIMARY KEY,
    category_en TEXT
);
"""

LOADERS = [
    (
        "customers",
        "olist_customers_dataset.csv",
        ["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
        "INSERT INTO customers VALUES (?,?,?,?)",
    ),
    (
        "orders",
        "olist_orders_dataset.csv",
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
    ),
    (
        "order_items",
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
        "INSERT INTO order_items VALUES (?,?,?,?,?,?)",
    ),
    (
        "order_payments",
        "olist_order_payments_dataset.csv",
        ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
        "INSERT INTO order_payments VALUES (?,?,?,?,?)",
    ),
    (
        "order_reviews",
        "olist_order_reviews_dataset.csv",
        ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message"],
        "INSERT INTO order_reviews VALUES (?,?,?,?,?)",
    ),
    (
        "products",
        "olist_products_dataset.csv",
        ["product_id", "product_category_name", "product_weight_g", "product_photos_qty"],
        "INSERT INTO products VALUES (?,?,?,?)",
    ),
    (
        "product_category_translation",
        "product_category_name_translation.csv",
        ["product_category_name", "product_category_name_english"],
        "INSERT INTO product_category_translation VALUES (?,?)",
    ),
]


def load_table(
    conn: sqlite3.Connection,
    name: str,
    fname: str,
    fields: list[str],
    insert: str,
    order_filter: set[str] | None,
) -> int:
    path = RAW / fname
    if not path.exists():
        print(f"[warn] missing {fname} — skipping {name}")
        return 0
    rows = 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for rec in reader:
            if order_filter is not None:
                oid = rec.get("order_id")
                if oid is not None and oid not in order_filter:
                    continue
            batch.append(tuple(rec.get(k) or None for k in fields))
            if len(batch) >= 5000:
                conn.executemany(insert, batch)
                rows += len(batch)
                batch = []
        if batch:
            conn.executemany(insert, batch)
            rows += len(batch)
    print(f"[ok]   {name}: {rows} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-orders", type=int, default=0, help="keep only first N orders (0 = all)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orders_csv = RAW / "olist_orders_dataset.csv"
    if not orders_csv.exists():
        print("[error] olist data missing — run scripts/download_data.py first")
        return 2

    order_filter: set[str] | None = None
    if args.max_orders:
        with orders_csv.open(encoding="utf-8") as f:
            order_filter = {
                r["order_id"] for r in itertools.islice(csv.DictReader(f), args.max_orders)
            }
        print(f"[info] subsetting to {len(order_filter)} orders")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    counts = {}
    for name, fname, fields, insert in LOADERS:
        counts[name] = load_table(conn, name, fname, fields, insert, order_filter)

    statuses = Counter(r[0] for r in conn.execute("SELECT order_status FROM orders"))
    report = {"db": str(DB_PATH), "row_counts": counts, "order_status_distribution": dict(statuses)}
    (OUT_DIR / "business_db_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2)[:600])
    conn.commit()
    conn.close()
    print(f"[done] {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
