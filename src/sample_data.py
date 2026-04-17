"""Generate a small but realistic e-commerce SQLite database.

Tables: customers, products, orders, order_items.

The data is seeded so everyone who runs the app sees the same numbers,
which makes the demo predictable.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    region        TEXT    NOT NULL,
    signup_date   DATE    NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    price         REAL    NOT NULL
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    DATE    NOT NULL,
    status        TEXT    NOT NULL
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL
);
"""

REGIONS = ["North America", "Europe", "Asia", "South America", "Oceania"]
FIRST_NAMES = [
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Drew",
    "Jamie", "Avery", "Quinn", "Hayden", "Blake", "Cameron", "Reese",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson",
]

PRODUCTS = [
    ("Wireless Mouse",        "Electronics",  24.99),
    ("Mechanical Keyboard",   "Electronics", 129.00),
    ("USB-C Hub",             "Electronics",  39.99),
    ("4K Monitor",            "Electronics", 349.00),
    ("Laptop Stand",          "Office",       49.50),
    ("Ergonomic Chair",       "Office",      299.00),
    ("Standing Desk",         "Office",      499.00),
    ("Notebook (pack of 3)",  "Office",       14.99),
    ("Running Shoes",         "Apparel",      89.00),
    ("Hoodie",                "Apparel",      45.00),
    ("Baseball Cap",          "Apparel",      19.99),
    ("Yoga Mat",              "Fitness",      29.99),
    ("Dumbbell Set",          "Fitness",     119.00),
    ("Water Bottle",          "Fitness",      17.50),
    ("Coffee Beans (1kg)",    "Grocery",      22.00),
    ("Green Tea (100 bags)",  "Grocery",      12.50),
    ("Dark Chocolate Bar",    "Grocery",       4.50),
]

ORDER_STATUSES = ["completed", "completed", "completed", "completed",
                  "shipped", "shipped", "processing", "cancelled"]


def _random_date(start: date, end: date, rng: random.Random) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def build_sample_db(path: Path, seed: int = 42) -> None:
    """Create (or replace) a SQLite file at `path` with seeded demo data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)

        customers = []
        for i in range(1, 201):
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}{i}@example.com"
            region = rng.choice(REGIONS)
            signup = _random_date(date(2023, 1, 1), date(2024, 12, 1), rng)
            customers.append((i, name, email, region, signup.isoformat()))
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers
        )

        products = [(i + 1, *row) for i, row in enumerate(PRODUCTS)]
        conn.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)", products
        )

        orders = []
        order_items = []
        order_item_id = 1
        for order_id in range(1, 1201):
            customer = rng.choice(customers)
            customer_id = customer[0]
            signup = date.fromisoformat(customer[4])
            earliest = max(signup, date(2024, 1, 1))
            order_date = _random_date(earliest, date(2025, 3, 31), rng)
            status = rng.choice(ORDER_STATUSES)
            orders.append((order_id, customer_id, order_date.isoformat(), status))

            for _ in range(rng.randint(1, 4)):
                product = rng.choice(products)
                quantity = rng.randint(1, 3)
                unit_price = product[3]
                order_items.append(
                    (order_item_id, order_id, product[0], quantity, unit_price)
                )
                order_item_id += 1

        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?)", orders
        )
        conn.executemany(
            "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", order_items
        )
        conn.commit()
    finally:
        conn.close()


def ensure_sample_db(path: Path) -> Path:
    """Build the sample DB if it doesn't exist, and return its path."""
    path = Path(path)
    if not path.exists():
        build_sample_db(path)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample.db"
    build_sample_db(out)
    print(f"Wrote sample database to {out} at {datetime.now().isoformat(timespec='seconds')}")
