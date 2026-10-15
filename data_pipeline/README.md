# Module 1 — Data Pipeline (`/data_pipeline`)

Scrape → clean → convert → store → query, on [books.toscrape.com](https://books.toscrape.com/)
(a public scraping-practice site: no login, no key, no paid tier).

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | The whole pipeline, end to end (scrape, clean, INR conversion, SQLite load, SQL queries, pandas read-back + merge check) |
| `test_cleaning.py` | Offline test feeding deliberately messy rows to the cleaner (bad price, bad rating, unknown availability, missing title) |
| `books.db` | The SQLite database produced by the last run (regenerated from scratch on every run) |
| `outputs/query_results.md` | Every SQL query string with its output, the `pd.read_sql` read-backs and the side-by-side `pd.read_sql` vs `pd.merge` join comparison |
| `outputs/run_log.txt` | Full console log of the last run |
| `outputs/books_raw.csv`, `outputs/books_clean.csv` | Raw scraped text and cleaned/typed data |

## Install & run

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -r data_pipeline/requirements.txt
python data_pipeline/pipeline.py          # needs internet (scrapes the site); ~30 s
python data_pipeline/test_cleaning.py     # offline messy-row test
```

`pipeline.py` drops and recreates both tables each run, so `books.db` is always reproducible from scratch.

## What it does

1. **Scrape** (`requests` + `BeautifulSoup`): four categories — Mystery (32), Historical Fiction (26),
   Science Fiction (16), Travel (11) — following each category's `next` pagination link.
   **85 books across 4 categories** (requirement: ≥ 60 across ≥ 3). For each book: `title`
   (from the link's `title` attribute, because the visible text is truncated), `price` (e.g. `£51.77`),
   `star_rating` (the CSS class, e.g. `Three`), `availability` text, and `category`. Requests use a
   session, a timeout, explicit `status_code == 200` checks and up to 3 retries with back-off.
2. **Clean** (`clean_books`):
   - `price_gbp` (float): regex-extracts the number, so the `£` symbol (or a mis-decoded `Â£`) is stripped.
     The HTML is parsed from raw bytes so the page's own UTF-8 declaration is honoured.
   - `rating` (int 1–5): `One…Five` mapped through a dictionary.
   - `in_stock` (bool): text starting with `In stock` → `True`, `Out of stock` → `False`.
3. **Convert**: `price_inr = round(price_gbp × 105.50, 2)` using the project's fixed baseline rate
   **1 GBP = 105.50 INR** — an artificial, project-defined constant, not a live or historical market rate,
   so there is no API call, no network access and no date reference. (The optional live-API stretch was not attempted.)
4. **Store**: normalized SQLite schema with a PK/FK relationship:

   ```sql
   CREATE TABLE categories (
       category_id   INTEGER PRIMARY KEY,
       category_name TEXT NOT NULL UNIQUE
   );
   CREATE TABLE books (
       book_id     INTEGER PRIMARY KEY,
       title       TEXT    NOT NULL,
       price_gbp   REAL    NOT NULL,
       price_inr   REAL    NOT NULL,
       rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
       in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
       category_id INTEGER NOT NULL REFERENCES categories(category_id)
   );
   ```

   Inserted with `sqlite3` (`executemany`) with `PRAGMA foreign_keys = ON`. `in_stock` is stored as 0/1
   because SQLite has no boolean type.
5. **Query**: six SQL queries (full text + output in [`outputs/query_results.md`](outputs/query_results.md)):

   | # | Query | Clauses demonstrated |
   |---|---|---|
   | 1 | 10 most expensive in-stock books | SELECT, WHERE, ORDER BY, LIMIT |
   | 2 | Distinct star ratings | DISTINCT, ORDER BY |
   | 3 | Books priced £20–£30 | WHERE … BETWEEN |
   | 4 | Cheapest 4- or 5-star books | WHERE … IN, ORDER BY, LIMIT |
   | 5 | Top-3 highest-rated books per category | **JOIN** + `ROW_NUMBER() OVER (PARTITION BY …)` |
   | 6 | Per-category count / avg price / avg rating | **JOIN**, GROUP BY, aggregates |

6. **pandas**: queries 1, 5 and 6 are read back with `pd.read_sql(...)`. Query 5 (the JOIN) is then rebuilt
   with **no SQL** — `pd.merge(books, categories, on="category_id")` → sort by rating desc, price desc,
   book_id → `groupby("category_id").head(3)` — and checked with `pd.testing.assert_frame_equal`
   (passes). Both versions are printed side by side in `outputs/query_results.md` and the run log.

## Cleaning decisions (messy rows)

- **Numeric fields (`price_gbp`, `rating`) that fail to parse → median imputation.** One bad price or
  rating should not cost the whole row; the median is robust to the right-skewed price distribution, and
  the imputed rating is rounded back to an integer so it stays on the 1–5 scale.
- **Non-numeric fields (`title`, `category`, `in_stock`) that fail to parse → drop the row.** There is no
  meaningful "median" title or stock flag, and guessing stock status would publish false availability to analysts.
- Duplicate `(title, category)` pairs are dropped.
- The live site happens to be perfectly clean, so the actual run imputed 0 values and dropped 0 rows
  (logged as `Cleaning: dropped 0 row(s) … median-imputed 0 price(s) and 0 rating(s)`).
  `test_cleaning.py` proves the policy on synthetic messy rows: a bad price is imputed with the median,
  an unknown rating word is imputed, and rows with unparseable availability or a missing title are dropped,
  without the pipeline crashing.
- Every book on the site's listing pages is "In stock", so `in_stock` is `True` for all 85 rows; the parser
  still handles "Out of stock" (covered in the test).
