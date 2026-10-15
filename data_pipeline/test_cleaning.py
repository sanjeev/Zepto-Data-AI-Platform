"""Offline check that clean_books() survives messy rows instead of crashing.

Run: python data_pipeline/test_cleaning.py   (or: pytest data_pipeline)
"""

import pandas as pd

from pipeline import GBP_TO_INR, clean_books


def test_messy_rows_are_imputed_or_dropped():
    raw = pd.DataFrame(
        [
            # well-formed rows
            {"title": "A", "price": "£10.00", "star_rating": "Three", "availability": "In stock", "category": "X"},
            {"title": "B", "price": "£30.00", "star_rating": "Five", "availability": "In stock (3 available)", "category": "X"},
            {"title": "C", "price": "£20.00", "star_rating": "One", "availability": "Out of stock", "category": "Y"},
            # bad price -> median-imputed (median of 10, 12.5, 20, 30 = 16.25)
            {"title": "D", "price": "N/A", "star_rating": "Two", "availability": "In stock", "category": "Y"},
            # bad rating -> median-imputed (median of 3,5,1,2 = 2.5 -> round -> 2)
            {"title": "E", "price": "£12.50", "star_rating": "Seven", "availability": "In stock", "category": "Y"},
            # unparseable availability -> dropped
            {"title": "F", "price": "£5.00", "star_rating": "Four", "availability": "Call us", "category": "Y"},
            # missing title -> dropped
            {"title": None, "price": "£5.00", "star_rating": "Four", "availability": "In stock", "category": "Y"},
        ]
    )
    clean = clean_books(raw)

    assert list(clean["title"]) == ["A", "B", "C", "D", "E"]
    assert clean.loc[clean.title == "D", "price_gbp"].item() == 16.25
    assert clean.loc[clean.title == "E", "rating"].item() == 2
    assert clean["rating"].dtype == "int64"
    assert clean["in_stock"].dtype == bool
    assert not clean.loc[clean.title == "C", "in_stock"].item()
    assert (clean["price_inr"] == (clean["price_gbp"] * GBP_TO_INR).round(2)).all()


if __name__ == "__main__":
    test_messy_rows_are_imputed_or_dropped()
    print("test_messy_rows_are_imputed_or_dropped: OK")
