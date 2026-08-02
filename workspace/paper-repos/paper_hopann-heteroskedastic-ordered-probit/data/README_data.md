# Data README — Amazon Software Product Reviews

## Dataset Required

This project uses the **Amazon Software Product Reviews** dataset from:

> Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews
> and fine-grained aspects. EMNLP-IJCNLP 2019.

The paper uses a filtered subset: **6,173 reviews** for **929 software items** from **328 brands**,
collected from **January 1, 2018 to September 26, 2018**.

---

## Download Instructions

1. Visit the UCSD Amazon Review Data page:
   https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

2. Download the **Software** category (5-core):
   ```
   wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Software_5.json.gz
   ```

3. Run the preprocessing script to apply the paper's filtering and feature engineering:
   ```bash
   python data/preprocess.py \
     --raw-path data/Software_5.json.gz \
     --output-path data/amazon_software_reviews.csv \
     --start-date 2018-01-01 \
     --end-date 2018-09-26
   ```

---

## Expected Output Structure

After preprocessing, `data/amazon_software_reviews.csv` should contain:

| Column | Type | Description |
|---|---|---|
| `overall` | int (1-5) | Star rating — dependent variable |
| `price` | float | Software price (USD) |
| `asin_n_rev` | int | Prior reviews of this product before t_i |
| `asin_n_reviewers` | int | Prior reviewers of this product before t_i |
| `asin_verified_share` | float [0,1] | Share of purchase-verified prior reviews |
| `user_n_rev` | int | Reviewer's prior reviews before t_i |
| `user_reviewed_asin` | int | Reviewer's prior reviewed software before t_i |
| `day_*` | int (0/1) | Day-of-week binary dummies (Saturday = base group) |
| `timestamp` | int | Unix timestamp of review |
| `reviewer_id` | str | Unique reviewer identifier |
| `asin` | str | Product identifier |

Expected row count: **~6,173** (one review per consumer, first review only).

---

## Notes

- The paper uses **one review per consumer** (no duplication of reviewer_id).
- Data is **sorted by timestamp** before splitting.
- The train/val/test split is **time-ordered** to prevent data leakage.
- Standardisation is fit on the **training set only** and applied to val/test.
- `asin_verified_share` and day dummy columns are **NOT standardised**.

---

## Synthetic Data (for smoke tests)

If the real dataset is unavailable, you can generate a synthetic version:

```bash
python data/preprocess.py --synthetic --n-samples 500 --output-path data/amazon_software_reviews.csv
```

This produces a correctly-formatted CSV with random data for testing the pipeline.
