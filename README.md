# Short-Horizon Volatility Forecasting from Order-Book Data

Predicting each stock's realised volatility over the next 10 minutes from the previous 10 minutes of limit order book and trade data, using Optiver's [Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) dataset (Kaggle, 2021).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joannapeniket/optiver-volatility/blob/main/optiver_volatility.ipynb)

## Results

Held-out test set: 20% of time windows, never seen during training or model selection. Metric: RMSPE (root mean squared percentage error).

| Model | Test RMSPE | vs naive |
|---|---|---|
| Naive persistence (next vol = last vol) | 0.3423 | — |
| Naive, scaled by one fitted constant (×0.837) | 0.2821 | −17.6% |
| Linear regression in logs, 5 features | 0.2582 | −24.6% |
| **LightGBM, all features** | **0.2235** | **−34.7%** |

## Approach

- **Data:** 112 stocks, 3,830 ten-minute windows, 428,932 (stock, window) samples, with level-1 and level-2 order book snapshots and executed trades.
- **Features:** 19 per (stock, window), built from the first 10 minutes only:
  - realised volatility of the weighted average price (WAP) over the full window, the last 5 minutes and the last 2.5 minutes
  - bid-ask spread, book depth and size imbalance
  - trade activity
  - market-wide averages across all stocks in the same window
- **Split:** 70/10/20 train/validation/test, **grouped by time window**. The same window across different stocks shares market conditions, so splitting by row would leak the test period into training.
- **Metric-aligned training:** LightGBM is trained with sample weights of 1/y², which makes the weighted squared error equal to the squared percentage error, so the model optimises RMSPE directly.

## Key findings

- **Volatility clustering dominates.** Log volatility in consecutive windows has a correlation of 0.91, and recent-volatility features carry about 80% of LightGBM's gain importance.
- **Volatility mean-reverts.** The next window is about 16% calmer on average, and the log-linear elasticities on the stock's own volatility sum to about 0.80.
- **The market regime matters non-linearly.** Market-wide features are the most valuable group in the ablation (removing them costs +0.0054 RMSPE), yet their coefficient in the linear model is close to zero.
- **Spread, depth, imbalance and trade features added nothing measurable** beyond quote-based realised volatility.
- **The model shrinks towards typical volatility.** It over-predicts the calmest decile of windows by 13% and under-predicts the most volatile by 13%. The calmest windows dominate the error.

The full write-up, with ablations and error analysis, is at the end of the notebook.

## Limitations

- Optiver shuffled the time windows, so this is not a walk-forward (past → future) test.
- Results come from a single split with no confidence intervals, and there was no hyperparameter search beyond early stopping.
- Scores are not directly comparable to the competition leaderboard, which used a different, hidden test set.

## Running it

1. Accept the competition rules on Kaggle and download the data. It is not included here, because of the competition terms and its size.
2. Put `train.csv`, `book_train.parquet/` and `trade_train.parquet/` in a Google Drive folder called `optiver`.
3. Open the notebook in Colab (CPU runtime) and select **Run all**. Feature extraction takes about 1–2 minutes; the full notebook takes about 15–30 minutes.

**Stack:** Python, pandas, NumPy, LightGBM, scikit-learn, matplotlib, Google Colab
