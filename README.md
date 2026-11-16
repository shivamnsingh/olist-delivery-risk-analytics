# Olist Sentinel: E-Commerce Performance and Delivery Risk Analytics

Predicting which orders are likely to arrive late, using the public Olist Brazilian e-commerce dataset (9 relational tables, about 100,000 orders).

## The problem

About 8% of delivered orders arrive after the delivery date promised to the customer. An operations team cannot check every order, so this project ranks orders by late-delivery risk and measures how many late orders are found if only the riskiest 10% are reviewed.

## Main result

| Measure | Result |
|---|---|
| Delivered orders analysed | 96,470 |
| Late-delivery rate | 8.11% (7,826 late orders) |
| Best model | Logistic regression |
| Average PR-AUC lift over random ranking | 2.00x (5 rolling time-based test windows) |
| Late orders found in the riskiest 10% of orders | about 23% (random selection finds about 10%) |

Logistic regression performed slightly better than XGBoost on average, on both the rolling backtest and the final holdout. "Lift" is the model's precision-recall score divided by the late rate, so 1.0x means no better than random.

### Rolling backtest (train on the past, test on the next 2 months)

| Test window | Late rate | Logistic lift | XGBoost lift | Logistic top-10% capture | XGBoost top-10% capture |
|---|---:|---:|---:|---:|---:|
| from 2017-10 | 11% | 1.37 | 1.26 | 17% | 15% |
| from 2017-12 | 7% | 1.34 | 1.60 | 16% | 18% |
| from 2018-02 | 19% | 1.68 | 1.56 | 20% | 18% |
| from 2018-04 | 7% | 2.43 | 2.17 | 29% | 26% |
| from 2018-06 | 3% | 3.18 | 3.23 | 31% | 26% |
| **Average** | **9%** | **2.00** | **1.97** | **23%** | **21%** |

Performance is strongest in calm periods and weakest during disruption months, because one-off events are hard to predict from order details.

### Final holdout (train on the earliest 80% of orders, test on the latest 20%)

| Model | Lift | Top-10% capture | Top-10% precision |
|---|---:|---:|---:|
| Logistic regression | 1.76 | 20.7% | 10.9% |
| XGBoost | 1.39 | 15.8% | 8.3% |

## Key findings from the analysis

- The late rate swings strongly by month, from 1.4% in June 2018 to a peak of 21.4% in March 2018, so lateness is driven largely by time-specific events.
- On-time delivery rate across delivered orders: 91.89%.
- Because month-to-month swings dominate, the calendar month was deliberately **not** used as a feature (the data covers under two years, so the model would only memorise specific months).

## Approach

1. **Target:** `is_late = 1` if the delivery date is after the estimated delivery date. Only delivered orders with a known delivery date are used.
2. **One row per order:** items are aggregated (total price, total freight, item count) and joined to customer, seller and product details.
3. **Features (13), all known at purchase time:**
   - Order and product: purchase hour, day of week, price, freight, item count, product weight, product volume, promised delivery days
   - Geography: shipping distance between seller and customer (haversine on average zip-prefix coordinates) and a same-state flag
   - History: the seller's past late rate and order count, and the platform's recent late rate over the last 1,000 delivered orders
4. **Avoiding leakage:** history features are joined backwards in time, so an order only sees orders already delivered before its own purchase time. Every test set is later in time than its training set.
5. **Evaluation:** a rolling time-based backtest and a chronological holdout, using lift and top-10% capture instead of accuracy, because only about 8% of orders are late.
6. **Models:** class-balanced logistic regression (baseline) and a deliberately small, regularised XGBoost.
7. **Explanation:** SHAP summary plot for the XGBoost model.

## Project structure

```
olist-delivery-risk-analytics/
  data/              the 9 Olist CSV files (not included in the repo)
  ipynb_files/
    late_delivery_prediction.ipynb   main model, backtest and SHAP
    code.ipynb                       business KPIs and monthly analysis
    basic.ipynb                      first exploration
  data_overview.md   column inventory of the tables
  requirements.txt
  README.md
```

## How to run

1. Download the dataset from Kaggle ("Brazilian E-Commerce Public Dataset by Olist") and put the CSV files in a folder named `data/` at the top of the project.
2. Install the libraries: `pip install -r requirements.txt`
3. Open `ipynb_files/late_delivery_prediction.ipynb` in Jupyter or VS Code and choose Run All.

The notebooks read the data from `../data`, so keep the folder layout above.

## Limitations

- Orders with several items use the first seller and first product only.
- Distance is based on zip-prefix averages, not exact addresses, and the duplicate rows in the geolocation table were not removed.
- Only delivered orders are used, so canceled and unfinished orders are not covered.
- The backtest windows share training data, there are no confidence intervals, and no statistical test compares the two models.
- The SHAP plot explains the XGBoost model, which is not the best-performing model.
- This is a ranking and prioritisation result only. No intervention was tested, so the project does not measure reduced late deliveries, cost savings or customer-satisfaction gains.

## Possible next steps

- Feature-group ablation to measure how much seller history, distance and promised days each contribute.
- Compare capture at the top 5%, 10% and 20% of orders.
- Explain the logistic regression model directly with standardised coefficients.
- Test a real intervention (for example expedited handling) before claiming business impact.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, Jupyter.
