
# Olist E-commerce Delivery Risk & Customer Experience Analytics

An end-to-end analytics project on the public Olist Brazilian e-commerce dataset. It combines KPI reporting, a leakage-aware late-delivery risk model, review-score diagnostics, and Portuguese review text mining to answer one question: **where do deliveries go wrong, and what does it cost in customer experience?**

![Olist ER Diagram](<Olist%20E-commerce%20ER%20Diagram.png>)

---

## Key Results

| Area                                    | Result                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Delivered orders analysed               | 96,470 (8.1% arrived late)                                                                             |
| On-time delivery rate                   | 91.89%                                                                                                 |
| Average review score                    | 4.09 / 5                                                                                               |
| Customer repeat rate                    | 3.12%                                                                                                  |
| Average order value                     | R$160.99                                                                                               |
| Peak monthly late rate                  | 21.4% (Mar 2018), 16.0% (Feb 2018)                                                                     |
| Review score, on-time vs late           | 4.29 vs 2.57 (gap of 1.73 points)                                                                      |
| Seller concentration                    | Top 10% of sellers account for 73% of late orders                                                      |
| Late-risk model (rolling backtest)      | Logistic regression: 2.00x PR-AUC lift, 23% of late orders captured in the riskiest 10% (random = 10%) |
| Late-risk model (chronological holdout) | Logistic regression: 1.76x lift, 20.7% top-10% capture; XGBoost: 1.39x lift, 15.8%                     |
| Negative review classifier              | 0.92 accuracy, 0.867 F1 and 0.92 recall on the negative class                                          |
| Negative review themes                  | Delivery-related keywords appear in 5,780 of 10,889 negative reviews (53.08%)                          |

---

## Notebooks

| Notebook                                       | Purpose                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `basic.ipynb`                                | Dataset inventory: file list, row counts, schemas, and table grain                                 |
| `olist_ecommerce_performance_analysis.ipynb` | KPI scorecard, monthly revenue and late-rate trends, review vs delivery link, seller concentration |
| `late_delivery_prediction.ipynb`             | Late-delivery risk model: feature engineering, rolling backtest, holdout, SHAP, seller drill-down  |
| `review_text_classification.ipynb`           | TF-IDF sentiment classifier and NMF complaint-theme modelling on Portuguese reviews                |

---

## Approach

### 1. Data inventory

Nine CSV files (customers, geolocation, orders, order items, payments, reviews, products, sellers, category translation). Row counts and schemas are checked first because the tables sit at different grains and must be aggregated before joining.

### 2. Delivery outcome

An order is **late** when the customer delivery date is after the estimated delivery date. Only delivered orders with a recorded delivery timestamp are used. `delay_days` is kept for analysis only and is never used as a model feature.

### 3. Late-delivery risk model

Framed as a **risk-ranking problem**: which orders should operations review first?

- **One row per order:** item rows are aggregated (price and freight summed, item count), then joined to customer, seller, and product data. For multi-item orders, the first seller and product are used.
- **Features:** purchase hour and weekday, freight, price, item count, product weight and volume, seller-to-customer haversine distance, same-state flag, promised delivery days, seller historical late rate and order count, and a rolling platform-wide late rate.
- **Leakage controls:** seller and platform history are built with `merge_asof` (backward, no exact matches), so only previously delivered orders contribute. `month` is deliberately excluded because under two years of data would let the model memorise specific months.
- **Evaluation:** a rolling time-based backtest (five two-month windows) plus a chronological 80/20 holdout. Metrics are PR-AUC lift and top-decile capture, since accuracy is uninformative at an 8% positive rate.
- **Models:** class-weighted logistic regression (baseline) vs regularised XGBoost, with SHAP used to interpret the XGBoost model.

### 4. Performance and experience diagnostics

KPI scorecard (revenue, AOV, on-time rate, review score, repeat rate), monthly late-rate trend, review score comparison for late vs on-time orders, and seller-level concentration of late orders.

### 5. Review text mining

- 40,950 Portuguese reviews with comments; neutral 3-star reviews excluded; scores 1-2 labelled negative.
- TF-IDF (unigrams and bigrams, Portuguese stop words, `min_df=5`) with class-balanced logistic regression on a stratified 80/20 split.
- NMF (6 components) on negative reviews to surface complaint themes: non-receipt, delays, missing items, and defective or wrong products.

---

## Model Results

**Rolling backtest (average across 5 windows)**

| Model               | PR-AUC lift | Top-10% capture |
| ------------------- | ----------- | --------------- |
| Logistic regression | 2.00x       | 23%             |
| XGBoost             | 1.97x       | 21%             |

**Chronological holdout (late rate 5.3%)**

| Model               | PR-AUC lift | Top-10% capture | Top-10% precision |
| ------------------- | ----------- | --------------- | ----------------- |
| Logistic regression | 1.76x       | 20.7%           | 10.9%             |
| XGBoost             | 1.39x       | 15.8%           | 8.3%              |

The signal is real but moderate: roughly 2x better than random ranking. The simpler logistic regression performed better than XGBoost on this data.

---

## Limitations

- Findings are **associations, not causal effects**. No intervention was tested, so no cost savings or reduction in late deliveries is claimed.
- Seller concentration is based on **late-order counts, not rates**, so high-volume sellers rank high partly due to volume.
- Delivery distance uses **zip-prefix average coordinates**, not exact addresses.
- Multi-item orders use only the **first seller and product**.
- The delivery-complaint share is **keyword-based** and may overcount; a single review can mention several issues.
- In the backtest, training orders purchased before a cutoff may have been delivered after it, so the pipeline is best described as leakage-aware rather than leakage-proof.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, NLTK, matplotlib, Jupyter.

---

## Getting Started

```bash
git clone <your-repo-url>
cd olist-delivery-risk-analytics
pip install -r requirements.txt
```

1. Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place the CSV files in the `data/` folder.
2. Open the notebooks in `ipynb_files/` and run them in this order:
   1. `basic.ipynb`
   2. `olist_ecommerce_performance_analysis.ipynb`
   3. `late_delivery_prediction.ipynb`
   4. `review_text_classification.ipynb`
      NLTK Portuguese stop words are downloaded automatically by the text classification notebook.

---

## Project Structure

```
olist-delivery-risk-analytics/
├── data/                              # Olist CSV files
├── ipynb_files/                       # Analysis notebooks
├── Olist E-commerce ER Diagram.png    # Dataset entity-relationship diagram
├── requirements.txt                   # Python dependencies
├── .gitattributes
└── README.md
```

---

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): roughly 100K orders placed between 2016 and 2018. Refer to the Kaggle page for licence terms.

# Olist E-commerce Delivery Risk & Customer Experience Analytics

An end-to-end analytics project on the public Olist Brazilian e-commerce dataset. It combines KPI reporting, a leakage-aware late-delivery risk model, review-score diagnostics, and Portuguese review text mining to answer one question: **where do deliveries go wrong, and what does it cost in customer experience?**

![Olist ER Diagram](<Olist%20E-commerce%20ER%20Diagram.png>)

---

## Key Results

| Area                                    | Result                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Delivered orders analysed               | 96,470 (8.1% arrived late)                                                                             |
| On-time delivery rate                   | 91.89%                                                                                                 |
| Average review score                    | 4.09 / 5                                                                                               |
| Customer repeat rate                    | 3.12%                                                                                                  |
| Average order value                     | R$160.99                                                                                               |
| Peak monthly late rate                  | 21.4% (Mar 2018), 16.0% (Feb 2018)                                                                     |
| Review score, on-time vs late           | 4.29 vs 2.57 (gap of 1.73 points)                                                                      |
| Seller concentration                    | Top 10% of sellers account for 73% of late orders                                                      |
| Late-risk model (rolling backtest)      | Logistic regression: 2.00x PR-AUC lift, 23% of late orders captured in the riskiest 10% (random = 10%) |
| Late-risk model (chronological holdout) | Logistic regression: 1.76x lift, 20.7% top-10% capture; XGBoost: 1.39x lift, 15.8%                     |
| Negative review classifier              | 0.92 accuracy, 0.867 F1 and 0.92 recall on the negative class                                          |
| Negative review themes                  | Delivery-related keywords appear in 5,780 of 10,889 negative reviews (53.08%)                          |

---

## Notebooks

| Notebook                                       | Purpose                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `basic.ipynb`                                | Dataset inventory: file list, row counts, schemas, and table grain                                 |
| `olist_ecommerce_performance_analysis.ipynb` | KPI scorecard, monthly revenue and late-rate trends, review vs delivery link, seller concentration |
| `late_delivery_prediction.ipynb`             | Late-delivery risk model: feature engineering, rolling backtest, holdout, SHAP, seller drill-down  |
| `review_text_classification.ipynb`           | TF-IDF sentiment classifier and NMF complaint-theme modelling on Portuguese reviews                |

---

## Approach

### 1. Data inventory

Nine CSV files (customers, geolocation, orders, order items, payments, reviews, products, sellers, category translation). Row counts and schemas are checked first because the tables sit at different grains and must be aggregated before joining.

### 2. Delivery outcome

An order is **late** when the customer delivery date is after the estimated delivery date. Only delivered orders with a recorded delivery timestamp are used. `delay_days` is kept for analysis only and is never used as a model feature.

### 3. Late-delivery risk model

Framed as a **risk-ranking problem**: which orders should operations review first?

- **One row per order:** item rows are aggregated (price and freight summed, item count), then joined to customer, seller, and product data. For multi-item orders, the first seller and product are used.
- **Features:** purchase hour and weekday, freight, price, item count, product weight and volume, seller-to-customer haversine distance, same-state flag, promised delivery days, seller historical late rate and order count, and a rolling platform-wide late rate.
- **Leakage controls:** seller and platform history are built with `merge_asof` (backward, no exact matches), so only previously delivered orders contribute. `month` is deliberately excluded because under two years of data would let the model memorise specific months.
- **Evaluation:** a rolling time-based backtest (five two-month windows) plus a chronological 80/20 holdout. Metrics are PR-AUC lift and top-decile capture, since accuracy is uninformative at an 8% positive rate.
- **Models:** class-weighted logistic regression (baseline) vs regularised XGBoost, with SHAP used to interpret the XGBoost model.

### 4. Performance and experience diagnostics

KPI scorecard (revenue, AOV, on-time rate, review score, repeat rate), monthly late-rate trend, review score comparison for late vs on-time orders, and seller-level concentration of late orders.

### 5. Review text mining

- 40,950 Portuguese reviews with comments; neutral 3-star reviews excluded; scores 1-2 labelled negative.
- TF-IDF (unigrams and bigrams, Portuguese stop words, `min_df=5`) with class-balanced logistic regression on a stratified 80/20 split.
- NMF (6 components) on negative reviews to surface complaint themes: non-receipt, delays, missing items, and defective or wrong products.

---

## Model Results

**Rolling backtest (average across 5 windows)**

| Model               | PR-AUC lift | Top-10% capture |
| ------------------- | ----------- | --------------- |
| Logistic regression | 2.00x       | 23%             |
| XGBoost             | 1.97x       | 21%             |

**Chronological holdout (late rate 5.3%)**

| Model               | PR-AUC lift | Top-10% capture | Top-10% precision |
| ------------------- | ----------- | --------------- | ----------------- |
| Logistic regression | 1.76x       | 20.7%           | 10.9%             |
| XGBoost             | 1.39x       | 15.8%           | 8.3%              |

The signal is real but moderate: roughly 2x better than random ranking. The simpler logistic regression performed better than XGBoost on this data.

---

## Limitations

- Findings are **associations, not causal effects**. No intervention was tested, so no cost savings or reduction in late deliveries is claimed.
- Seller concentration is based on **late-order counts, not rates**, so high-volume sellers rank high partly due to volume.
- Delivery distance uses **zip-prefix average coordinates**, not exact addresses.
- Multi-item orders use only the **first seller and product**.
- The delivery-complaint share is **keyword-based** and may overcount; a single review can mention several issues.
- In the backtest, training orders purchased before a cutoff may have been delivered after it, so the pipeline is best described as leakage-aware rather than leakage-proof.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, NLTK, matplotlib, Jupyter.

---

## Getting Started

```bash
git clone <your-repo-url>
cd olist-delivery-risk-analytics
pip install -r requirements.txt
```

1. Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place the CSV files in the `data/` folder.
2. Open the notebooks in `ipynb_files/` and run them in this order:
   1. `basic.ipynb`
   2. `olist_ecommerce_performance_analysis.ipynb`
   3. `late_delivery_prediction.ipynb`
   4. `review_text_classification.ipynb`
      NLTK Portuguese stop words are downloaded automatically by the text classification notebook.

---

## Project Structure

```
olist-delivery-risk-analytics/
├── data/                              # Olist CSV files
├── ipynb_files/                       # Analysis notebooks
├── Olist E-commerce ER Diagram.png    # Dataset entity-relationship diagram
├── requirements.txt                   # Python dependencies
├── .gitattributes
└── README.md
```

---

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): roughly 100K orders placed between 2016 and 2018. Refer to the Kaggle page for licence terms.
