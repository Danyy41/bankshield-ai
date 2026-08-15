# BankShield AI -- Phase 1 EDA Summary

- Total transactions: 50,000
- Fraudulent transactions: 745 (1.49%)
- Legitimate transactions: 49,255
- Missing values: none

## Fraud rate by risk factor

- `new_device`: False=1.31%  True=6.31%
- `new_beneficiary`: False=0.88%  True=9.93%
- `country_mismatch`: False=1.22%  True=6.61%
- `is_night`: False=1.41%  True=3.47%

## Fraud rate by merchant category

- `wire_transfer`: 11.48%
- `cash_withdrawal`: 7.63%
- `crypto_exchange`: 7.58%
- `gambling`: 3.13%
- `travel`: 2.30%
- `electronics`: 1.33%
- `online_retail`: 0.69%
- `healthcare`: 0.62%
- `grocery`: 0.61%
- `restaurant`: 0.55%
- `fuel`: 0.52%
- `utilities`: 0.50%
- `entertainment`: 0.50%

## Figures

- `reports/figures/class_balance.png`
- `reports/figures/amount_distribution.png`
- `reports/figures/fraud_rate_by_hour.png`
- `reports/figures/fraud_rate_by_category.png`
- `reports/figures/risk_factor_lift.png`
- `reports/figures/correlation_matrix.png`
