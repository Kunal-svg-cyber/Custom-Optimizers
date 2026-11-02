# Walk-forward study: Yahoo Finance via yfinance, 2008-01-01 to 2025-12-31

4528 days, 10 assets, window 756 days, refit every 21 days, 20 epochs of gradient budget per refit for the iterative methods, cost 5 bps per unit traded.

### Real run

| Method | days | mean daily rank IC | pooled IC | Sharpe (gross) | Sharpe (net of costs) | mean turnover |
|---|---|---|---|---|---|---|
| OLS (closed form) | 3749 | -0.0022 | +0.0187 | +0.15 | -1.21 | 1.09 |
| Ridge (validated) | 3749 | +0.0079 | +0.0292 | +0.31 | -0.76 | 0.90 |
| Adam, budget-limited | 3749 | -0.0018 | +0.0268 | -0.01 | -1.35 | 1.06 |
| Jacobi SVRG, budget-limited | 3749 | -0.0022 | +0.0187 | +0.15 | -1.21 | 1.09 |
| 21-day momentum (no fitting) | 3749 | -0.0018 | -0.0186 | +0.04 | -0.32 | 0.32 |

Paired differences against OLS (moving-block bootstrap over days, block 21, 95% interval):

| Method | mean daily IC difference | net Sharpe difference |
|---|---|---|
| Ridge (validated) | +0.0101 [+0.0012, +0.0188] | +0.45 [+0.20, +0.69] |
| Adam, budget-limited | +0.0004 [-0.0078, +0.0087] | -0.14 [-0.43, +0.18] |
| Jacobi SVRG, budget-limited | +0.0000 [-0.0000, +0.0001] | +0.00 [-0.00, +0.00] |
| 21-day momentum (no fitting) | +0.0004 [-0.0207, +0.0208] | +0.89 [+0.19, +1.60] |

Jacobi SVRG minus Adam, mean daily IC: -0.0004 [-0.0087, +0.0078].

### Placebo (circularly shifted targets: no signal by construction)

| Method | days | mean daily rank IC | pooled IC | Sharpe (gross) | Sharpe (net of costs) | mean turnover |
|---|---|---|---|---|---|---|
| OLS (closed form) | 3749 | -0.0053 | -0.0027 | -0.20 | -1.70 | 1.09 |
| Ridge (validated) | 3749 | -0.0091 | -0.0043 | -0.30 | -1.52 | 0.90 |
| Adam, budget-limited | 3749 | -0.0009 | -0.0014 | -0.08 | -1.52 | 1.06 |
| Jacobi SVRG, budget-limited | 3749 | -0.0053 | -0.0027 | -0.20 | -1.70 | 1.09 |
| 21-day momentum (no fitting) | 3749 | +0.0090 | +0.0012 | +0.36 | -0.10 | 0.32 |

Paired differences against OLS (moving-block bootstrap over days, block 21, 95% interval):

| Method | mean daily IC difference | net Sharpe difference |
|---|---|---|
| Ridge (validated) | -0.0038 [-0.0105, +0.0024] | +0.18 [-0.10, +0.42] |
| Adam, budget-limited | +0.0044 [-0.0045, +0.0122] | +0.18 [-0.18, +0.50] |
| Jacobi SVRG, budget-limited | -0.0000 [-0.0001, +0.0000] | -0.00 [-0.00, +0.00] |
| 21-day momentum (no fitting) | +0.0143 [-0.0005, +0.0307] | +1.60 [+0.92, +2.40] |

Jacobi SVRG minus Adam, mean daily IC: -0.0044 [-0.0122, +0.0045].
