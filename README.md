# US Macro-Driven Sector Rotation Model

**Live Demo:** https://macro-rotation-model-jkcgrsjbc5v6xf6k2t9iv5.streamlit.app/

A machine learning model that predicts which S&P 500 sector will 
outperform the market based on macroeconomic indicators.

## What it does
- Fetches macroeconomic data (Fed Funds Rate, CPI, Unemployment, 
  Consumer Sentiment) from FRED API
- Downloads historical sector ETF data via yfinance (XLK/XLF/XLE/XLV/XLI/XLP/XLY)
- Trains a Random Forest classifier to predict sector outperformance
- Backtests strategy with transaction costs vs buy-and-hold and SPY benchmark
- Serves real-time predictions via interactive Streamlit dashboard

## Key Results (Out-of-Sample Backtest 2022–2026)
- Sharpe Ratio: 2.16
- Max Drawdown: -7.8%
- Strategy Cumulative Return: 1.68x (vs buy-and-hold)
- Market Exposure: only 37% of the time, reducing drawdown risk
- Best predicted sector: XLI (Industrial), 75.7% accuracy
- Key finding: Consumer Sentiment and CPI change are stronger 
  predictors than interest rate direction for industrial sector performance

## Tech Stack
Python | pandas | yfinance | fredapi | scikit-learn | plotly | streamlit

## Data Sources
- Macroeconomic data: FRED (Federal Reserve Economic Data)
- Sector ETF prices: Yahoo Finance

## Roadmap
- [ ] Add FinBERT sentiment analysis as real-time signal
- [ ] Fama-French factor comparison

## How to run
pip install yfinance fredapi scikit-learn plotly streamlit
streamlit run app.py
