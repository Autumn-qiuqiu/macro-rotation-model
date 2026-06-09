# US Macro-Driven Sector Rotation Model
A machine learning model that predicts which S&P 500 sector will 
outperform the market based on macroeconomic indicators.

**Live Demo:** https://macro-rotation-model-jkcgrsjbc5v6xf6k2t9iv5.streamlit.app/

## What it does
- Fetches macroeconomic data (Fed Funds Rate, CPI, Unemployment, 
  Consumer Sentiment) from FRED API
- Downloads historical sector ETF data via yfinance
- Trains a Random Forest classifier to predict sector outperformance
- Backtests strategy performance vs buy-and-hold and SPY benchmark

## Key Results
- Best performing sector: XLI (Industrial), 75.7% prediction accuracy
- Sharpe Ratio: 1.73 (vs buy-and-hold)
- Market exposure: only 27% of the time, reducing drawdown risk
- Key finding: Consumer Sentiment is a stronger predictor than 
  interest rate changes for industrial sector performance

## Tech Stack
Python | pandas | yfinance | pandas_datareader | 
scikit-learn | plotly

## Data Sources
- Macroeconomic data: FRED (Federal Reserve Economic Data)
- Sector ETF prices: Yahoo Finance

## How to run
pip install yfinance pandas_datareader scikit-learn plotly
