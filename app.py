import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

fred = Fred(api_key='')

st.title("美股宏观驱动板块轮动模型")
st.markdown("基于美联储利率、CPI、失业率、消费者信心预测下月最强板块")

@st.cache_data
def load_data():
    start = "2005-01-01"
    end = datetime.today().strftime('%Y-%m-%d')
    
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY"]
    sector_data = yf.download(sectors, start=start, end=end, auto_adjust=True)["Close"]
    
    macro_data = pd.DataFrame({
        "联邦基金利率": fred.get_series('FEDFUNDS', start, end),
        "CPI": fred.get_series('CPIAUCSL', start, end),
        "失业率": fred.get_series('UNRATE', start, end),
        "消费者信心指数": fred.get_series('UMCSENT', start, end)
    })
    
    sector_monthly = sector_data.resample("MS").first()
    sector_returns = sector_monthly.pct_change()
    combined = macro_data.join(sector_returns, how="inner").dropna()
    return combined

st.info("正在加载数据，请稍候...")
combined = load_data()
st.success("数据加载完成！")

feature_data = pd.DataFrame(index=combined.index)
feature_data["rate_change"] = combined["联邦基金利率"].diff()
feature_data["rate_level"] = combined["联邦基金利率"]
feature_data["cpi_change"] = combined["CPI"].pct_change()
feature_data["cpi_level"] = combined["CPI"]
feature_data["unemployment_change"] = combined["失业率"].diff()
feature_data["unemployment_level"] = combined["失业率"]
feature_data["sentiment_change"] = combined["消费者信心指数"].diff()
feature_data["sentiment_level"] = combined["消费者信心指数"]
feature_data = feature_data.dropna()
combined = combined.loc[feature_data.index]

st.subheader("当前宏观环境")
col1, col2, col3, col4 = st.columns(4)
col1.metric("联邦基金利率", f"{combined['联邦基金利率'].iloc[-1]:.2f}%")
col2.metric("CPI", f"{combined['CPI'].iloc[-1]:.1f}")
col3.metric("失业率", f"{combined['失业率'].iloc[-1]:.1f}%")
col4.metric("消费者信心", f"{combined['消费者信心指数'].iloc[-1]:.1f}")

st.subheader("下月板块预测")
sector_cols = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY"]
sector_names = {
    "XLE": "能源", "XLF": "金融", "XLI": "工业",
    "XLK": "科技", "XLP": "消费必需", "XLV": "医疗", "XLY": "可选消费"
}

proba_results = {}
for sector in sector_cols:
    y = (combined[sector].shift(-1) > combined[sector].mean()).astype(int)
    y = y.loc[feature_data.index].dropna()
    X = feature_data.loc[y.index]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    proba = model.predict_proba(feature_data.iloc[[-1]])[:, 1][0]
    proba_results[sector] = proba

best_sector = max(proba_results, key=proba_results.get)
st.success(f"推荐超配板块：{sector_names[best_sector]} ({best_sector})")

fig = go.Figure(go.Bar(
    x=list(sector_names.values()),
    y=[proba_results[s] for s in sector_cols],
    marker_color=['green' if s == best_sector else 'steelblue' for s in sector_cols]
))
fig.update_layout(title="各板块跑赢概率", yaxis_title="概率", height=400)
st.plotly_chart(fig)

st.subheader("回测表现")

split = int(len(feature_data) * 0.8)
X_train = feature_data.iloc[:split]
X_test = feature_data.iloc[split:]
best_sector = "XLI"

y_train = (combined[best_sector].shift(-1) > combined[best_sector].mean()).astype(int).loc[X_train.index].dropna()
X_train = X_train.loc[y_train.index]

model_bt = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_bt.fit(X_train, y_train)

proba_bt = model_bt.predict_proba(X_test)[:, 1]
print(f"概率范围: {proba_bt.min():.3f} - {proba_bt.max():.3f}")
threshold = float(pd.Series(proba_bt).quantile(0.6))
pred_bt = pd.Series((proba_bt > threshold).astype(int), index=X_test.index)
test_returns = combined[best_sector].loc[X_test.index]

strategy_returns = test_returns * pred_bt
cum_strategy = (1 + strategy_returns).cumprod()
cum_buyhold = (1 + test_returns).cumprod()

active = strategy_returns[pred_bt == 1]
sharpe = active.mean() / active.std() * (12**0.5) if len(active) > 1 else 0

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cum_strategy.index, y=cum_strategy.values, name="宏观轮动策略", line=dict(color="green", width=2)))
fig2.add_trace(go.Scatter(x=cum_buyhold.index, y=cum_buyhold.values, name="买入持有XLI", line=dict(color="blue", width=2)))
fig2.update_layout(title="累计收益对比（样本外测试）", height=400)
st.plotly_chart(fig2)

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("策略累计收益", f"{cum_strategy.iloc[-1]:.2f}x")
col3.metric("持仓月数", f"{pred_bt.sum()} / {len(pred_bt)}")