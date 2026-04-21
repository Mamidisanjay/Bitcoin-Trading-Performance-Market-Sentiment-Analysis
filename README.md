# Bitcoin Trading Performance & Market Sentiment Analysis
## Primetrade.ai Data Science Assignment

### 📊 Project Overview

This comprehensive analysis explores the relationship between trader performance on the Hyperliquid platform and Bitcoin market sentiment (Fear & Greed Index). The project uncovers hidden patterns, quantifies sentiment impact on trading outcomes, and provides actionable insights for developing smarter trading strategies.

---

## 📁 Deliverables

### 1. **Visualizations** (PNG files)
- `trader_sentiment_analysis.png` - 8-panel comprehensive dashboard showing:
  - Market sentiment distribution over time
  - Average PnL by sentiment
  - Win rates across sentiment regimes
  - Trading volume analysis
  - Hourly performance patterns
  - Top traded assets
  - Buy vs Sell performance
  - Cumulative PnL trends

- `correlation_heatmap.png` - Correlation matrix showing relationships between:
  - Sentiment scores
  - Trading outcomes
  - Position sizes
  - Time of day
  - Trade profitability

- `trader_performance_details.png` - Deep dive into trader metrics:
  - Top 15 and bottom 15 traders
  - PnL distribution
  - Trade size distributions by sentiment

### 2. **Analysis Scripts**
- `analysis.py` - Core data processing and statistical analysis
- `visualizations.py` - Comprehensive visualization generation
- `generate_report.py` - Detailed insights report generator

### 3. **Reports**
- `detailed_insights_report.txt` - 200+ line comprehensive text report covering:
  - Executive summary
  - 5 major findings with quantified metrics
  - Hidden pattern detection
  - Strategic recommendations
  - Data quality notes

### 4. **Interactive Notebook**
- `trading_analysis_notebook.ipynb` - Fully executable Jupyter notebook with:
  - Step-by-step analysis workflow
  - Interactive visualizations
  - Code explanations
  - Key insights summary

---

## 🔍 Key Findings

### 1. **Sentiment Drives Performance**
- **Extreme Greed** periods show:
  - Highest average PnL: **$67.89** per trade
  - Best win rate: **46.49%**
  - Total profits: **$2.7M** across period
  
- **Extreme Fear** shows:
  - Lower average PnL: **$34.54**
  - Win rate: **37.06%**
  - Contrarian opportunities exist

### 2. **Strong Temporal Patterns**
- **Peak Performance Hour**: 12:00 UTC
  - Average PnL: **$131.17** (3x overall average)
  - Aligns with European market hours + US pre-market
  
- **Weakest Hours**: 14:00, 21:00-23:00 UTC
  - Significant performance drop-off
  - Suggests reduced liquidity or adverse selection

### 3. **Highly Concentrated Success**
- **90.6%** of traders are profitable (29 out of 32)
- Top 10 traders control **>50%** of total profits
- Top performer: **$2.14M** total PnL
- Median trader: Significantly lower profitability

### 4. **Asset Concentration**
- **HYPE token**: 32.2% of all trades (68,005 trades)
- Platform-native token dominates activity
- Top 5 assets: HYPE, @107, BTC, ETH, SOL

### 5. **Directional Bias by Sentiment**
- **Extreme Greed**: Long positions outperform
- **Extreme Fear**: Short positions show edge
- Clear regime-dependent optimal strategies

---

## 🎯 Strategic Recommendations

### For Traders
1. **Sentiment-Based Position Bias**
   - Go long during Extreme Greed
   - Consider shorts during Extreme Fear
   - Monitor sentiment transitions for entry/exit

2. **Optimal Timing**
   - Concentrate active trading: 11:00-13:00 UTC
   - Reduce exposure: 14:00, 21:00-23:00 UTC
   - Automate execution for optimal windows

3. **Risk Management**
   - Study top 10 trader patterns
   - Position sizing > trade frequency
   - Diversify beyond top 3 assets

### For Platform Development
1. **Algorithm Development**
   - Sentiment transition detection systems
   - Dynamic position sizing based on regime
   - Composite indicators (sentiment + time + volume)

2. **User Education**
   - Highlight temporal performance patterns
   - Provide sentiment-aware trading guidance
   - Share top performer insights (anonymized)

---

## 📈 Data Summary

- **Date Range**: May 1, 2023 - May 1, 2025 (2 years)
- **Total Trades**: 211,224
- **Unique Traders**: 32
- **Total Volume**: $1.09 Billion USD
- **Total PnL**: $10.25 Million
- **Sentiment Records**: 2,644 daily observations

---

## 🛠️ Technical Stack

- **Python 3.12**
- **Libraries**:
  - pandas - Data manipulation
  - numpy - Numerical computing
  - matplotlib - Static visualizations
  - seaborn - Statistical visualizations
  - scikit-learn - Advanced analytics

---

## 📊 How to Use

### View Static Results
1. Open the PNG visualization files directly
2. Read `detailed_insights_report.txt` for comprehensive analysis

### Run Interactive Analysis
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch Jupyter notebook
jupyter notebook trading_analysis_notebook.ipynb
```

### Regenerate Analysis
```bash
# Run core analysis
python analysis.py

# Generate visualizations
python visualizations.py

# Create report
python generate_report.py
```

---

## 🔬 Methodology

1. **Data Preprocessing**
   - Timestamp normalization to UTC
   - Missing value handling
   - Numerical type conversions

2. **Sentiment Merging**
   - Daily sentiment scores matched to trades
   - Forward-fill for weekend gaps

3. **Statistical Analysis**
   - Grouped aggregations by sentiment
   - Win rate calculations
   - Correlation analysis

4. **Pattern Detection**
   - Temporal decomposition
   - Trader clustering
   - Sentiment regime transitions

---

## 📝 Future Enhancements

1. **Machine Learning Models**
   - Predict trade outcomes using sentiment + features
   - Trader classification (strategies)
   - Optimal entry/exit timing models

2. **Real-Time Integration**
   - Live sentiment tracking
   - Automated trading signals
   - Performance monitoring dashboard

3. **Expanded Scope**
   - Multi-asset correlation analysis
   - Volatility regime integration
   - Order flow analysis

---

## 👤 Contact

**Submission for**: Primetrade.ai Data Science Position  
**Prepared by**: M.J.N.Sanjay  
**Date**: April 21 2026

---

## 📜 License

This analysis is submitted as part of a hiring assessment for Primetrade.ai. All rights reserved.
