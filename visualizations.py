import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("Loading datasets...")
fear_greed = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')
historical = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')

# Data preprocessing
fear_greed['date'] = pd.to_datetime(fear_greed['date'])
historical['Timestamp'] = pd.to_datetime(historical['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
historical['date'] = historical['Timestamp'].dt.normalize()

historical['Closed PnL'] = pd.to_numeric(historical['Closed PnL'], errors='coerce')
historical['Size USD'] = pd.to_numeric(historical['Size USD'], errors='coerce')
historical['Execution Price'] = pd.to_numeric(historical['Execution Price'], errors='coerce')

merged = historical.merge(fear_greed[['date', 'classification', 'value']], on='date', how='left')
merged['hour'] = merged['Timestamp'].dt.hour
merged['day_of_week'] = merged['Timestamp'].dt.day_name()

print("Creating visualizations...")

# Create a comprehensive dashboard
fig = plt.figure(figsize=(20, 24))

# 1. Market Sentiment Distribution Over Time
ax1 = plt.subplot(4, 2, 1)
sentiment_over_time = fear_greed.groupby(['date', 'classification']).size().unstack(fill_value=0)
sentiment_over_time_resampled = sentiment_over_time.resample('ME').sum()
sentiment_over_time_resampled.plot(kind='area', stacked=True, ax=ax1, alpha=0.7)
ax1.set_title('Market Sentiment Distribution Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Days')
ax1.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Average PnL by Market Sentiment
ax2 = plt.subplot(4, 2, 2)
sentiment_pnl = merged.groupby('classification')['Closed PnL'].mean().sort_values()
colors = ['#d62728' if x < 0 else '#2ca02c' for x in sentiment_pnl.values]
sentiment_pnl.plot(kind='barh', ax=ax2, color=colors)
ax2.set_title('Average PnL by Market Sentiment', fontsize=14, fontweight='bold')
ax2.set_xlabel('Average Closed PnL ($)')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(sentiment_pnl.values):
    ax2.text(v, i, f' ${v:.2f}', va='center', fontweight='bold')

# 3. Win Rate by Sentiment
ax3 = plt.subplot(4, 2, 3)
win_rate = merged.groupby('classification').apply(
    lambda x: (x['Closed PnL'] > 0).mean() * 100
).sort_values(ascending=False)
win_rate.plot(kind='bar', ax=ax3, color='steelblue')
ax3.set_title('Win Rate (%) by Market Sentiment', fontsize=14, fontweight='bold')
ax3.set_ylabel('Win Rate (%)')
ax3.set_xlabel('')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(win_rate.values):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# 4. Trading Volume by Sentiment
ax4 = plt.subplot(4, 2, 4)
volume_by_sentiment = merged.groupby('classification')['Size USD'].sum() / 1e6
volume_by_sentiment.sort_values(ascending=True).plot(kind='barh', ax=ax4, color='coral')
ax4.set_title('Total Trading Volume by Sentiment', fontsize=14, fontweight='bold')
ax4.set_xlabel('Volume (Million USD)')
ax4.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(volume_by_sentiment.sort_values().values):
    ax4.text(v, i, f' ${v:.1f}M', va='center', fontweight='bold')

# 5. Hourly PnL Pattern
ax5 = plt.subplot(4, 2, 5)
hourly_pnl = merged.groupby('hour')['Closed PnL'].mean()
ax5.plot(hourly_pnl.index, hourly_pnl.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
ax5.fill_between(hourly_pnl.index, 0, hourly_pnl.values, alpha=0.3, color='green')
ax5.set_title('Average PnL by Hour of Day (UTC)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Hour')
ax5.set_ylabel('Average Closed PnL ($)')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
max_hour = hourly_pnl.idxmax()
ax5.annotate(f'Peak: ${hourly_pnl.max():.2f}', 
             xy=(max_hour, hourly_pnl.max()), 
             xytext=(max_hour+1, hourly_pnl.max()+10),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, fontweight='bold')

# 6. Top 10 Coins by Trade Count
ax6 = plt.subplot(4, 2, 6)
top_coins = merged['Coin'].value_counts().head(10)
top_coins.plot(kind='bar', ax=ax6, color='purple', alpha=0.7)
ax6.set_title('Top 10 Most Traded Coins', fontsize=14, fontweight='bold')
ax6.set_xlabel('Coin')
ax6.set_ylabel('Number of Trades')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Buy vs Sell Performance by Sentiment
ax7 = plt.subplot(4, 2, 7)
side_sentiment = merged.groupby(['classification', 'Side'])['Closed PnL'].mean().unstack()
side_sentiment.plot(kind='bar', ax=ax7, width=0.8)
ax7.set_title('Average PnL: Buy vs Sell by Sentiment', fontsize=14, fontweight='bold')
ax7.set_ylabel('Average Closed PnL ($)')
ax7.set_xlabel('')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
ax7.legend(title='Side')
ax7.grid(True, alpha=0.3, axis='y')
ax7.axhline(y=0, color='red', linestyle='--', linewidth=1)

# 8. Cumulative PnL Over Time by Sentiment
ax8 = plt.subplot(4, 2, 8)
daily_pnl_sentiment = merged.groupby(['date', 'classification'])['Closed PnL'].sum().unstack(fill_value=0)
cumulative_pnl = daily_pnl_sentiment.cumsum()
for col in cumulative_pnl.columns:
    ax8.plot(cumulative_pnl.index, cumulative_pnl[col], label=col, linewidth=2, alpha=0.8)
ax8.set_title('Cumulative PnL Over Time by Sentiment', fontsize=14, fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_ylabel('Cumulative PnL ($)')
ax8.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
ax8.grid(True, alpha=0.3)
ax8.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('/home/claude/trader_sentiment_analysis.png', dpi=300, bbox_inches='tight')
print("Comprehensive visualization saved: trader_sentiment_analysis.png")

# Create additional focused charts
# Chart 2: Correlation heatmap
fig2, ax = plt.subplots(figsize=(10, 8))
# Create numerical features for correlation
merged_numeric = merged.copy()
merged_numeric['is_profitable'] = (merged_numeric['Closed PnL'] > 0).astype(int)
merged_numeric['is_buy'] = (merged_numeric['Side'] == 'BUY').astype(int)
merged_numeric['sentiment_score'] = merged_numeric['value']

corr_data = merged_numeric[['sentiment_score', 'Closed PnL', 'Size USD', 
                             'is_profitable', 'is_buy', 'hour']].corr()
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix: Trading Metrics & Sentiment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved: correlation_heatmap.png")

# Chart 3: Trader Performance Distribution
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top traders
trader_perf = merged.groupby('Account').agg({
    'Closed PnL': 'sum',
    'Size USD': 'sum'
}).sort_values('Closed PnL', ascending=False)

# 3.1 Top 15 Traders by PnL
top_15 = trader_perf.head(15)['Closed PnL'] / 1000
top_15.plot(kind='barh', ax=axes[0, 0], color='darkgreen')
axes[0, 0].set_title('Top 15 Traders by Total PnL', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Total PnL (Thousands $)')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 3.2 Bottom 15 Traders
bottom_15 = trader_perf.tail(15)['Closed PnL']
colors = ['red' if x < 0 else 'green' for x in bottom_15.values]
bottom_15.plot(kind='barh', ax=axes[0, 1], color=colors)
axes[0, 1].set_title('Bottom 15 Traders by Total PnL', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Total PnL ($)')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3.3 PnL Distribution
axes[1, 0].hist(merged['Closed PnL'].dropna(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Individual Trade PnL', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Closed PnL ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].grid(True, alpha=0.3)

# 3.4 Trade Size Distribution by Sentiment
merged_clean = merged[merged['Size USD'] < merged['Size USD'].quantile(0.95)].copy()
for sentiment in merged_clean['classification'].unique():
    if pd.notna(sentiment):
        data = merged_clean[merged_clean['classification'] == sentiment]['Size USD']
        axes[1, 1].hist(data, bins=50, alpha=0.5, label=sentiment)
axes[1, 1].set_title('Trade Size Distribution by Sentiment (95th percentile)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Size USD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/trader_performance_details.png', dpi=300, bbox_inches='tight')
print("Trader performance details saved: trader_performance_details.png")

print("\nAll visualizations generated successfully!")
