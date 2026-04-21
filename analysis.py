import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Loading datasets...")
# Load datasets
fear_greed = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')
historical = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')

print(f"Fear/Greed Index: {len(fear_greed)} records")
print(f"Historical Trading Data: {len(historical)} records")

# Data preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Convert dates
fear_greed['date'] = pd.to_datetime(fear_greed['date'])
historical['Timestamp'] = pd.to_datetime(historical['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
historical['date'] = historical['Timestamp'].dt.normalize()

# Clean numerical columns
historical['Closed PnL'] = pd.to_numeric(historical['Closed PnL'], errors='coerce')
historical['Size USD'] = pd.to_numeric(historical['Size USD'], errors='coerce')
historical['Execution Price'] = pd.to_numeric(historical['Execution Price'], errors='coerce')

# Merge datasets
merged = historical.merge(fear_greed[['date', 'classification', 'value']], 
                          on='date', how='left')

print(f"\nMerged dataset: {len(merged)} records")
print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

# Basic statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)

print("\nFear/Greed Distribution:")
print(fear_greed['classification'].value_counts())

print("\nTrading Side Distribution:")
print(historical['Side'].value_counts())

print("\nTop 5 Most Traded Coins:")
print(historical['Coin'].value_counts().head())

# Calculate key metrics
print("\n" + "="*60)
print("KEY METRICS BY MARKET SENTIMENT")
print("="*60)

# Group by sentiment
sentiment_metrics = merged.groupby('classification').agg({
    'Closed PnL': ['mean', 'median', 'sum', 'count'],
    'Size USD': ['mean', 'median', 'sum'],
    'Account': 'nunique'
}).round(2)

print("\n", sentiment_metrics)

# Trader performance analysis
print("\n" + "="*60)
print("TRADER PERFORMANCE ANALYSIS")
print("="*60)

trader_performance = merged.groupby('Account').agg({
    'Closed PnL': 'sum',
    'Size USD': 'sum',
    'Account': 'count'
}).rename(columns={'Account': 'trade_count'})

trader_performance['avg_pnl_per_trade'] = (
    trader_performance['Closed PnL'] / trader_performance['trade_count']
)

print(f"\nTotal unique traders: {len(trader_performance)}")
print(f"Profitable traders: {(trader_performance['Closed PnL'] > 0).sum()}")
print(f"Loss-making traders: {(trader_performance['Closed PnL'] < 0).sum()}")
print(f"\nTop 10 Traders by Total PnL:")
print(trader_performance.nlargest(10, 'Closed PnL')[['Closed PnL', 'trade_count', 'avg_pnl_per_trade']])

# Sentiment impact on win rate
print("\n" + "="*60)
print("SENTIMENT IMPACT ON TRADING SUCCESS")
print("="*60)

sentiment_success = merged.groupby('classification').agg({
    'Closed PnL': lambda x: (x > 0).mean() * 100
}).rename(columns={'Closed PnL': 'Win_Rate_%'}).round(2)

print("\n", sentiment_success)

# Time-based analysis
print("\n" + "="*60)
print("TEMPORAL PATTERNS")
print("="*60)

merged['hour'] = merged['Timestamp'].dt.hour
merged['day_of_week'] = merged['Timestamp'].dt.day_name()

hourly_pnl = merged.groupby('hour')['Closed PnL'].mean().round(2)
print("\nAverage PnL by Hour of Day:")
print(hourly_pnl)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nGenerating visualizations...")
