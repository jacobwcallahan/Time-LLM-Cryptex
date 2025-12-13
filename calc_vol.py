import pandas as pd
import numpy as np

# Load the BTC-USD dataset
df = pd.read_csv('dataset/BTC-USD.csv')

# Calculate log returns from Close prices
df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Calculate rolling volatility (30-day window)
# volatility is the standard deviation of returns, annualized
window = 30
df['volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(365)

# Display basic statistics
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows with volatility:")
print(df[['timestamp','volatility']].head(35))
print(f"\nvolatility statistics:")
print(df['volatility'].describe())

# Save the updated dataset
output_file = 'dataset/BTC-USD_with_volatility.csv'
df.drop(columns=['Returns', 'Close', "Volume", "Open", "High", "Low"], inplace=True)
df.to_csv(output_file, index=False)
print(f"\nSaved updated dataset to: {output_file}")

