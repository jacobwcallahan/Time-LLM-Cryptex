"""
Script to fetch cryptocurrency sentiment and market data
- Crypto Fear & Greed Index from Alternative.me (free, no API key needed)
- CoinMarketCap market data (requires API key)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()


class FearGreedIndexFetcher:
    """Fetch Crypto Fear & Greed Index from Alternative.me (free)"""
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.session = requests.Session()
    
    def fetch_current(self):
        """Fetch current Fear & Greed Index value"""
        try:
            response = self.session.get(self.BASE_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if request was successful (error can be None, 'null', or not present)
            error = data.get('metadata', {}).get('error')
            if error is None or error == 'null' or error == 'None':
                current = data['data'][0]
                return {
                    'value': int(current['value']),
                    'classification': current['value_classification'],
                    'timestamp': int(current['timestamp']),
                    'date': datetime.fromtimestamp(int(current['timestamp']))
                }
            else:
                print(f"[ERROR] API returned error: {error}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to fetch current index: {e}")
            return None
    
    def fetch_historical(self, days=365):
        """
        Fetch historical Fear & Greed Index
        
        Args:
            days: Number of days of historical data (max ~2000)
            
        Returns:
            pd.DataFrame with columns [date, timestamp, fear_greed_value, fear_greed_classification]
        """
        url = f"{self.BASE_URL}?limit={days}"
        
        try:
            print(f"Fetching Fear & Greed Index for last {days} days...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if request was successful (error can be None, 'null', or not present)
            error = data.get('metadata', {}).get('error')
            if error is None or error == 'null' or error == 'None':
                records = []
                for item in data['data']:
                    records.append({
                        'date': datetime.fromtimestamp(int(item['timestamp'])),
                        'timestamp': int(item['timestamp']),
                        'fear_greed_value': int(item['value']),
                        'fear_greed_classification': item['value_classification']
                    })
                
                df = pd.DataFrame(records)
                df = df.sort_values('date').reset_index(drop=True)
                
                print(f"[OK] Fetched {len(df)} records")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"\nFear & Greed Classification Guide:")
                print("  0-24: Extreme Fear")
                print("  25-49: Fear")
                print("  50-74: Greed")
                print("  75-100: Extreme Greed")
                
                return df
            else:
                print(f"[ERROR] API returned error: {error}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch historical data: {e}")
            return None


class CoinMarketCapFetcher:
    """Fetch market data from CoinMarketCap API (requires API key)"""
    
    BASE_URL = "https://pro-api.coinmarketcap.com"
    
    def __init__(self, api_key=None):
        """
        Initialize CoinMarketCap fetcher
        
        Args:
            api_key: Your CoinMarketCap API key (get free at https://coinmarketcap.com/api/)
        """
        self.api_key = api_key or os.getenv('COINMARKETCAP_API_KEY')
        
        if not self.api_key:
            print("[WARNING] No CoinMarketCap API key provided.")
            print("          Set COINMARKETCAP_API_KEY environment variable or pass api_key parameter")
            print("          Get free key at: https://coinmarketcap.com/api/")
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.api_key,
            })
    
    def fetch_btc_quotes(self, convert='USD'):
        """
        Fetch current Bitcoin quotes and market data
        
        Args:
            convert: Currency to convert to (default: USD)
            
        Returns:
            dict with BTC market data
        """
        if not self.api_key:
            print("[ERROR] API key required for CoinMarketCap")
            return None
        
        url = f"{self.BASE_URL}/v2/cryptocurrency/quotes/latest"
        params = {
            'symbol': 'BTC',
            'convert': convert
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status']['error_code'] == 0:
                btc_data = data['data']['BTC'][0]
                quote = btc_data['quote'][convert]
                
                return {
                    'name': btc_data['name'],
                    'symbol': btc_data['symbol'],
                    'price': quote['price'],
                    'volume_24h': quote['volume_24h'],
                    'volume_change_24h': quote['volume_change_24h'],
                    'percent_change_1h': quote['percent_change_1h'],
                    'percent_change_24h': quote['percent_change_24h'],
                    'percent_change_7d': quote['percent_change_7d'],
                    'percent_change_30d': quote['percent_change_30d'],
                    'market_cap': quote['market_cap'],
                    'market_cap_dominance': quote['market_cap_dominance'],
                    'last_updated': quote['last_updated']
                }
            else:
                print(f"[ERROR] API error: {data['status']['error_message']}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch BTC quotes: {e}")
            return None
    
    def fetch_global_metrics(self):
        """Fetch global cryptocurrency market metrics"""
        if not self.api_key:
            print("[ERROR] API key required for CoinMarketCap")
            return None
        
        url = f"{self.BASE_URL}/v1/global-metrics/quotes/latest"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status']['error_code'] == 0:
                metrics = data['data']['quote']['USD']
                return {
                    'total_market_cap': metrics['total_market_cap'],
                    'total_volume_24h': metrics['total_volume_24h'],
                    'btc_dominance': metrics['btc_dominance'],
                    'eth_dominance': metrics['eth_dominance'],
                    'last_updated': metrics['last_updated']
                }
            else:
                print(f"[ERROR] API error: {data['status']['error_message']}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch global metrics: {e}")
            return None


def fetch_fear_greed_index(days=None, start_date='2019-09-01', 
                           output_file='dataset/sentiment/fear_greed_index.csv'):
    """
    Fetch Fear & Greed Index and save to CSV
    
    Args:
        days: Number of days (if None, calculates from start_date to today)
        start_date: Start date in YYYY-MM-DD format
        output_file: Path to save CSV
    """
    fetcher = FearGreedIndexFetcher()
    
    # Calculate days if not provided
    if days is None:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.Timestamp.now()
        days = (end_dt - start_dt).days
    
    # Fetch data
    df = fetcher.fetch_historical(days=min(days, 2000))  # API limit
    
    if df is not None:
        # Filter by start_date if needed
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved to {output_file}")
        print(f"\nPreview:")
        print(df.head(10))
        print(f"\nSummary statistics:")
        print(df['fear_greed_value'].describe())
        print(f"\nValue distribution:")
        print(df['fear_greed_classification'].value_counts())
        
        return df
    else:
        return None


def fetch_cmc_current_data(api_key, output_file='dataset/sentiment/cmc_current.json'):
    """
    Fetch current CoinMarketCap data
    
    Args:
        api_key: CoinMarketCap API key
        output_file: Path to save JSON
    """
    fetcher = CoinMarketCapFetcher(api_key=api_key)
    
    print("Fetching current Bitcoin data from CoinMarketCap...")
    btc_data = fetcher.fetch_btc_quotes()
    
    if btc_data:
        print("\n[OK] Current Bitcoin Data:")
        print(f"  Price: ${btc_data['price']:,.2f}")
        print(f"  24h Change: {btc_data['percent_change_24h']:.2f}%")
        print(f"  7d Change: {btc_data['percent_change_7d']:.2f}%")
        print(f"  Market Cap Dominance: {btc_data['market_cap_dominance']:.2f}%")
        print(f"  24h Volume: ${btc_data['volume_24h']:,.0f}")
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(btc_data, f, indent=2)
        print(f"\n[OK] Saved to {output_file}")
        
        return btc_data
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency sentiment data')
    parser.add_argument('--source', type=str, default='fear_greed',
                       choices=['fear_greed', 'cmc', 'both'],
                       help='Data source: fear_greed (free), cmc (requires API key), or both')
    parser.add_argument('--start_date', type=str, default='2019-09-01',
                       help='Start date (YYYY-MM-DD) for Fear & Greed Index')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of days for Fear & Greed Index (overrides start_date)')
    parser.add_argument('--cmc_api_key', type=str, default=None,
                       help='CoinMarketCap API key (or set COINMARKETCAP_API_KEY env var)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.source in ['fear_greed', 'both']:
        output = args.output or 'dataset/sentiment/fear_greed_index.csv'
        fetch_fear_greed_index(
            days=args.days,
            start_date=args.start_date,
            output_file=output
        )
    
    if args.source in ['cmc', 'both']:
        if not args.cmc_api_key and not os.getenv('COINMARKETCAP_API_KEY'):
            print("\n[ERROR] CoinMarketCap API key required!")
            print("  Use --cmc_api_key YOUR_KEY or set COINMARKETCAP_API_KEY environment variable")
            print("  Get free key at: https://coinmarketcap.com/api/")
        else:
            output = args.output or 'dataset/sentiment/cmc_current.json'
            fetch_cmc_current_data(
                api_key=args.cmc_api_key,
                output_file=output
            )

