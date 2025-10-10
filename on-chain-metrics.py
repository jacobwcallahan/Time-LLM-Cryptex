"""
Script to fetch on-chain metrics from Blockchain.com API
Can be used as additional features for Bitcoin price prediction
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os


class BlockchainDataFetcher:
    """Fetch various on-chain metrics from Blockchain.com API"""
    
    BASE_URL = "https://api.blockchain.info/charts"
    
    # Available charts that might be useful for BTC prediction
    AVAILABLE_CHARTS = {
        'transactions_per_day': 'n-transactions',  # Daily confirmed transactions
        'transaction_fees': 'transaction-fees',     # Total transaction fees
        'hash_rate': 'hash-rate',                   # Network hash rate
        'difficulty': 'difficulty',                 # Mining difficulty
        'miners_revenue': 'miners-revenue',         # Miners revenue (USD)
        'market_price': 'market-price',             # Market price (USD)
        'trade_volume': 'trade-volume-usd',         # Trade volume (USD)
        'mempool_size': 'mempool-size',             # Mempool size (bytes)
        'avg_block_size': 'avg-block-size',         # Average block size
        'n_unique_addresses': 'n-unique-addresses', # Unique addresses used
        'total_output_volume': 'output-volume',     # Total output volume
        'estimated_transaction_volume': 'estimated-transaction-volume-usd',  # Est. transaction volume
    }
    
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_chart_data(self, chart_name, timespan='5years', start_date=None, 
                        end_date=None, rolling_average=None, sampled=False):
        """
        Fetch data for a specific chart
        
        Args:
            chart_name: Name of the chart (use AVAILABLE_CHARTS keys)
            timespan: Duration like '1year', '5years', etc.
            start_date: Start date in YYYY-MM-DD format
            end_date: Not directly supported by API, we'll filter after
            rolling_average: Duration like '8hours', '1day' etc.
            sampled: If True, limits to ~1.5k datapoints
            
        Returns:
            pd.DataFrame with columns [timestamp, date, value]
        """
        # Get the actual chart ID from our mapping
        if chart_name in self.AVAILABLE_CHARTS:
            chart_id = self.AVAILABLE_CHARTS[chart_name]
        else:
            chart_id = chart_name  # Use as-is if not in mapping
            
        # Build URL
        url = f"{self.BASE_URL}/{chart_id}"
        params = {
            'format': 'json',
            'timespan': timespan,
            'sampled': str(sampled).lower()
        }
        
        if start_date:
            params['start'] = start_date
            
        if rolling_average:
            params['rollingAverage'] = rolling_average
            
        print(f"Fetching {chart_name} data from {url}...")
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            if data['status'] == 'ok':
                values = data['values']
                df = pd.DataFrame(values)
                df.columns = ['timestamp', chart_name]
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Reorder columns
                df = df[['date', 'timestamp', chart_name]]
                
                # Filter by end_date if provided
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df['date'] <= end_dt]
                
                print(f"[OK] Fetched {len(df)} records for {chart_name}")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                return df
            else:
                print(f"[ERROR] API returned status '{data.get('status')}'")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Error parsing data: {e}")
            return None
    
    def fetch_multiple_metrics(self, chart_names, timespan='5years', start_date=None):
        """
        Fetch multiple on-chain metrics and merge them
        
        Args:
            chart_names: List of chart names to fetch
            timespan: Duration for all charts
            start_date: Start date for all charts
            
        Returns:
            pd.DataFrame with all metrics merged on date
        """
        dfs = []
        
        for chart_name in chart_names:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            df = self.fetch_chart_data(chart_name, timespan=timespan, start_date=start_date)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            print("[ERROR] No data fetched")
            return None
            
        # Merge all dataframes on date
        print(f"\nMerging {len(dfs)} metrics...")
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=['date', 'timestamp'], how='outer')
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        print(f"[OK] Merged dataset: {len(merged)} rows x {len(merged.columns)} columns")
        return merged
    
    def fetch_stats(self):
        """Fetch current blockchain statistics"""
        url = "https://api.blockchain.info/stats"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            stats = response.json()
            return stats
        except Exception as e:
            print(f"[ERROR] Error fetching stats: {e}")
            return None


def fetch_daily_transactions(start_date='2019-09-01', end_date=None, 
                             output_file='dataset/onchain/daily_transactions.csv'):
    """
    Convenience function to fetch daily confirmed transactions
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        output_file: Path to save CSV file
    """
    fetcher = BlockchainDataFetcher()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate timespan
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days_diff = (end_dt - start_dt).days
    years = days_diff / 365
    
    if years > 5:
        timespan = f'{int(years)}years'
    elif years > 1:
        timespan = f'{int(years * 12)}months'
    else:
        timespan = f'{days_diff}days'
    
    # Fetch the data
    df = fetcher.fetch_chart_data(
        chart_name='transactions_per_day',
        timespan=timespan,
        start_date=start_date,
        end_date=end_date
    )
    
    if df is not None:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved to {output_file}")
        print(f"\nPreview:")
        print(df.head(10))
        print(f"\nSummary statistics:")
        print(df['transactions_per_day'].describe())
        
        return df
    else:
        return None


def fetch_onchain_features(start_date='2019-09-01', end_date=None,
                           output_file='dataset/onchain/onchain_features.csv'):
    """
    Fetch multiple on-chain metrics useful for BTC prediction
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Path to save CSV file
    """
    fetcher = BlockchainDataFetcher()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Key metrics that might predict price
    metrics = [
        'transactions_per_day',      # Network activity
        'hash_rate',                  # Network security
        'difficulty',                 # Mining difficulty
        'n_unique_addresses',         # User adoption
        'estimated_transaction_volume',  # Economic activity
        'avg_block_size',            # Network capacity usage
    ]
    
    print(f"Fetching on-chain features from {start_date} to {end_date}...")
    print(f"Metrics: {', '.join(metrics)}\n")
    
    # Calculate timespan
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days_diff = (end_dt - start_dt).days
    years = days_diff / 365
    
    if years > 5:
        timespan = f'{int(years)}years'
    else:
        timespan = f'{int(years * 12)}months'
    
    # Fetch all metrics
    df = fetcher.fetch_multiple_metrics(
        chart_names=metrics,
        timespan=timespan,
        start_date=start_date
    )
    
    if df is not None:
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved to {output_file}")
        print(f"\nPreview:")
        print(df.head())
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total records: {len(df)}")
        
        return df
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch on-chain metrics from Blockchain.com')
    parser.add_argument('--metric', type=str, default='transactions',
                       choices=['transactions', 'addresses', 'all', 'stats'],
                       help='Which metric(s) to fetch')
    parser.add_argument('--start_date', type=str, default='2019-09-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    if args.metric == 'transactions':
        output = args.output or 'dataset/onchain/daily_transactions.csv'
        fetch_daily_transactions(
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=output
        )
    
    elif args.metric == 'addresses':
        output = args.output or 'dataset/onchain/unique_addresses.csv'
        fetcher = BlockchainDataFetcher()
        
        # Calculate timespan
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date) if args.end_date else pd.Timestamp.now()
        days_diff = (end_dt - start_dt).days
        years = days_diff / 365
        
        if years > 5:
            timespan = f'{int(years)}years'
        else:
            timespan = f'{int(years * 12)}months'
        
        df = fetcher.fetch_chart_data(
            chart_name='n_unique_addresses',
            timespan=timespan,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if df is not None:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            df.to_csv(output, index=False)
            print(f"\n[OK] Saved to {output}")
            print(f"\nPreview:")
            print(df.head(10))
            print(f"\nSummary statistics:")
            print(df['n_unique_addresses'].describe())
        
    elif args.metric == 'all':
        output = args.output or 'dataset/onchain/onchain_features.csv'
        fetch_onchain_features(
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=output
        )
        
    elif args.metric == 'stats':
        fetcher = BlockchainDataFetcher()
        stats = fetcher.fetch_stats()
        if stats:
            print("\nCurrent Blockchain Statistics:")
            print("=" * 50)
            for key, value in stats.items():
                print(f"{key:40s}: {value}")

