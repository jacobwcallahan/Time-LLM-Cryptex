assets = ["ETH-USD", "XRP-USD", "DOGE-USD", "SOL-USD", "ADA-USD", "BCH-USD", "LTC-USD","EURUSD=X", "JPY-X", "AAPL", "NVDA","TSLA",]
from run_inf_and_backtest import run_inference_and_backtest
for asset in assets:
    run_inference_and_backtest(
                            model_name = "hourly_1week_to_1day_50pct_20251104_172640", 
                            experiment_name = "LLAMA3.1",
                            granularity = "hourly",
                            aggregate = 1,
                            start_date = "2024-01-01",
                            end_date = "2024-12-31",
                            dataset_path = ".", 
                            custom_dataset_path = f"dataset/other_assets/{asset.replace('=', '-')}.csv", 
                            save_path = f"temp", 
                            asset = asset
                        )