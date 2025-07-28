import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from backtesting import Backtest

warnings.filterwarnings('ignore')

# Import your strategies
from utils import (
    load_and_prepare_data, analyze_predictions
)
from strategies import (
    SimpleAIStrategy, SLTPStrategy, MomentumAIStrategy,
    MultiHorizonStrategy, VolumeAIStrategy, RSIAIStrategy, BollingerAIStrategy,
    ProbabilisticAIStrategy, MeanReversionAIStrategy, TrendFollowingAIStrategy
)

CASH = 1000000  # Default initial cash for backtests
COMMISSION = 0.002  # Default commission rate
# Dictionary to map strategy names to classes and default parameters (set to relatively optimized values)
STRATEGIES = {
    'SimpleAI': {
        'class': SimpleAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.025,
        }
    },
    'SLTP': {
        'class': SLTPStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15,
        }
    },
    'MomentumAI': {
        'class': MomentumAIStrategy,
        'params': {
            'prediction_horizon': 2,
            'confidence_threshold': 0.01,
            'momentum_window': 20,
        }
    },
    'MultiHorizon': {
        'class': MultiHorizonStrategy,
        'params': {
            'short_horizon': 1,
            'long_horizon': 6,
            'confidence_threshold': 0.015,
        }
    },
    'VolumeAI': {
        'class': VolumeAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
            'volume_multiplier': 2.0,
            'volume_window': 30,
        }
    },
    'RSIAI': {
        'class': RSIAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.015,
            'rsi_period': 14,
            'rsi_oversold': 20,
            'rsi_overbought': 60,
        }
    },
    'BollingerAI': {
        'class': BollingerAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
            'bb_period': 20,
            'bb_std': 2.0,
        }
    },
    'ProbabilisticAI': {
        'class': ProbabilisticAIStrategy,
        'params': {
            'prediction_horizons': (1, 3, 5, 7),
            'min_confidence': 0.8,
            'magnitude_threshold': 0.02,
        }
    },
    'MeanReversionAI': {
        'class': MeanReversionAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'lookback_period': 20,
            'mean_reversion_threshold': 1.5,
            'confidence_threshold': 0.015,
        }
    },
    'TrendFollowingAI': {
        'class': TrendFollowingAIStrategy,
        'params': {
            'prediction_horizon': 3,
            'confidence_threshold': 0.01,
            'ema_short': 5,
            'ema_long': 100,
        }
    }
}
# Define optimization ranges for different strategies
OPTIMIZATION_RANGES = {
    'SimpleAI': {
        'confidence_threshold': [0.005, 0.01, 0.015, 0.02, 0.025],
        'prediction_horizon': [1, 2, 3, 5]
    },
    'SLTP': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'stop_loss_pct': [0.03, 0.05, 0.07],
        'take_profit_pct': [0.08, 0.10, 0.15],
        'prediction_horizon': [1, 2, 3]
    },
    'MomentumAI': {
        'confidence_threshold': [0.01, 0.015, 0.02],
        'momentum_window': [5, 10, 20],
        'prediction_horizon': [1, 2, 3]
    },
    'MultiHorizon': {
        'short_horizon': [1, 2, 3],
        'long_horizon': [4, 5, 6, 7],
        'confidence_threshold': [0.005, 0.01, 0.015]
    },
    'VolumeAI': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'volume_multiplier': [1.2, 1.5, 2.0],
        'volume_window': [10, 20, 30],
        'prediction_horizon': [1, 2, 3]
    },
    'RSIAI': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'rsi_period': [7, 14, 21],
        'rsi_oversold': [20, 30, 40],
        'rsi_overbought': [60, 70, 80],
        'prediction_horizon': [1, 2, 3]
    },
    'BollingerAI': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'bb_period': [10, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
        'prediction_horizon': [1, 2, 3]
    },
    'ProbabilisticAI': {
        'prediction_horizons': [(1, 2, 3), (1, 3, 5), (2, 4, 6), (1, 3, 5, 7), (1, 2, 3, 4, 5, 6, 7)],
        'min_confidence': [0.1, 0.2, 0.5, 0.67, 0.8, 0.9],
        'magnitude_threshold': [0.005, 0.01, 0.02]
    },
    'MeanReversionAI': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'lookback_period': [5, 10, 20],
        'mean_reversion_threshold': [0.01, 0.1, 0.5, 1, 1.5, 2],
        'prediction_horizon': [1, 2, 3]
    },
    'TrendFollowingAI': {
        'confidence_threshold': [0.005, 0.01, 0.015],
        'ema_short': [5, 10, 20],
        'ema_long': [30, 50, 100],
        'prediction_horizon': [1, 2, 3]
    }
}



class BacktestRunner:
    """Main class for running and analyzing backtests"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None # pd.DataFrame
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare data"""
        self.data = load_and_prepare_data(self.data_path)
        print(f"Found prediction columns: {[col for col in self.data.columns if col.startswith('close_predicted_')]}")
        print(f"Data loaded with {len(self.data)} rows and {len(self.data.columns)} columns")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
    
    def run_single_strategy(self, strategy_name, **params):
        """Run backtest for a single strategy"""        
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found. Available: {list(STRATEGIES.keys())}")
        
        strategy_class = STRATEGIES[strategy_name]['class']

        default_params = STRATEGIES[strategy_name]['params']
        full_params = {**default_params, **params}

        print(f"=== Running {strategy_name} ===")
        print(f"Parameters: {full_params}")

        bt = Backtest(self.data, strategy_class, cash=CASH, commission=COMMISSION)
        run_result = bt.run(**full_params)

        print(f"Backtest Results:")
        print(f"Initial Cash [$]: {CASH}")
        for key, value in run_result.items():
            if key in ['Equity Final [$]', 'Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', '# Trades', 'Win Rate [%]']:
                print(f"{key}: {value:.4f}")
        print()
        self.results[strategy_name] = {
            'backtest': bt,
            'stats': run_result,
            'params': full_params
        }
        
        return bt, run_result
    
    def run_all_strategies(self):
        """Run all available strategies with default parameters"""
        for strategy_name in list(STRATEGIES.keys()):
            # Run strategy with default parameters
            self.run_single_strategy(strategy_name)

    def analyze_predictions(self):
        """Analyze AI model prediction accuracy"""
        print("\n=== AI Model Prediction Analysis ===")
        analysis = analyze_predictions(self.data)
        
        for horizon, metrics in analysis.items():
            print(f"{horizon} | Correlation = {metrics['correlation']:.3f} | "
                  f"Direction_Accuracy = {metrics['direction_accuracy']:.3f} | "
                  f"MAE = {metrics['mae']:.2f}")
        print()
        return analysis
    
    def summarize_results(self, sort_by='Sharpe Ratio', ascending=False):
        """Return a DataFrame summarizing self.results, sorted by the given metric."""
        summary = []
        for name, result in self.results.items():
            stats = result['stats']
            summary.append({
                'Strategy': name,
                'Return [%]': stats['Return [%]'],
                'Sharpe Ratio': stats['Sharpe Ratio'],
                'Max Drawdown [%]': stats['Max. Drawdown [%]'],
                'Win Rate [%]': stats['Win Rate [%]'],
                '# Trades': stats['# Trades'],
            })  
        df = pd.DataFrame(summary)
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        return df

    def create_performance_charts(self, filename):
        """Create performance visualization charts"""
        df_report = self.summarize_results()

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Return vs Sharpe scatter
        axes[0, 0].scatter(df_report['Return [%]'], df_report['Sharpe Ratio'], 
                          s=100, alpha=0.7)
        for i, txt in enumerate(df_report['Strategy']):
            axes[0, 0].annotate(txt, (df_report['Return [%]'].iloc[i], 
                                     df_report['Sharpe Ratio'].iloc[i]), 
                               fontsize=8, rotation=45)
        axes[0, 0].set_xlabel('Return [%]')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Return vs Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns bar chart
        axes[0, 1].bar(range(len(df_report)), df_report['Return [%]'], 
                       color='green' if df_report['Return [%]'].iloc[0] > 0 else 'red',
                       alpha=0.7)
        axes[0, 1].set_xticks(range(len(df_report)))
        axes[0, 1].set_xticklabels(df_report['Strategy'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Return [%]')
        axes[0, 1].set_title('Strategy Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown
        axes[1, 0].bar(range(len(df_report)), df_report['Max Drawdown [%]'], 
                       color='red', alpha=0.7)
        axes[1, 0].set_xticks(range(len(df_report)))
        axes[1, 0].set_xticklabels(df_report['Strategy'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Max Drawdown [%]')
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win Rate vs Number of Trades
        axes[1, 1].scatter(df_report['# Trades'], df_report['Win Rate [%]'], 
                          s=100, alpha=0.7)
        for i, txt in enumerate(df_report['Strategy']):
            axes[1, 1].annotate(txt, (df_report['# Trades'].iloc[i], 
                                     df_report['Win Rate [%]'].iloc[i]), 
                               fontsize=8, rotation=45)
        axes[1, 1].set_xlabel('Number of Trades')
        axes[1, 1].set_ylabel('Win Rate [%]')
        axes[1, 1].set_title('Win Rate vs Trade Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Performance charts saved to {filename}")
        plt.show()
    
    def optimize_strategy(self, strategy_name, param_ranges, metric='Sharpe Ratio'):
        """Optimize a specific strategy"""
        
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy_class = STRATEGIES[strategy_name]['class']
        
        print(f"=== Optimizing {strategy_name} ===")
        print(f"Parameter ranges: {param_ranges}")

        bt = Backtest(self.data, strategy_class, cash=CASH, commission=COMMISSION)
        opt_result = bt.optimize(**param_ranges, maximize=metric)
    
        print(f"Optimization Results:")
        print(f"Best parameters: {opt_result._strategy._params}")

        default_params = STRATEGIES[strategy_name]['params']
        full_params = {**default_params, **opt_result._strategy._params}

        print(f"Initial Cash [$]: {CASH}")
        for key, value in opt_result.items():
            if key in ['Equity Final [$]', 'Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', '# Trades', 'Win Rate [%]']:
                print(f"{key}: {value:.4f}")
        print()
        self.results[strategy_name] = {
            'backtest': bt,
            'stats': opt_result,
            'params': full_params
        }
        
        return bt, opt_result
    
    def save_results(self, filename):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save.")
            return
        
        # Convert results to serializable format
        serializable_results = {}
        for strategy_name, result_data in self.results.items():
            stats = result_data['stats']
            params = result_data['params']
            
            serializable_results[strategy_name] = {
                'stats': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                         for k, v in stats.items()},
                'params': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                         for k, v in params.items()},
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='AI Trading Strategy Backtester')
    parser.add_argument('--data', required=True, help='Path to CSV file with AI predictions')
    parser.add_argument('--strategy', help='Specific strategy to run (default: all)')
    parser.add_argument('--optimize', help='Strategy to optimize')
    parser.add_argument('--output_dir', default='results', help='Output directory for results')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    
    args = parser.parse_args()

    # Setting root directory for output as current script directory
    script_dir = Path(__file__).resolve().parent
    args.output_dir = script_dir / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames for Backtesting.py plots
    def bt_plot_filename(bt_results):
        # _strategy attribute example: SimpleAIStrategy(prediction_horizon=1,confidence_threshold=0.01)
        return str(args.output_dir / f"{bt_results._strategy}.html")
    
    # Initialize runner
    runner = BacktestRunner(args.data)
    
    # Analyze predictions first
    runner.analyze_predictions()
    
    if args.strategy:
        # Run specific strategy
        bt, stats = runner.run_single_strategy(args.strategy)
        runner.save_results(f"{args.output_dir}/detailed_results.json")
        
        if args.plot:
            bt.plot(filename=bt_plot_filename(stats))                
    
    elif args.optimize:
        # Optimize specific strategy        
        if args.optimize in OPTIMIZATION_RANGES:
            bt, stats = runner.optimize_strategy(args.optimize, OPTIMIZATION_RANGES[args.optimize])
            runner.save_results(f"{args.output_dir}/detailed_results.json")
            
            if args.plot:
                bt.plot(filename=bt_plot_filename(stats))
                    
        else:
            print(f"No optimization ranges defined for {args.optimize}")
    
    else:
        # Run all strategies by default
        runner.run_all_strategies()
        runner.save_results(f"{args.output_dir}/detailed_results.json")

        if args.plot:
            print("Creating performance charts...")
            runner.create_performance_charts(f"{args.output_dir}/performance_charts.png")

            best_strategy = max(runner.results,
                                key=lambda k: runner.results[k]['stats']['Sharpe Ratio'])
            best_stats = runner.results[best_strategy]['stats']

            print(f"Plotting best strategy {best_strategy}...")
            runner.results[best_strategy]['backtest'].plot(filename=bt_plot_filename(best_stats))
            
if __name__ == "__main__":
    main()