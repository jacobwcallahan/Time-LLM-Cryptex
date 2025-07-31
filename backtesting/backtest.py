import argparse
import backtrader as bt
import numpy as np
import pandas as pd
import tqdm

from utils import load_and_prepare_data
from strategies import (
    SimpleAIStrategy, SLTPStrategy, MomentumAIStrategy,
    RSIAIStrategy, BollingerAIStrategy, MeanReversionAIStrategy,
    TrendFollowingAIStrategy
)

# Strategy configurations
STRATEGIES = {
    'SimpleAI': {
        'class': SimpleAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
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
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
            'momentum_window': 20,
        }
    },
    'RSIAI': {
        'class': RSIAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.015,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
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
    'MeanReversionAI': {
        'class': MeanReversionAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.015,
            'lookback_period': 20,
            'mean_reversion_threshold': 1.5,
        }
    },
    'TrendFollowingAI': {
        'class': TrendFollowingAIStrategy,
        'params': {
            'prediction_horizon': 1,
            'confidence_threshold': 0.01,
            'ema_short': 5,
            'ema_long': 20,
        }
    }
}

# Optimization ranges (all parameters should be passed as iterables)

OPTIMIZATION_RANGES = {
    'SimpleAI': {
        'prediction_horizon': range(1, 4),  # 1, 2, 3
        'confidence_threshold': np.arange(0.005, 0.03, 0.005),  # 0.005 to 0.025
    },
    'SLTP': {
        'prediction_horizon': range(1, 4),  # 1, 2, 3
        'confidence_threshold': np.arange(0.005, 0.02, 0.005),  # 0.005 to 0.015
        'stop_loss_pct': np.arange(0.03, 0.08, 0.02),           # 0.03 to 0.07
        'take_profit_pct': np.arange(0.10, 0.25, 0.05),         # 0.10 to 0.20
    },
    'MomentumAI': {
        'prediction_horizon': range(1, 4),
        'confidence_threshold': np.arange(0.01, 0.025, 0.005),  # 0.01 to 0.02
        'momentum_window': range(10, 40, 10),                   # 10, 20, 30
    },
    'RSIAI': {
        'prediction_horizon': range(1, 4),
        'confidence_threshold': np.arange(0.005, 0.02, 0.005),
        'rsi_period': [14, 21],
        'rsi_oversold': range(20, 40, 10),                      # 20, 30
        'rsi_overbought': range(70, 90, 10),                    # 70, 80
    },
    'BollingerAI': {
        'prediction_horizon': range(1, 4),
        'confidence_threshold': np.arange(0.005, 0.02, 0.005),
        'bb_period': range(15, 30, 5),                          # 15, 20, 25
        'bb_std': [1.5, 2.0, 2.5],
    },
    'MeanReversionAI': {
        'prediction_horizon': range(1, 4),
        'confidence_threshold': np.arange(0.005, 0.02, 0.005),
        'lookback_period': range(15, 30, 5),                    # 15, 20, 25
        'mean_reversion_threshold': np.arange(1.0, 2.5, 0.5),   # 1.0, 1.5, 2.0
    },
    'TrendFollowingAI': {
        'prediction_horizon': range(1, 4),
        'confidence_threshold': np.arange(0.005, 0.02, 0.005),
        'ema_short': [5, 10],
        'ema_long': [20, 30],
    }
}

class BacktestRunner:
    """Main backtesting runner using backtrader"""
    
    def __init__(self, data_path, cash=100000, commission=0.001):
        self.data_path = data_path
        self.cash = cash
        self.commission = commission
        self.data = None
        self.data_feed_class = None
        self.results = {}
        self.load_data()
    
    def load_data(self):
        """Load and prepare data"""
        self.data, self.data_feed_class = load_and_prepare_data(self.data_path)
        print(f"Data loaded with shape: {self.data.shape} from {self.data.index.min()} to {self.data.index.max()}\n")
    
    def run_strategy(self, strategy_name):
        """Run a single strategy"""
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found. Available: {list(STRATEGIES.keys())}")
        
        strategy_class = STRATEGIES[strategy_name]['class']
        params = STRATEGIES[strategy_name]['params']
        
        print(f"[Running] {strategy_name} | Params: {params}")
        
        # Setup cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **params)
        
        # Add data
        data_feed = self.data_feed_class(dataname=self.data)
        cerebro.adddata(data_feed)
        
        # Set cash and commission
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # Run backtest
        results = cerebro.run()
        strategy_result = results[0]
        
        # Extract analyzer results
        analyzer_results = {
            'returns': strategy_result.analyzers.returns.get_analysis(),
            'sharpe': strategy_result.analyzers.sharpe.get_analysis(),
            'drawdown': strategy_result.analyzers.drawdown.get_analysis(),
            'trades': strategy_result.analyzers.trades.get_analysis(),
            'sqn': strategy_result.analyzers.sqn.get_analysis(),
        }
        
        # Store results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.cash) / self.cash * 100
        
        self.results[strategy_name] = {
            'cerebro': cerebro,
            'params': params,
            'final_value': final_value,
            'total_return': total_return,
            'analyzers': analyzer_results
        }
        
        return cerebro, analyzer_results
    
    def optimize_strategy(self, strategy_name):
        """Optimize a strategy using grid search"""
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        if strategy_name not in OPTIMIZATION_RANGES:
            print(f"No optimization ranges defined for {strategy_name}")
            return
        
        strategy_class = STRATEGIES[strategy_name]['class']
        param_ranges = OPTIMIZATION_RANGES[strategy_name]

        print(f"[Optimizing] {strategy_name} | Params: {param_ranges}")

        # Setup cerebro
        cerebro = bt.Cerebro(maxcpus=1) # Mutliprocessing wouldn't work with custom data feed 
        cerebro.optstrategy(strategy_class, **param_ranges)
        
        # Add data
        data_feed = self.data_feed_class(dataname=self.data)
        cerebro.adddata(data_feed)
            
        # Set cash and commission
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
            
        # Run optimization
        opt_results = cerebro.run()

        summary_data = []
        for strat_list in opt_results:
            strat = strat_list[0]  # one instance per param combo
            
            params = strat.params._getkwargs()
            
            # Extract analyzers
            returns = strat.analyzers.returns.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
            drawdown = strat.analyzers.drawdown.get_analysis()
            max_dd = drawdown.get('max', {}).get('drawdown', 0) or 0
            
            trades = strat.analyzers.trades.get_analysis()
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_return = returns.get('rtot', 0)
            final_value = self.cash * (1 + total_return)
            
            summary_data.append({
                **params,
                'Total Return (%)': total_return * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_dd * 100,
                'Total Trades': total_trades,
                'Win Rate (%)': win_rate,
                'Initial Value ($)': self.cash,
                'Final Value ($)': final_value,
                'Strategy Instance': strat
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
        
        print(f"[Optimization Results] {strategy_name}")
        print(df.drop(columns=['Strategy Instance', 'position_size']).head(10).to_string(index=False, float_format='%.2f'))

    def run_all_strategies(self):
        """Run all available strategies"""
        for strategy_name in STRATEGIES.keys():
            try:
                self.run_strategy(strategy_name)
            except Exception as e:
                print(f"/!\\ Error running {strategy_name}: {e}")
    
    def create_summary_table(self):
        """Print summary table of all results and plot best strategy"""
        if not self.results:
            print("No results to summarize")
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            analyzers = result['analyzers']
            
            # Extract key metrics safely
            total_return = result['total_return']
            sharpe = analyzers.get('sharpe', {}).get('sharperatio', 0) or 0
            max_dd = analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0) or 0
            max_dd_pct = max_dd * 100
            
            trades_info = analyzers.get('trades', {})
            total_trades = trades_info.get('total', {}).get('total', 0)
            won_trades = trades_info.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            summary_data.append({
                'Strategy': name,
                'Total Return (%)': total_return,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_dd_pct,
                'Total Trades': total_trades,
                'Win Rate (%)': win_rate,
                'Initial Value ($)': self.cash,
                'Final Value ($)': result['final_value']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
        
        print("[Results]")
        print(df.to_string(index=False, float_format='%.2f'))

        # Show plot for best strategy or single strategy
        strategy_to_plot = df.iloc[0]['Strategy']
        cerebro = self.results[strategy_to_plot]['cerebro']

        print(f"\n[Plot] Showing {strategy_to_plot}")
        cerebro.plot(style='candlestick', barup='green', bardown='red')

        return df
    
    def walk_forward_optimization(self, strategy_name, train_days=12, test_days=3, step_days=1):
        """
        Perform walk-forward optimization on a strategy
        
        Args:
            strategy_name: Name of strategy to optimize
            train_days: Number of days for training/optimization window
            test_days: Number of days for testing window  
            step_days: Number of days to step forward each iteration
        """
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        if strategy_name not in OPTIMIZATION_RANGES:
            print(f"No optimization ranges defined for {strategy_name}")
            return
        
        strategy_class = STRATEGIES[strategy_name]['class']
        param_ranges = OPTIMIZATION_RANGES[strategy_name]
        
        print(f"[Walk Forward Optimization] {strategy_name}")
        print(f"Training: {train_days} days | Testing: {test_days} days | Step: {step_days} days")
        
        # Convert data index to datetime if it isn't already
        data_index = pd.to_datetime(self.data.index)
        start_date = data_index.min()
        end_date = data_index.max()
        
        # Calculate walk-forward periods
        periods = []
        current_start = start_date
        
        while True:
            train_end = current_start + pd.DateOffset(days=train_days)
            test_start = train_end
            test_end = test_start + pd.DateOffset(days=test_days)
            
            # Break if we don't have enough data for test period
            if test_end > end_date:
                break
                
            periods.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start, 
                'test_end': test_end
            })
            
            # Step forward
            current_start = current_start + pd.DateOffset(days=step_days)
                
        wf_results = []
        all_test_trades = []
        
        for i, period in tqdm.tqdm(enumerate(periods), total=len(periods), desc="Processing periods...", ncols=100, leave=False):
            # print(f"\n--- Period {i+1}/{len(periods)} ---")
            # print(f"Train: {period['train_start'].strftime('%Y-%m-%d')} to {period['train_end'].strftime('%Y-%m-%d')}")
            # print(f"Test:  {period['test_start'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}")
            
            # Get training data
            train_mask = (data_index >= period['train_start']) & (data_index < period['train_end'])
            train_data = self.data[train_mask]
            
            # if len(train_data) < 30:  # Skip if insufficient training data
            #     print("Insufficient training data, skipping...")
            #     continue
            
            # Optimize on training data
            # print("Optimizing on training data...")
            cerebro_opt = bt.Cerebro(maxcpus=1)
            cerebro_opt.optstrategy(strategy_class, **param_ranges)
            
            train_feed = self.data_feed_class(dataname=train_data)
            cerebro_opt.adddata(train_feed)
            cerebro_opt.broker.setcash(self.cash)
            cerebro_opt.broker.setcommission(commission=self.commission)
            cerebro_opt.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro_opt.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            opt_results = cerebro_opt.run()
            
            # Find best parameters based on Sharpe ratio
            best_sharpe = -999
            best_params = None
            
            for strat_list in opt_results:
                strat = strat_list[0]
                sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = strat.params._getkwargs()
            
            if best_params is None:
                print("No valid optimization results, skipping...")
                continue
            
            # print(f"Best params: {best_params} (Sharpe: {best_sharpe:.3f})")
            
            # Test on out-of-sample data
            test_mask = (data_index >= period['test_start']) & (data_index < period['test_end'])
            test_data = self.data[test_mask]
            
            # if len(test_data) < 10:  # Skip if insufficient test data
            #     print("Insufficient test data, skipping...")
            #     continue
            
            # print("Testing on out-of-sample data...")
            cerebro_test = bt.Cerebro()
            cerebro_test.addstrategy(strategy_class, **best_params)
            
            test_feed = self.data_feed_class(dataname=test_data)
            cerebro_test.adddata(test_feed)
            cerebro_test.broker.setcash(self.cash)
            cerebro_test.broker.setcommission(commission=self.commission)
            
            # Add analyzers for test
            cerebro_test.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro_test.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro_test.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro_test.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            test_results = cerebro_test.run()
            test_strat = test_results[0]
            
            # Extract test results
            test_analyzers = {
                'returns': test_strat.analyzers.returns.get_analysis(),
                'sharpe': test_strat.analyzers.sharpe.get_analysis(),
                'drawdown': test_strat.analyzers.drawdown.get_analysis(),
                'trades': test_strat.analyzers.trades.get_analysis(),
            }
            
            test_return = test_analyzers['returns'].get('rtot', 0) * 100
            test_sharpe = test_analyzers['sharpe'].get('sharperatio', 0) or 0
            test_dd = test_analyzers['drawdown'].get('max', {}).get('drawdown', 0) * 100 or 0
            
            trades_info = test_analyzers['trades']
            total_trades = trades_info.get('total', {}).get('total', 0)
            won_trades = trades_info.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            period_result = {
                'period': i + 1,
                'train_start': period['train_start'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'best_params': best_params,
                'train_sharpe': best_sharpe,
                'test_return_pct': test_return,
                'test_sharpe': test_sharpe,
                'test_max_dd_pct': test_dd,
                'test_trades': total_trades,
                'test_win_rate': win_rate
            }
            
            wf_results.append(period_result)
            all_test_trades.extend([test_return])  # Collect returns for aggregation
                    
        # Create summary
        if wf_results:
            wf_df = pd.DataFrame(wf_results)
            
            # Calculate aggregate statistics
            total_periods = len(wf_results)
            avg_test_return = wf_df['test_return_pct'].mean()
            avg_test_sharpe = wf_df['test_sharpe'].mean()
            avg_test_dd = wf_df['test_max_dd_pct'].mean()
            positive_periods = len(wf_df[wf_df['test_return_pct'] > 0])
            win_rate_periods = (positive_periods / total_periods * 100) if total_periods > 0 else 0
            
            # Calculate cumulative return (compound returns)
            cumulative_return = 1.0
            for ret in wf_df['test_return_pct']:
                cumulative_return *= (1 + ret/100)
            cumulative_return = (cumulative_return - 1) * 100
            
            print(f"[Walk Forward Summary] {strategy_name}")
            print(f"Total Periods: {total_periods}")
            print(f"Positive Periods: {positive_periods}/{total_periods} ({win_rate_periods:.1f}%)")
            print(f"Average Test Return: {avg_test_return:.2f}%")
            print(f"Cumulative Return: {cumulative_return:.2f}%")
            print(f"Average Sharpe Ratio: {avg_test_sharpe:.3f}")
            print(f"Average Max Drawdown: {avg_test_dd:.2f}%")
            
            print(f"\n[Period Details]")
            display_df = wf_df[['period', 'test_start', 'test_end', 'test_return_pct', 
                            'test_sharpe', 'test_max_dd_pct', 'test_trades']].copy()
            display_df['test_start'] = display_df['test_start'].dt.strftime('%Y-%m-%d')
            display_df['test_end'] = display_df['test_end'].dt.strftime('%Y-%m-%d')
            
            print(display_df.to_string(index=False, float_format='%.2f'))
            
            return wf_df
        else:
            print("No valid walk-forward periods completed")
            return pd.DataFrame()
    

def main():
    parser = argparse.ArgumentParser(description='AI Trading Strategy Backtester')
    parser.add_argument('--data', required=True, help='Path to CSV file with AI predictions')
    parser.add_argument('--strategy', help='Specific strategy to run (default: all)')
    parser.add_argument('--optimize', help='Strategy to optimize')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    parser.add_argument('--walk_forward', help='Strategy for walk-forward optimization')
    parser.add_argument('--train_days', type=int, default=60, help='Training days for walk-forward')
    parser.add_argument('--test_days', type=int, default=30, help='Test days for walk-forward')
    parser.add_argument('--step_days', type=int, default=30, help='Step days for walk-forward')

    
    args = parser.parse_args()
    
    runner = BacktestRunner(args.data, cash=args.cash, commission=args.commission)
    
    if args.optimize:
        # Optimize specific strategy
        runner.optimize_strategy(args.optimize)

    elif args.strategy:
        # Run specific strategy
        runner.run_strategy(args.strategy)
        runner.create_summary_table()
    
    elif args.walk_forward:
        # Walk-forward optimization
        runner.walk_forward_optimization(args.walk_forward, 
                                   train_days=args.train_days,
                                   test_days=args.test_days, 
                                   step_days=args.step_days)

    else:
        # Run all strategies
        runner.run_all_strategies()
        runner.create_summary_table()

if __name__ == "__main__":
    main()