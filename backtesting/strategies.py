from backtesting import Strategy
import pandas as pd
import numpy as np

class SimpleAIStrategy(Strategy):
    """
    Simple strategy: Buy when AI predicts price increase, sell when decrease
    """
    # Default strategy parameters (can be optimized)
    prediction_horizon = 1 # Which prediction to use (1, ..., `pred_len` days ahead)
    confidence_threshold = 0.01 # Minimum price change % to trigger trade
    
    def init(self):
        """Initialize strategy indicators and data"""
        # Use self.I() to register predicted returns as an indicator.
        # This prevents lookahead bias (data is revealed as the backtester moves forward in time).
        pred_col = f'close_predicted_{self.prediction_horizon}'
        if pred_col not in self.data.df.columns:
            raise ValueError(f"Prediction column {pred_col} not found in data")

        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1, # Calculate predicted returns
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        """ # Something you can do to visualize the predictions:
        self.visual = self.I(
            lambda predictions: predictions.shift(int(self.prediction_horizon)),
            self.data.df[pred_col],
            name='Raw Predictions',
            overlay=True
        )
        """
        
    def next(self):
        """Execute strategy logic for each time step"""
        # Skip if prediction is NaN
        if pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]

        # Simple long-only strategy
        if predicted_return > self.confidence_threshold and not self.position:
            self.buy()
        elif predicted_return < -self.confidence_threshold and self.position:
            self.position.close()


class SLTPStrategy(Strategy):
    """
    Stop Loss Take Profit strategy that uses AI model predictions for trading decisions.
    """
    # Strategy parameters (can be optimized)
    prediction_horizon = 1  # Which prediction to use (1, ..., `pred_len` days ahead)
    confidence_threshold = 0.01  # Minimum price change % to trigger trade
    stop_loss_pct = 0.05  # Stop loss percentage
    take_profit_pct = 0.10  # Take profit percentage
    
    def init(self):
        """Initialize strategy indicators and data"""
        pred_col = f'close_predicted_{self.prediction_horizon}'

        if pred_col not in self.data.df.columns:
            raise ValueError(f"Prediction column {pred_col} not found in data")
        
        # Calculate predicted returns indicator
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1, # Calculate predicted returns
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
    def next(self):
        """Execute strategy logic for each time step"""
        # Skip if prediction is NaN
        if pd.isna(self.predicted_returns[-1]):
            return
            
        current_price = self.data.Close[-1]
        predicted_return = self.predicted_returns[-1]
        
        # If we are not in a position, check for entry signals
        if not self.position:
            # Set stop-loss and take-profit levels based on the current price
            upper_band = current_price * (1 + self.take_profit_pct)
            lower_band = current_price * (1 - self.stop_loss_pct)

            # Long signal
            if predicted_return > self.confidence_threshold:
                self.buy(sl=lower_band, tp=upper_band) # Buy with stop-loss and take-profit
                
            # Short signal
            elif predicted_return < -self.confidence_threshold:
                # For short orders, the stop-loss is above the price
                # and take-profit is below.
                self.sell(sl=upper_band, tp=lower_band)


class MomentumAIStrategy(Strategy):
    """
    Momentum strategy: Combine AI predictions with price momentum
    """
    prediction_horizon = 1
    confidence_threshold = 0.01
    momentum_window = 5
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'

        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        # Calculate momentum indicator
        self.momentum = self.I(
            lambda close: close.pct_change(self.momentum_window),
            self.data.Close.s, # Pandas Series
            name=f'Momentum_Window_{self.momentum_window}'
        )
    
    def next(self):
        if len(self.data) < self.momentum_window + 1 or pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        momentum = self.momentum[-1]
        
        # Combine AI prediction with momentum
        if (predicted_return > self.confidence_threshold and 
            momentum > 0 and not self.position):
            self.buy()
        elif (predicted_return < -self.confidence_threshold or 
              momentum < -0.02) and self.position:
            self.position.close()


class MultiHorizonStrategy(Strategy):
    """
    Strategy using multiple prediction horizons
    """
    short_horizon = 1
    long_horizon = 5
    confidence_threshold = 0.01
    
    def init(self):
        short_col = f'close_predicted_{self.short_horizon}'
        long_col = f'close_predicted_{self.long_horizon}'
        
        self.short_pred_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[short_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.short_horizon}'
        )

        self.long_pred_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[long_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.long_horizon}'
        )
    
    def next(self):
        if (len(self.data) < 2 or 
            pd.isna(self.short_pred_returns[-1]) or 
            pd.isna(self.long_pred_returns[-1])):
            return
            
        short_pred = self.short_pred_returns[-1]
        long_pred = self.long_pred_returns[-1]
        
        # Buy when both short and long term predictions are positive
        if (short_pred > self.confidence_threshold and 
            long_pred > self.confidence_threshold and not self.position):
            self.buy()
        elif (short_pred < -self.confidence_threshold or 
              long_pred < -self.confidence_threshold) and self.position:
            self.position.close()


class VolumeAIStrategy(Strategy):
    """
    Strategy that combines AI predictions with volume analysis
    """
    prediction_horizon = 1
    confidence_threshold = 0.01
    volume_multiplier = 1.5  # Volume must be X times average
    volume_window = 10
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        # Calculate volume moving average
        self.volume_ma = self.I(
            lambda volume: volume.rolling(self.volume_window).mean(),
            self.data.Volume.s, # Pandas Series
            name=f'Volume_MA_{self.volume_window}'
        )
        
    def next(self):
        if len(self.data) < self.volume_window + 1 or pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        
        # Only trade when volume is above average (confirms conviction)
        volume_confirmed = current_volume > (avg_volume * self.volume_multiplier)
        
        if (predicted_return > self.confidence_threshold and 
            volume_confirmed and not self.position):
            self.buy()
        elif (predicted_return < -self.confidence_threshold or
              not volume_confirmed) and self.position:
            self.position.close()


class RSIAIStrategy(Strategy):
    """
    Strategy combining AI predictions with RSI indicator
    """
    prediction_horizon = 1
    confidence_threshold = 0.01
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        self.rsi = self.I(self._calculate_rsi, self.data.Close.s, self.rsi_period, name='RSI')
    
    @staticmethod
    def _calculate_rsi(close, period):
        """Simple RSI calculation"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def next(self):
        if len(self.data) < self.rsi_period + 1 or pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        current_rsi = self.rsi[-1]
        
        # Buy when AI predicts up move AND RSI shows oversold
        if (predicted_return > self.confidence_threshold and 
            current_rsi < self.rsi_oversold and not self.position):
            self.buy()
        # Sell when AI predicts down move OR RSI shows overbought
        elif ((predicted_return < -self.confidence_threshold or 
               current_rsi > self.rsi_overbought) and self.position):
            self.position.close()


class BollingerAIStrategy(Strategy):
    """
    Strategy using AI predictions with Bollinger Bands
    """
    prediction_horizon = 1
    confidence_threshold = 0.01
    bb_period = 20
    bb_std = 2
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        # Calculate Bollinger Bands
        self.bb_middle = self.I(lambda close: close.rolling(self.bb_period).mean(), self.data.Close.s, name='BB_Middle')
        bb_std_dev = self.I(lambda close: close.rolling(self.bb_period).std(), self.data.Close.s, name='BB_StdDev')
        self.bb_upper = self.bb_middle + (bb_std_dev * self.bb_std)
        self.bb_lower = self.bb_middle - (bb_std_dev * self.bb_std)
        
        # Calculate BB position (0 = lower band, 1 = upper band)
        self.bb_position = self.I(
            lambda close, lower, upper: (close - lower) / (upper - lower) if (upper - lower) != 0 else 0.5,
            self.data.Close,
            self.bb_lower,
            self.bb_upper,
            name='BB_Position'
        )
    
    def next(self):
        if len(self.data) < self.bb_period + 1 or pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        bb_pos = self.bb_position[-1]
        # current_price = self.data.Close[-1]
        
        # Buy when AI predicts up AND price near lower band (oversold)
        if (predicted_return > self.confidence_threshold and 
            bb_pos < 0.2 and not self.position):
            self.buy()
        # Sell when AI predicts down OR price near upper band (overbought)  
        elif ((predicted_return < -self.confidence_threshold or 
               bb_pos > 0.8) and self.position):
            self.position.close()


class ProbabilisticAIStrategy(Strategy):
    """
    Strategy using ensemble of predictions and probability-based decisions
    """
    prediction_horizons = [1, 2, 3]  # Use multiple horizons
    min_confidence = 0.67  # Minimum fraction of predictions agreeing
    magnitude_threshold = 0.01
    
    def init(self):
        # Load predictions for different horizons
        for horizon in self.prediction_horizons:
            pred_col = f'close_predicted_{horizon}'
            if pred_col in self.data.df.columns:
                indicator = self.I(
                    lambda predictions, close: predictions / close - 1,
                    self.data.df[pred_col],
                    self.data.Close,
                    name=f'Pred_Horizon_{horizon}'
                )
                setattr(self, f'pred_returns_{horizon}', indicator)

    def next(self):
        if len(self.data) < 2:
            return
            
        # Count predictions for each direction
        bullish_count = 0
        bearish_count = 0
        total_predictions = 0
        avg_magnitude = 0
        
        for horizon in self.prediction_horizons:
            indicator = getattr(self, f'pred_returns_{horizon}', None)
            pred_return = indicator[-1]
            if not pd.isna(pred_return):
                total_predictions += 1
                avg_magnitude += abs(pred_return)
                if pred_return > self.magnitude_threshold:
                    bullish_count += 1
                elif pred_return < -self.magnitude_threshold:
                    bearish_count += 1
        
        if total_predictions == 0:
            return
            
        avg_magnitude /= total_predictions
        bullish_confidence = bullish_count / total_predictions
        bearish_confidence = bearish_count / total_predictions
        
        # Make trading decisions based on confidence
        if (bullish_confidence >= self.min_confidence and 
            avg_magnitude > self.magnitude_threshold and not self.position):
            self.buy()
        elif (bearish_confidence >= self.min_confidence and self.position):
            self.position.close()


class MeanReversionAIStrategy(Strategy):
    """
    Mean reversion strategy using AI predictions
    """
    prediction_horizon = 1
    lookback_period = 20
    mean_reversion_threshold = 2.0  # Standard deviations from mean
    confidence_threshold = 0.01
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        # Calculate rolling mean and std
        self.price_mean = self.I(
            lambda close: close.rolling(self.lookback_period).mean(),
            self.data.Close.s,
            name=f'Price_Mean_Rolling_{self.lookback_period}'
        )
        self.price_std = self.I(
            lambda close: close.rolling(self.lookback_period).std(),
            self.data.Close.s,
            name=f'Price_StdDev_Rolling_{self.lookback_period}'
        )
        
        # Z-score (how many std devs from mean)
        self.z_score = self.I(
            lambda close, mean, std: (close - mean) / std if std != 0 else 0,
            self.data.Close,
            self.price_mean,
            self.price_std,
            name=f'Z_Score_Rolling_{self.lookback_period}'
        )
    
    def next(self):
        if len(self.data) < self.lookback_period + 1 or pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        current_z = self.z_score[-1]
        
        # Buy when price is oversold (negative z-score) AND AI predicts recovery
        if (current_z < -self.mean_reversion_threshold and 
            predicted_return > self.confidence_threshold and not self.position):
            self.buy()
        # Sell when price reverts to mean or AI predicts decline
        elif ((current_z > 0 or predicted_return < -self.confidence_threshold) 
              and self.position):
            self.position.close()


class TrendFollowingAIStrategy(Strategy):
    """
    Trend following strategy enhanced with AI predictions
    """
    prediction_horizon = 1
    confidence_threshold = 0.01
    ema_short = 12
    ema_long = 26
    
    def init(self):
        pred_col = f'close_predicted_{self.prediction_horizon}'
        self.predicted_returns = self.I(
            lambda predictions, close: predictions / close - 1,
            self.data.df[pred_col],
            self.data.Close,
            name=f'Pred_Horizon_{self.prediction_horizon}'
        )
        
        # Calculate EMAs for trend identification
        self.ema_short_line = self.I(
            lambda close: close.ewm(span=self.ema_short).mean(),
            self.data.Close.s,
            name=f'EMA_Short_{self.ema_short}'
        )
        self.ema_long_line = self.I(
            lambda close: close.ewm(span=self.ema_long).mean(),
            self.data.Close.s,
            name=f'EMA_Long_{self.ema_long}'
        )
        
        # Trend strength
        self.trend_strength = (self.ema_short_line - self.ema_long_line) / self.ema_long_line
    
    def next(self):
        if len(self.data) < max(self.ema_short, self.ema_long) + 1:
            return
            
        if pd.isna(self.predicted_returns[-1]):
            return
            
        predicted_return = self.predicted_returns[-1]
        trend_strength = self.trend_strength[-1]
        
        # Buy when uptrend confirmed by both EMA and AI
        if (trend_strength > 0 and 
            predicted_return > self.confidence_threshold and not self.position):
            self.buy()
        # Sell when downtrend or AI predicts decline
        elif ((trend_strength < 0 or predicted_return < -self.confidence_threshold) 
              and self.position):
            self.position.close()