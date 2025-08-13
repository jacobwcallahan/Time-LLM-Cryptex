import backtrader as bt
import numpy as np

class BaseAIStrategy(bt.Strategy):
    """Base strategy class with common AI prediction functionality"""
    
    params = (
        ('prediction_horizon', 1), # Which prediction to use (1, ..., `pred_len` days ahead)
        ('confidence_threshold', 0.01), # Minimum price change % to trigger trade
        ('position_size', 0.99),  # Percent of available cash to use
    )
    
    def __init__(self):
        # Add prediction data as lines
        self.prediction = self.datas[0].close_predicted_1  # Default to horizon 1
        
        # Set the correct prediction horizon
        horizon = self.params.prediction_horizon
        if hasattr(self.datas[0], f'close_predicted_{horizon}'):
            self.prediction = getattr(self.datas[0], f'close_predicted_{horizon}')
        
        # Trick to include raw predictions in the plot
        pred_plot = bt.indicators.SimpleMovingAverage(self.prediction, period=1, plotname=f'Raw Prediction {self.params.prediction_horizon}')
    
    def get_prediction_signal(self):
        """Get trading signal based on AI predictions"""

        current_price = self.data.close[0]
        predicted_price = self.prediction[0]  # Get the predicted price for the next period
        
        if np.isnan(predicted_price):
            return 0
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Generate signal based on confidence threshold
        if expected_return > self.params.confidence_threshold:
            return 1  # Buy signal
        elif expected_return < -self.params.confidence_threshold:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def get_position_size(self):
        """Calculate position size based on available cash"""
        cash = self.broker.get_cash()
        price = self.data.close[0]
        size = (cash / price) * self.params.position_size
        return round(size, 8)

class SimpleAIStrategy(BaseAIStrategy):
    """Simple AI strategy that trades based on predictions"""
    
    def next(self):
        signal = self.get_prediction_signal()
        
        if not self.position:
            if signal == 1:  # Buy signal
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            if signal == -1:  # Sell signal
                self.close()

class SLTPStrategy(BaseAIStrategy):
    """AI strategy with stop loss and take profit"""

    params = (
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.15),
    )

    def __init__(self):
        super().__init__()
        self.buy_price = None

    def next(self):
        signal = self.get_prediction_signal()
        current_price = self.data.close[0]

        if not self.position:
            if signal == 1:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
                    self.buy_price = current_price
        else:
            # Check stop loss and take profit
            loss_pct = (self.buy_price - current_price) / self.buy_price
            profit_pct = (current_price - self.buy_price) / self.buy_price

            if loss_pct >= self.params.stop_loss_pct or profit_pct >= self.params.take_profit_pct:
                self.close()
                self.buy_price = None

class MomentumAIStrategy(BaseAIStrategy):
    """AI strategy combined with momentum indicator"""
    
    params = (
        ('momentum_window', 20),
    )
    
    def __init__(self):
        super().__init__()
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_window)
    
    def next(self):
        signal = self.get_prediction_signal()
        
        # Only trade if momentum aligns with prediction
        momentum_positive = self.momentum[0] > 0
        
        if not self.position:
            if signal == 1 and momentum_positive:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            if signal == -1 or not momentum_positive:
                self.close()

class RSIAIStrategy(BaseAIStrategy):
    """AI strategy combined with RSI"""
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
    
    def next(self):
        signal = self.get_prediction_signal()
        rsi_value = self.rsi[0]
        
        if not self.position:
            # Buy if AI predicts up and RSI not overbought
            if signal == 1 and rsi_value < self.params.rsi_overbought:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            # Sell if AI predicts down or RSI overbought
            if signal == -1 or rsi_value > self.params.rsi_overbought:
                self.close()

class BollingerAIStrategy(BaseAIStrategy):
    """AI strategy combined with Bollinger Bands"""
    
    params = (
        ('bb_period', 20),
        ('bb_std', 2.0),
    )
    
    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(self.data.close, 
                                               period=self.params.bb_period,
                                               devfactor=self.params.bb_std)
    
    def next(self):
        signal = self.get_prediction_signal()
        price = self.data.close[0]
        
        if not self.position:
            # Buy if AI predicts up and price near lower band
            if signal == 1 and price <= self.bb.lines.bot[0]:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            # Sell if AI predicts down or price near upper band
            if signal == -1 or price >= self.bb.lines.top[0]:
                self.close()

class MeanReversionAIStrategy(BaseAIStrategy):
    """Mean reversion strategy with AI predictions"""
    
    params = (
        ('lookback_period', 20),
        ('mean_reversion_threshold', 1.5),
    )
    
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.lookback_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.params.lookback_period)
    
    def next(self):
        signal = self.get_prediction_signal()
        price = self.data.close[0]
        
        if len(self.sma) < 1:
            return
        
        mean = self.sma[0]
        std = self.std[0]
        
        # Check if price is far from mean
        z_score = abs(price - mean) / std if std > 0 else 0
        
        if not self.position:
            # Buy if AI predicts up and price below mean
            if signal == 1 and price < mean and z_score > self.params.mean_reversion_threshold:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            # Sell if price returns to mean or AI predicts down
            if signal == -1 or z_score < 0.5:
                self.close()

class TrendFollowingAIStrategy(BaseAIStrategy):
    """Trend following strategy with AI predictions"""
    
    params = (
        ('ema_short', 5),
        ('ema_long', 20),
    )
    
    def __init__(self):
        super().__init__()
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.params.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.ema_long)
    
    def next(self):
        signal = self.get_prediction_signal()
        
        # Check trend direction
        trend_up = self.ema_short[0] > self.ema_long[0]
        
        if not self.position:
            # Buy if AI predicts up and trend is up
            if signal == 1 and trend_up:
                size = self.get_position_size()
                if size > 0:
                    self.buy(size=size)
        else:
            # Sell if AI predicts down or trend changes
            if signal == -1 or not trend_up:
                self.close()