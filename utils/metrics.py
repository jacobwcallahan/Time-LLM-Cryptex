import numpy as np
import torch
from torch import nn

def get_loss_function(loss_name):
    # Returns an instance of the specified loss function.
    loss_name = loss_name.upper()
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAE':
        return nn.L1Loss()
    elif loss_name == 'MAPE':
        return MAPELoss()
    elif loss_name == 'MADL':
        return MADLLoss()
    elif loss_name == 'GMADL':
        return GMADLLoss()
    elif loss_name == 'MADLSTE':
        return MADLLossSTE()
    elif loss_name == 'TRADING' or loss_name == 'SHARPE':
        return DifferentiableTradingLoss(metric='sharpe')
    elif loss_name == 'SORTINO':
        return DifferentiableTradingLoss(metric='sortino')
    elif loss_name == 'TRADING_RETURNS':
        return DifferentiableTradingLoss(metric='returns')
    else:
        raise ValueError(f"Unsupported loss type: {loss_name}")

def get_metric_function(metric_name):
    # Returns an instance of the specified evaluation metric.
    metric_name = metric_name.upper()
    if metric_name == 'MSE':
        return nn.MSELoss()
    elif metric_name == 'MAE':
        return nn.L1Loss()
    elif metric_name == 'MAPE':
        return MAPELoss()
    elif metric_name == 'MDA':
        return MDAMetric()
    elif metric_name == 'SHARPE':
        return SharpeRatioMetric()
    elif metric_name == 'MADLSTE':
        return MADLLossSTE()
    else:
        raise ValueError(f"Unsupported metric type: {metric_name}")


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error:
        MAPE = mean( |[pred - true] / true| )
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps # to avoid division by zero

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # ensure true is nonzero
        denom = torch.where(torch.abs(true) < self.eps,
                            torch.full_like(true, self.eps),
                            true)
        return torch.mean(torch.abs((pred - true) / denom))


class MDAMetric(nn.Module):
    """
    Mean Directional Accuracy (MDA)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # pred, true shape: [batch, seq_len, feature_dim] or [batch, seq_len]
        # Compare change direction relative to previous timestep

        if pred.shape[1] < 2:
            print(f"[Warning] Not enough steps to compute MDA. pred.shape: {pred.shape}")
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]

        correct = (pred_diff * true_diff) > 0  # boolean tensor: True if same direction
        mda = correct.float().mean()  # take mean accuracy over all elements

        return mda

class SharpeRatioMetric(nn.Module):
    """
    Computes the Sharpe Ratio for a batch of returns (predictions).
    Assumes risk-free rate = 0.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # to avoid division by zero

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true shape: [batch, seq_len, feature_dim] or [batch, seq_len]
        Computes over the prediction period only.
        """
        # calculate returns as diff relative to previous timestep
        returns = pred[:, 1:] - pred[:, :-1]

        mean_return = returns.mean()
        std_return = returns.std()

        # prevent divide-by-zero
        sharpe_ratio = mean_return / (std_return + self.eps)

        return sharpe_ratio

class MADLLoss(nn.Module):
    # Mean Absolute Directional Loss (MADL) by F Michankow (2023)
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [batch, seq_len] or [batch, seq_len, 1] (predicted and true returns)
        """
        # Ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        product_sign = torch.sign(true * pred)  # sign(Ri * R̂i)
        abs_return = torch.abs(true)

        loss = (-1.0) * product_sign * abs_return

        return loss.mean()

class GMADLLoss(nn.Module):
    # Generalized Mean Absolute Directional Loss (GMADL) by F. Michankow (2024)
    def __init__(self, a=1.0, b=1.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [batch, seq_len] or [batch, seq_len, 1] (predicted and true returns)
        """
        # ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        # The paper uses a=1000 and b=1:5
        product = self.a * true * pred  # element-wise Ri * R̂i

        sigmoid_term = 1.0 / (1.0 + torch.exp(-product))  # 1 / (1 + exp(-a Ri R̂i))

        adjustment = sigmoid_term - 0.5  # ( ... ) - 0.5

        weighted_abs_return = torch.abs(true) ** self.b  # |Ri|^b

        loss = -1.0 * adjustment * weighted_abs_return

        # Mean over all elements
        return loss.mean()

class MADLLossSTE(nn.Module):
    """
    Mean Absolute Directional Loss with Straight-Through Estimator (MADL-STE)
    Forward pass: uses sign(x) for directional accuracy
    Backward pass: uses identity gradient to enable learning
    
    This solves the zero-gradient problem of the original MADL implementation
    while maintaining the same forward behavior.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [batch, seq_len] or [batch, seq_len, 1] (predicted and true returns)
        """
        # Ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
        
        product = true * pred  # Element-wise Ri * R̂i
        product_sign = torch.sign(product)  # sign(Ri * R̂i)
        
        # Straight-through estimator: forward uses sign, backward uses product gradient
        # This gives the sign behavior in forward but non-zero gradients in backward
        product_sign_ste = product_sign.detach() + product - product.detach()
        
        abs_return = torch.abs(true)
        loss = (-1.0) * product_sign_ste * abs_return
        
        return loss.mean()

class MDAChange(nn.Module):
    """
    Computes accuracy of price changing direction over the prediction period.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        
        pred, true: [batch, seq_len] (predicted and true prices)
        """
        pass


class DifferentiableTradingLoss(nn.Module):
    """
    Differentiable approximation of backtester performance.
    
    Instead of discrete buy/sell decisions, uses soft positions via tanh
    to maintain gradient flow during backpropagation.
    
    The model learns to maximize trading performance (Sharpe ratio) directly,
    rather than minimizing prediction error.
    
    Args:
        confidence_threshold: Minimum expected return to trigger a position (default 0.01 = 1%)
        transaction_cost: Cost per unit of position change (default 0.001 = 0.1%)
        temperature: Controls sharpness of position decisions (higher = more binary-like)
        metric: Trading metric to optimize ('sharpe', 'sortino', 'returns')
        annualization_factor: Factor to annualize returns (e.g., 252 for daily, 365*24 for hourly)
        eps: Small constant for numerical stability
    """
    def __init__(
        self, 
        confidence_threshold: float = 0.01,
        transaction_cost: float = 0.001,
        temperature: float = 10.0,
        metric: str = 'sharpe',
        annualization_factor: float = 252.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.transaction_cost = transaction_cost
        self.temperature = temperature
        self.metric = metric.lower()
        self.annualization_factor = annualization_factor
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable trading loss.
        
        Args:
            pred: [batch, pred_len, features] - predicted prices
            true: [batch, pred_len, features] - actual prices
        
        Returns:
            Negative trading metric (to minimize = maximize performance)
        """
        # Need at least 2 timesteps to compute returns
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Calculate actual returns from true prices
        actual_returns = (true[:, 1:] - true[:, :-1]) / (true[:, :-1].abs() + self.eps)
        
        # Calculate predicted returns (what the model expects to happen)
        predicted_returns = (pred[:, 1:] - pred[:, :-1]) / (pred[:, :-1].abs() + self.eps)
        
        # Soft position sizing using tanh for differentiability
        # Maps predicted return to position in [-1, 1] (short to long)
        # temperature controls how "sharp" the decision boundary is
        positions = torch.tanh(
            self.temperature * predicted_returns / (self.confidence_threshold + self.eps)
        )
        
        # Strategy returns = position * actual_return
        # If we predict up (position > 0) and price goes up (return > 0), we profit
        strategy_returns = positions * actual_returns
        
        # Account for transaction costs based on position changes
        if positions.shape[1] > 1:
            position_changes = torch.abs(positions[:, 1:] - positions[:, :-1])
            # Pad first timestep with initial position change (from 0 to first position)
            initial_change = torch.abs(positions[:, :1])
            position_changes = torch.cat([initial_change, position_changes], dim=1)
        else:
            position_changes = torch.abs(positions)
        
        transaction_costs = position_changes * self.transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Flatten for metric calculation
        net_returns_flat = net_returns.reshape(net_returns.shape[0], -1)
        
        # Calculate the chosen metric
        if self.metric == 'sharpe':
            loss = self._negative_sharpe(net_returns_flat)
        elif self.metric == 'sortino':
            loss = self._negative_sortino(net_returns_flat)
        elif self.metric == 'returns':
            loss = -net_returns_flat.sum(dim=1).mean()  # Negative total return
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return loss
    
    def _negative_sharpe(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute negative Sharpe ratio (we minimize, so negative = maximize Sharpe).
        
        Args:
            returns: [batch, num_returns] - strategy returns per timestep
        
        Returns:
            Negative annualized Sharpe ratio (scalar)
        """
        mean_return = returns.mean(dim=1)
        std_return = returns.std(dim=1) + self.eps
        
        # Annualize
        sharpe = (mean_return / std_return) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=returns.device)
        )
        
        return -sharpe.mean()
    
    def _negative_sortino(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute negative Sortino ratio (penalizes only downside volatility).
        
        Args:
            returns: [batch, num_returns] - strategy returns per timestep
        
        Returns:
            Negative annualized Sortino ratio (scalar)
        """
        mean_return = returns.mean(dim=1)
        
        # Downside deviation: std of negative returns only
        negative_returns = torch.clamp(returns, max=0)
        downside_std = torch.sqrt((negative_returns ** 2).mean(dim=1)) + self.eps
        
        # Annualize
        sortino = (mean_return / downside_std) * torch.sqrt(
            torch.tensor(self.annualization_factor, device=returns.device)
        )
        
        return -sortino.mean()


class DifferentiableTradingLossWithContext(DifferentiableTradingLoss):
    """
    Extended trading loss that uses the input sequence context for position decisions.
    
    This version takes the last price from the input sequence to compute
    the expected return for the first prediction, making it more realistic.
    
    Use this when you have access to both input and output sequences.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward_with_context(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor,
        last_input_price: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trading loss with context from input sequence.
        
        Args:
            pred: [batch, pred_len, features] - predicted prices
            true: [batch, pred_len, features] - actual prices  
            last_input_price: [batch, 1, features] - last price from input sequence
        
        Returns:
            Negative trading metric
        """
        # Prepend last input price to create full price series
        pred_full = torch.cat([last_input_price, pred], dim=1)
        true_full = torch.cat([last_input_price, true], dim=1)
        
        return self.forward(pred_full, true_full)