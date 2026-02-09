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
        return TradingLoss()
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


class TradingLoss(nn.Module):
    """
    Trading-oriented loss function combining MSE with risk-adjusted directional penalty.
    Penalizes wrong trading decisions proportionally to risk-adjusted loss magnitude.
    
    Formula: loss = alpha * MSE(pred, true) + beta * risk_adj_penalty * sharpe_weight
    
    Args:
        alpha: Weight for MSE component (default: 1.0)
        beta: Weight for directional penalty component (default: 1.0)
        eps: Small epsilon to avoid division by zero (default: 1e-8)
    """
    def __init__(self, alpha=1.0, beta=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, true: torch.Tensor, input_data: torch.Tensor = None) -> torch.Tensor:
        """
        Compute trading loss with risk-adjusted returns.
        
        Args:
            pred: [batch, pred_len] or [batch, pred_len, feature_dim] - predicted values
            true: [batch, pred_len] or [batch, pred_len, feature_dim] - true values
            input_data: [batch, seq_len] or [batch, seq_len, feature_dim] - input sequence (optional)
                        Used to compute trading returns from entry price. If None, falls back to MSE-only for pred_len=1.
            
        Returns:
            Scalar loss tensor
        """
        # Ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        # MSE component - standard regression loss to narrow down predictions
        mse_loss = torch.mean((pred - true) ** 2)

        # If input_data is not provided, use fallback logic
        if input_data is None:
            # For pred_len=1, can't compute trading returns without input, return MSE-only
            if pred.shape[1] == 1:
                return self.alpha * mse_loss
            
            # For pred_len>1, use returns-based logic as fallback (smooth directional penalty)
            if pred.dim() == 2:
                pred_returns = pred[:, 1:] - pred[:, :-1]
                true_returns = true[:, 1:] - true[:, :-1]
            else:
                pred_returns = pred[:, 1:, :] - pred[:, :-1, :]
                true_returns = true[:, 1:, :] - true[:, :-1, :]
            
            smooth_wrong = torch.relu(-pred_returns * true_returns)
            return_error = torch.abs(pred_returns - true_returns)
            directional_penalty = (smooth_wrong * return_error).mean()
            
            true_returns_flat = true_returns.flatten()
            mean_return = true_returns_flat.mean()
            std_return = torch.clamp(true_returns_flat.std() + self.eps, min=1e-4)
            sharpe_like = mean_return / std_return
            sharpe_weight = torch.clamp(1.0 / (1.0 + torch.abs(sharpe_like)), min=0.1, max=1.0)
            
            return self.alpha * mse_loss + self.beta * directional_penalty * sharpe_weight

        # Compute trading returns from last input price (entry point for trade)
        # Handle both 2D and 3D cases
        if pred.dim() == 2:
            # 2D: [batch, pred_len]
            last_price = input_data[:, -1:]  # [batch, 1]
            pred_final = pred[:, -1:]  # [batch, 1] - final prediction
            true_final = true[:, -1:]  # [batch, 1] - final true value
        else:
            # 3D: [batch, pred_len, feature_dim]
            last_price = input_data[:, -1:, :]  # [batch, 1, feature_dim]
            pred_final = pred[:, -1:, :]  # [batch, 1, feature_dim]
            true_final = true[:, -1:, :]  # [batch, 1, feature_dim]

        # Compute trading directions and returns
        pred_dir = pred_final - last_price  # Predicted direction from entry
        actual_return = true_final - last_price  # Actual return from trade (can be positive or negative)

        # Smooth wrong-direction surrogate (differentiable)
        smooth_wrong = torch.relu(-pred_dir * actual_return)

        # Compute risk measure (volatility of actual returns across batch), with floor for stability
        actual_return_flat = actual_return.flatten()
        risk = torch.clamp(actual_return_flat.std() + self.eps, min=1e-4)

        # Compute risk-adjusted return (Sharpe-like)
        risk_adj_return = actual_return / risk  # [batch, 1] or [batch, 1, feature_dim]

        # Risk-adjusted penalty: wrong-direction weighted by risk-adjusted magnitude (smooth)
        risk_adj_penalty = (smooth_wrong * torch.abs(risk_adj_return)).mean()

        # Sharpe-like weighting computed on actual returns; clamp so penalty always contributes
        mean_return = actual_return_flat.mean()
        std_return = actual_return_flat.std()
        sharpe_like = mean_return / (std_return + self.eps)
        sharpe_weight = torch.clamp(1.0 / (1.0 + torch.abs(sharpe_like)), min=0.1, max=1.0)

        # Final loss: MSE + risk-adjusted directional penalty weighted by Sharpe
        loss = self.alpha * mse_loss + self.beta * risk_adj_penalty * sharpe_weight

        return loss