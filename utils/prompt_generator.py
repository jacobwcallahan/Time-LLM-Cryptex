"""
Prompt generator module for creating custom prompt templates for each feature set.
These prompts explain the feature set's purpose and help guide the LLM's understanding.
"""

from typing import List, Dict, Optional
import os


PROMPT_TEMPLATES = {
    'momentum': """The Bitcoin Momentum Indicators dataset focuses on price momentum and trend-following signals for cryptocurrency price forecasting. This feature set captures short to medium-term price movements and momentum shifts through multiple temporal perspectives. It includes: closing price as the primary target, multi-period log returns (7-day and 30-day) to capture different momentum timeframes, simple moving averages (SMA) at 14-day and 50-day periods to identify trend directions, MACD signal line for trend change detection, and RSI (14-period) to identify overbought/oversold conditions. Volume trends are also included to confirm momentum. Higher momentum values indicate stronger price trends, while RSI values above 70 suggest overbought conditions and below 30 indicate oversold conditions. These indicators work together to identify continuation patterns and momentum shifts in cryptocurrency markets.""",

    'volatility': """The Bitcoin Volatility Indicators dataset emphasizes market volatility and risk metrics for cryptocurrency price prediction. Cryptocurrency markets are highly volatile, and understanding volatility regimes is crucial for accurate forecasting. This feature set includes: closing price as the target, high and low prices to capture intraday ranges, log returns for price changes, rolling standard deviation of returns over 14-day and 30-day windows to measure realized volatility, Average True Range (ATR) which captures intraday volatility accounting for gaps, Bollinger Bands position (%B) indicating where price sits relative to volatility-adjusted bands, and high-low range normalized by close price. Higher volatility values indicate increased market uncertainty and larger price swings, while ATR and Bollinger Bands help identify volatility breakouts and mean reversion opportunities.""",

    'onchain_price': """The Bitcoin On-Chain Fundamentals dataset combines price data with blockchain network activity metrics to capture both market dynamics and underlying network fundamentals. On-chain metrics often lead price movements, providing early signals of network adoption and economic activity. This feature set includes: closing price and returns as price signals, daily transaction count and its growth rate reflecting network economic activity, unique active addresses and address growth rate indicating user adoption and network participation, trading volume for market activity, and volatility measures. Higher transaction counts and address growth typically indicate increasing network usage and adoption, which historically correlates with price appreciation. The combination of price momentum with on-chain fundamentals provides a holistic view of both market sentiment and network health.""",

    'volume_price': """The Bitcoin Volume-Price Dynamics dataset focuses on volume analysis and price-volume relationships for cryptocurrency forecasting. Volume confirms price movements and strong volume-price relationships indicate trend strength. This feature set includes: closing price and returns as price signals, On-Balance Volume (OBV) which accumulates volume based on price direction, volume ratios comparing current volume to moving averages, volume momentum showing rate of change in trading activity, volume-price trend combining volume with price changes, rolling correlations between price and volume, and volume-weighted average price (VWAP). Rising OBV and positive volume-price correlations suggest increasing buying pressure, while falling OBV indicates selling pressure. These indicators help identify whether price movements are supported by volume or may reverse.""",

    'technical': """The Bitcoin Technical Analysis Indicators dataset provides a comprehensive suite of technical indicators commonly used in cryptocurrency trading. Technical indicators help identify trends, momentum shifts, and potential reversal points through mathematical analysis of price and volume. This feature set includes: closing, high, and low prices, multiple timeframe returns (7-day log returns), Relative Strength Index (RSI) for momentum (0-100 scale, where >70 is overbought, <30 is oversold), MACD signal line for trend changes (centered around 0, positive indicates bullish momentum), Simple Moving Averages at 14-day and 50-day periods with price ratios to identify trend position, Bollinger Bands %B (0-1 scale indicating position within volatility bands), Stochastic Oscillator %K for momentum (0-100 scale), volume ratios, and volatility measures. These indicators provide diverse perspectives on market conditions, each with distinct statistical properties and interpretation ranges.""",

    'hybrid': """The Bitcoin Hybrid Volatility-OnChain dataset combines short-term volatility measures with medium-term fundamental on-chain data for cryptocurrency price prediction. This feature set integrates volatility patterns with network fundamentals to capture both market dynamics and underlying adoption trends. It includes: closing price and returns as price signals, rolling volatility measures (14-day and 30-day standard deviation) and Average True Range (ATR) for volatility assessment, daily transaction counts and growth rates showing network economic activity, unique address counts and growth rates indicating user adoption momentum, and trading volume. By combining volatility indicators that measure short-term price dispersion with on-chain metrics that reflect longer-term network health, this feature set enables the model to capture both market uncertainty and fundamental trends that often drive cryptocurrency price movements.""",

    'returns': """The Bitcoin Returns Analysis dataset provides deep analysis of returns patterns, momentum, and mean reversion signals across multiple timeframes. Returns are fundamental to price prediction, and understanding returns dynamics at different scales is critical. This feature set includes: closing price as the target, multi-period log returns (1-day, 7-day, 14-day, 30-day) capturing different momentum patterns, rolling volatility of returns measuring price dispersion at different windows, price position relative to moving averages indicating mean reversion opportunities, volume patterns during positive versus negative return periods, and RSI for momentum confirmation. Returns at different periods reveal distinct patterns: short-term returns capture noise and momentum, while longer-term returns reflect structural changes. Volume patterns during different return regimes help identify whether moves are sustainable or likely to reverse.""",

    'minimal': """The Bitcoin Minimal High-Value Indicators dataset represents a carefully selected subset of features with maximum information content and statistical diversity. Each feature has been chosen to provide unique predictive signals while minimizing redundancy. This optimized feature set includes: closing price as the target, carefully selected return periods (7-day and 14-day) that capture distinct momentum patterns, volatility measures (14-day rolling standard deviation) that provide unique risk signals, the most informative on-chain metric (either transaction or address growth, selected based on mutual information), volume indicators (volume ratio) with distinct statistical properties, and momentum oscillators (RSI 14-period) that operate at different scales. Despite its small size, this feature set maintains high predictive power by ensuring each feature contributes unique information and generates distinct statistical patterns for the model's prompt generation.""",

    'temporal': """The Bitcoin Temporal Pattern Diversity dataset captures price dynamics across multiple time scales and measurement approaches. By including features with different temporal characteristics and statistical properties, the model can learn patterns at various timeframes simultaneously. This feature set includes: closing price as the target, short-term returns (7-day) and long-term returns (30-day) capturing different momentum patterns, rolling volatility (14-day) measuring price dispersion, volume ratios indicating trading activity relative to historical norms, RSI (14-period) as a normalized momentum oscillator, transaction growth rates from on-chain metrics showing network activity trends, and address growth rates indicating user adoption momentum. Each feature operates at different time scales and has distinct statistical properties (ranges, distributions, trend patterns), allowing the model to capture both short-term noise and long-term structural changes in cryptocurrency markets."""
}


FEATURE_DESCRIPTIONS = {
    'close': 'closing price',
    'open': 'opening price',
    'high': 'high price',
    'low': 'low price',
    'volume': 'trading volume',
    'returns_1d': '1-day log returns',
    'returns_7d': '7-day log returns',
    'returns_14d': '14-day log returns',
    'returns_30d': '30-day log returns',
    'volatility_14d': '14-day rolling standard deviation of returns',
    'volatility_30d': '30-day rolling standard deviation of returns',
    'atr_14': '14-day Average True Range',
    'bb_pctb_14': 'Bollinger Bands %B position (14-day)',
    'rsi_14': '14-period Relative Strength Index',
    'macd_signal': 'MACD signal line',
    'sma_14': '14-day Simple Moving Average',
    'sma_50': '50-day Simple Moving Average',
    'transactions_growth': 'transaction count growth rate',
    'addresses_growth': 'unique address growth rate',
    'volume_ratio_14': 'volume ratio to 14-day moving average',
}


def generate_feature_set_prompt(feature_set_name: str, 
                                features: List[str],
                                feature_descriptions: Optional[Dict[str, str]] = None) -> str:
    """
    Generate custom prompt template for a feature set.
    
    Args:
        feature_set_name: Name of feature set (e.g., 'momentum', 'volatility')
        features: List of feature names in the set
        feature_descriptions: Optional dict mapping feature names to descriptions
    
    Returns:
        Prompt template text
    """
    # Use pre-defined template if available
    if feature_set_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[feature_set_name]
    
    # Otherwise generate generic prompt
    descriptions = feature_descriptions or FEATURE_DESCRIPTIONS
    
    feature_list = []
    for feat in features[:10]:  # Limit to first 10 for brevity
        desc = descriptions.get(feat, feat)
        feature_list.append(f"{feat} ({desc})")
    
    if len(features) > 10:
        feature_list.append(f"... and {len(features) - 10} more features")
    
    prompt = f"""The Bitcoin {feature_set_name.replace('_', ' ').title()} dataset focuses on specific aspects of cryptocurrency price forecasting. This feature set includes: {', '.join(feature_list)}. These features have been selected to provide diverse statistical properties and capture different aspects of market dynamics for accurate price prediction."""
    
    return prompt


def save_prompt_template(prompt_text: str, 
                        feature_set_name: str,
                        output_dir: str = 'dataset/prompt_bank/') -> str:
    """
    Save prompt template to file.
    
    Args:
        prompt_text: Prompt text to save
        feature_set_name: Name of feature set
        output_dir: Output directory for prompt files
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f'CRYPTEX_features_{feature_set_name}.txt'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(prompt_text)
    
    return filepath


def generate_and_save_prompt(feature_set_name: str,
                             features: List[str],
                             feature_descriptions: Optional[Dict[str, str]] = None,
                             output_dir: str = 'dataset/prompt_bank/') -> str:
    """
    Generate and save prompt template for a feature set.
    
    Args:
        feature_set_name: Name of feature set
        features: List of feature names
        feature_descriptions: Optional feature descriptions
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    prompt_text = generate_feature_set_prompt(feature_set_name, features, feature_descriptions)
    filepath = save_prompt_template(prompt_text, feature_set_name, output_dir)
    return filepath



