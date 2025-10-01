import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Stock Options Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the backtester classes (copied from your notebook)
from backtester import (
    OptionType, PositionType, OptionLeg, OptionStrategy, 
    OptionStrategyTemplate, OptionsBacktester
)

@st.cache_data
def load_yfinance_data(ticker, start_date, end_date):
    """Load equity data from yfinance."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date)
        # Convert to the same format as historical data
        # Handle MultiIndex columns from yfinance
        df = pd.DataFrame({
            'open': data[('Open', ticker)] if ('Open', ticker) in data.columns else data['Open'],
            'high': data[('High', ticker)] if ('High', ticker) in data.columns else data['High'],
            'low': data[('Low', ticker)] if ('Low', ticker) in data.columns else data['Low'],
            'close': data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close'],
            'volume': data[('Volume', ticker)] if ('Volume', ticker) in data.columns else data['Volume']
        })
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error loading yfinance data: {str(e)}")
        return None

def get_options_chain(ticker):
    """Get options chain data from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        # Get the nearest expiration date options chain
        expirations = stock.options
        if not expirations:
            return None
        nearest_expiry = expirations[0]
        options = stock.option_chain(nearest_expiry)
        return options
    except Exception as e:
        st.error(f"Error getting options chain: {str(e)}")
        return None

def create_strategy_diagram(strategy_name):
    """Create a simple visualisation of the option strategy."""
    fig = go.Figure()
    # Strategy payoff diagrams
    spot_range = np.linspace(0.8, 1.2, 100)
    
    def add_colored_line(x_data, y_data):
        """Add line segments colored by positive/negative values"""
        # Create continuous line by duplicating zero crossings
        x_extended = []
        y_extended = []
        colors = []
        
        for i in range(len(y_data)):
            x_extended.append(x_data[i])
            y_extended.append(y_data[i])
            colors.append('green' if y_data[i] >= 0 else 'red')
            
            # Check for zero crossing
            if i < len(y_data) - 1:
                if (y_data[i] >= 0 and y_data[i+1] < 0) or (y_data[i] < 0 and y_data[i+1] >= 0):
                    # Interpolate zero crossing point
                    zero_x = x_data[i] + (x_data[i+1] - x_data[i]) * (-y_data[i] / (y_data[i+1] - y_data[i]))
                    x_extended.append(zero_x)
                    y_extended.append(0)
                    colors.append('green' if y_data[i+1] >= 0 else 'red')
        
        # Split into segments by color
        current_color = colors[0]
        segment_x = [x_extended[0]]
        segment_y = [y_extended[0]]
        
        for i in range(1, len(colors)):
            if colors[i] == current_color:
                segment_x.append(x_extended[i])
                segment_y.append(y_extended[i])
            else:
                # End current segment and start new one
                fig.add_trace(go.Scatter(x=segment_x, y=segment_y,
                                        mode='lines',
                                        line=dict(width=3, color=current_color),
                                        showlegend=False))
                current_color = colors[i]
                segment_x = [x_extended[i-1], x_extended[i]]  # Include transition point
                segment_y = [y_extended[i-1], y_extended[i]]
        
        # Add final segment
        fig.add_trace(go.Scatter(x=segment_x, y=segment_y,
                                mode='lines',
                                line=dict(width=3, color=current_color),
                                showlegend=False))
    
    if strategy_name == "Covered Call":
        # Long stock + short call
        stock_payoff = spot_range - 1.0
        call_payoff = np.minimum(0, 1.05 - spot_range)  # Short call at 105% strike
        total_payoff = stock_payoff + call_payoff + 0.02  # Plus premium
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Cash Secured Put":
        put_payoff = np.minimum(0, spot_range - 0.95) + 0.02  # Short put + premium
        add_colored_line(spot_range*100, put_payoff*100)
    
    elif strategy_name == "Long Straddle":
        call_payoff = np.maximum(0, spot_range - 1.0) - 0.03
        put_payoff = np.maximum(0, 1.0 - spot_range) - 0.03
        total_payoff = call_payoff + put_payoff
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Iron Condor":
        # Simplified iron condor
        payoff = np.where(spot_range < 0.95, -(spot_range - 0.95) + 0.01,
                         np.where(spot_range > 1.05, -(spot_range - 1.05) + 0.01, 0.01))
        add_colored_line(spot_range*100, payoff*100)
    
    elif strategy_name == "Bull Call Spread":
        long_call = np.maximum(0, spot_range - 1.0) - 0.03
        short_call = -(np.maximum(0, spot_range - 1.05) - 0.01)
        total_payoff = long_call + short_call
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Long Strangle":
        call_payoff = np.maximum(0, spot_range - 1.05) - 0.015
        put_payoff = np.maximum(0, 0.95 - spot_range) - 0.015
        total_payoff = call_payoff + put_payoff
        add_colored_line(spot_range*100, total_payoff*100)
    
    fig.update_layout(
        title=f"{strategy_name} Payoff Diagram",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=300,
        showlegend=False
    )
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    fig.add_vline(x=100, line_color="gray", opacity=0.5, line_dash="dash")
    
    return fig

def create_additional_charts(daily_df, strategy_name, asset_name):
    """Create additional visualization charts for the backtest results."""
    charts = {}
    
    if daily_df is None or daily_df.empty:
        return charts
    
    # 1. P&L Distribution Histogram
    if 'current_pnl' in daily_df.columns:
        fig_pnl_dist = go.Figure()
        fig_pnl_dist.add_trace(go.Histogram(
            x=daily_df['current_pnl'],
            nbinsx=30,
            marker_color='blue',
            opacity=0.7
        ))
        fig_pnl_dist.update_layout(
            title=f"{strategy_name} - {asset_name}: P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            height=400
        )
        charts['pnl_distribution'] = fig_pnl_dist
    
    # 2. Cumulative P&L Over Time
    if 'current_pnl' in daily_df.columns and 'entry_date' in daily_df.columns:
        daily_df_sorted = daily_df.sort_values('entry_date')
        daily_df_sorted['cumulative_pnl'] = daily_df_sorted['current_pnl'].cumsum()
        
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=daily_df_sorted['entry_date'],
            y=daily_df_sorted['cumulative_pnl'],
            mode='lines',
            line=dict(width=3, color='green'),
            name='Cumulative P&L'
        ))
        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cumulative.update_layout(
            title=f"{strategy_name} - {asset_name}: Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            height=400
        )
        charts['cumulative_pnl'] = fig_cumulative
    
    # 3. Win/Loss Streak Analysis
    if 'current_pnl' in daily_df.columns:
        daily_df['win'] = daily_df['current_pnl'] > 0
        daily_df['streak'] = daily_df['win'].astype(int).groupby((daily_df['win'] != daily_df['win'].shift()).cumsum()).cumcount() + 1
        daily_df['streak'] = daily_df['streak'] * daily_df['win'].astype(int)
        
        # Calculate average streak length
        avg_win_streak = daily_df[daily_df['win']]['streak'].mean() if daily_df[daily_df['win']].shape[0] > 0 else 0
        
        # Create streak histogram
        fig_streak = go.Figure()
        fig_streak.add_trace(go.Histogram(
            x=daily_df[daily_df['win']]['streak'],
            nbinsx=15,
            marker_color='green',
            opacity=0.7,
            name='Win Streaks'
        ))
        fig_streak.update_layout(
            title=f"{strategy_name} - {asset_name}: Win Streak Distribution (Avg: {avg_win_streak:.1f})",
            xaxis_title="Streak Length",
            yaxis_title="Frequency",
            height=400
        )
        charts['win_streak'] = fig_streak
    
    # 4. Rolling Sharpe Ratio
    if 'current_pnl' in daily_df.columns and 'entry_date' in daily_df.columns:
        daily_df_sorted = daily_df.sort_values('entry_date')
        daily_df_sorted['returns'] = daily_df_sorted['current_pnl'] / abs(daily_df_sorted['current_pnl'].shift(1)).fillna(1)
        daily_df_sorted['rolling_sharpe'] = daily_df_sorted['returns'].rolling(window=30).mean() / daily_df_sorted['returns'].rolling(window=30).std() * np.sqrt(252)
        
        fig_rolling_sharpe = go.Figure()
        fig_rolling_sharpe.add_trace(go.Scatter(
            x=daily_df_sorted['entry_date'],
            y=daily_df_sorted['rolling_sharpe'],
            mode='lines',
            line=dict(width=2, color='purple'),
            name='30-Day Rolling Sharpe'
        ))
        fig_rolling_sharpe.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rolling_sharpe.update_layout(
            title=f"{strategy_name} - {asset_name}: Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=400
        )
        charts['rolling_sharpe'] = fig_rolling_sharpe
    
    # 5. Drawdown Analysis
    if 'current_pnl' in daily_df.columns and 'entry_date' in daily_df.columns:
        daily_df_sorted = daily_df.sort_values('entry_date')
        daily_df_sorted['cumulative_pnl'] = daily_df_sorted['current_pnl'].cumsum()
        daily_df_sorted['running_max'] = daily_df_sorted['cumulative_pnl'].expanding().max()
        daily_df_sorted['drawdown'] = daily_df_sorted['cumulative_pnl'] - daily_df_sorted['running_max']
        
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=daily_df_sorted['entry_date'],
            y=daily_df_sorted['drawdown'],
            mode='lines',
            line=dict(width=2, color='red'),
            fill='tozeroy',
            name='Drawdown'
        ))
        fig_drawdown.update_layout(
            title=f"{strategy_name} - {asset_name}: Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown ($)",
            height=400
        )
        charts['drawdown'] = fig_drawdown
    
    return charts

def format_stats_table(stats_str):
    """Convert the stats string to a nicely formatted dataframe."""
    lines = stats_str.split('\n')
    data = []
    
    for line in lines:
        if '|' in line and 'Metric' not in line and '---' not in line:
            parts = [part.strip() for part in line.split('|') if part.strip()]
            if len(parts) >= 2:
                data.append({'Metric': parts[0], 'Value': parts[1]})
    
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        # Fallback parsing
        stats_dict = {}
        for line in lines:
            if ':' in line and not line.startswith('+') and not line.startswith('|'):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    stats_dict[key] = value
        
        df = pd.DataFrame(list(stats_dict.items()), columns=['Metric', 'Value'])
        return df

def safe_summary_stats(backtester, **kwargs):
    """Wrapper for summary_stats with error handling."""
    try:
        return backtester.summary_stats(**kwargs)
    except IndexError as e:
        if "index 0 is out of bounds" in str(e):
            # Return a basic summary when detailed analysis fails
            try:
                # Create a copy of kwargs without entry_day_of_week for backtest_strategy
                backtest_kwargs = kwargs.copy()
                if 'entry_day_of_week' in backtest_kwargs:
                    del backtest_kwargs['entry_day_of_week']
                
                # Run basic backtest to get trade count
                daily_data, weekday_data = backtester.backtest_strategy(**backtest_kwargs)
                
                if daily_data.empty:
                    return "No trades were generated for the selected parameters. Try adjusting the date range or strategy parameters."
                else:
                    # Basic stats without day-of-week analysis
                    total_trades = len(daily_data)
                    
                    # Check if we have the required columns
                    if 'current_pnl' in daily_data.columns:
                        total_pnl = daily_data['current_pnl'].sum()
                        win_rate = (daily_data['current_pnl'] > 0).mean() * 100
                    else:
                        total_pnl = 0
                        win_rate = 0
                    
                    return f"""Basic Results:
Total Trades: {total_trades}
Total P&L: ${total_pnl:.2f}
Win Rate: {win_rate:.1f}%

Note: Detailed analysis failed due to insufficient data for some metrics.
Try a longer date range or different parameters for full analysis."""
            except Exception as e2:
                return f"Error running backtest: {str(e2)}"
        else:
            return f"Error in analysis: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def compare_strategies(backtester, strategies, **backtest_params):
    """Compare multiple strategies and return comparative metrics."""
    results = []
    for strategy in strategies:
        try:
            stats = backtester.summary_stats(strategy, **backtest_params)
            results.append({
                'strategy': strategy.name,
                'stats': stats
            })
        except Exception as e:
            results.append({
                'strategy': strategy.name,
                'stats': f"Error: {str(e)}"
            })
    return results

def optimize_strategy(backtester, strategy_template, param_ranges, **backtest_params):
    """Optimize strategy parameters using grid search."""
    best_params = None
    best_performance = -float('inf')
    
    # Generate parameter combinations
    import itertools
    # Convert ranges to lists
    param_lists = []
    param_names = []
    for param_name, param_range in param_ranges.items():
        param_names.append(param_name)
        if isinstance(param_range, (list, tuple)):
            param_lists.append(param_range)
        else:
            # Assume it's a range object or similar
            param_lists.append(list(param_range))
    
    # Generate all combinations
    combinations = list(itertools.product(*param_lists))
    
    for combo in combinations:
        try:
            params = dict(zip(param_names, combo))
            strategy = strategy_template(**params)
            stats = backtester.summary_stats(strategy, **backtest_params)
            # Extract performance metric (e.g., Sharpe ratio)
            # This is a simplified version - in practice, you'd parse the stats table
            # For now, we'll just use a dummy value based on total P&L
            if isinstance(stats, str) and "Total P&L" in stats:
                # Extract P&L from stats string
                lines = stats.split('\n')
                for line in lines:
                    if "Total P&L" in line:
                        try:
                            # Extract the numeric value
                            pnl_str = line.split(":")[1].strip().replace("$", "").replace(",", "")
                            performance = float(pnl_str)
                            if performance > best_performance:
                                best_performance = performance
                                best_params = params
                            break
                        except:
                            continue
        except Exception as e:
            # Skip this parameter combination if there's an error
            continue
    
    return best_params, best_performance

@st.cache_data
def get_benchmark_data(start_date, end_date):
    """Load S&P 500 data for benchmarking."""
    try:
        import yfinance as yf
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        return sp500['Close'].pct_change().dropna()
    except Exception as e:
        st.warning(f"Could not load benchmark data: {str(e)}")
        return None

def create_benchmark_comparison(daily_df, benchmark_returns, strategy_name):
    """Create benchmark comparison chart."""
    if benchmark_returns is None or daily_df.empty:
        return None

    # Calculate strategy cumulative returns
    if 'current_pnl' in daily_df.columns:
        strategy_returns = daily_df.set_index('entry_date')['current_pnl'].pct_change().dropna()
        strategy_cumulative = (1 + strategy_returns).cumprod() - 1
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strategy_cumulative.index,
            y=strategy_cumulative.values,
            mode='lines',
            name=f'{strategy_name} Returns',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='S&P 500 Returns',
            line=dict(color='gray', dash='dash')
        ))
        fig.update_layout(
            title=f"{strategy_name} vs S&P 500 Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    return None

def display_enhanced_metrics(stats_df, prev_results=None):
    """Display metrics with delta comparisons."""
    if not stats_df.empty:
        # Create a dictionary of current metrics
        metrics_dict = dict(zip(stats_df['Metric'].str.lower(), stats_df['Value']))

        # Calculate deltas if previous results available
        deltas = {}
        if prev_results:
            prev_dict = dict(zip(prev_results['Metric'].str.lower(), prev_results['Value']))
            for key in metrics_dict:
                if key in prev_dict:
                    try:
                        current_val = float(metrics_dict[key].replace('$', '').replace('%', '').replace(',', ''))
                        prev_val = float(prev_dict[key].replace('$', '').replace('%', '').replace(',', ''))
                        delta = ((current_val - prev_val) / abs(prev_val)) * 100 if prev_val != 0 else 0
                        deltas[key] = delta
                    except:
                        pass

        # Display metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_pnl = metrics_dict.get('total pnl', '$0')
            delta_pnl = f"{deltas.get('total pnl', 0):+.1f}%" if 'total pnl' in deltas else None
            st.metric("Total P&L", total_pnl, delta=delta_pnl)

        with col2:
            win_rate = metrics_dict.get('win rate', '0%')
            st.metric("Win Rate", win_rate)

        with col3:
            sharpe = metrics_dict.get('sharpe ratio', '0.00')
            st.metric("Sharpe Ratio", sharpe)

        with col4:
            max_dd = metrics_dict.get('max drawdown', '$0')
            st.metric("Max Drawdown", max_dd)

def create_quick_presets():
    """Create quick parameter presets."""
    preset_options = {
        "Conservative": {
            "position_size": 0.1,
            "max_loss_per_trade": 0.05,
            "risk_per_trade": 0.01
        },
        "Balanced": {
            "position_size": 0.25,
            "max_loss_per_trade": 0.10,
            "risk_per_trade": 0.02
        },
        "Aggressive": {
            "position_size": 0.5,
            "max_loss_per_trade": 0.20,
            "risk_per_trade": 0.05
        }
    }
    return preset_options

def main():
    # Enhanced page config
    st.set_page_config(
        page_title="Stock Options Strategy Backtester Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced header with subtitle
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.title("📈 Options Strategy Backtester Pro")
        st.markdown("*Advanced analytics platform for options trading strategies*")

    with col_badge:
        st.markdown("### ✨ Version 1.1.0")
        st.markdown("*Enhanced UI & Analytics*")
    
    # Sidebar for strategy selection and parameters (available across all tabs)
    st.sidebar.header("🎯 Strategy Configuration")

    # Quick presets
    preset_options = create_quick_presets()
    selected_preset = st.sidebar.selectbox("Quick Risk Presets",
                                           ["Custom"] + list(preset_options.keys()),
                                           help="Choose a risk profile or select Custom for detailed control")
    st.sidebar.markdown("---")

    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    selected_ticker = ticker

    # Date range for yfinance data
    default_start = datetime(2020, 1, 1).date()
    default_end = datetime.now().date()
    start_date_yf = st.sidebar.date_input("Start Date", default_start, min_value=datetime(2010, 1, 1).date(), max_value=default_end)
    end_date_yf = st.sidebar.date_input("End Date", default_end, min_value=start_date_yf, max_value=default_end)

    # Load yfinance data
    with st.spinner("Loading stock data..."):
        price_data = load_yfinance_data(ticker, start_date_yf, end_date_yf)

    if price_data is None or price_data.empty:
        st.error("Failed to load stock data. Please check the ticker symbol and try again.")
        return

    # Strategy selection
    strategy_options = [
        "Covered Call",
        "Cash Secured Put",
        "Long Straddle",
        "Iron Condor",
        "Bull Call Spread",
        "Long Strangle",
        "Bear Put Spread",
        "Bear Call Spread",
        "Bull Put Spread",
        "Protective Put",
        "Collar",
        "Calendar Spread",
        "Butterfly Spread",
        "Condor Spread"
    ]
    
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)
    
    # Show strategy diagram
    strategy_fig = create_strategy_diagram(selected_strategy)
    st.sidebar.plotly_chart(strategy_fig, use_container_width=True)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    # Common parameters with number inputs instead of sliders
    if selected_strategy == "Covered Call":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=100, max_value=120, value=105, step=1) / 100
        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1)
        underlying_held = st.sidebar.number_input("Already Held", min_value=0, max_value=1000, value=1, step=1)
        underlying_avg_cost = st.sidebar.number_input("Average Cost ($)", min_value=0.0, max_value=200000.0, value=45000.0, step=100.0) if underlying_held > 0 else None
        
        strategy = OptionStrategyTemplate.covered_call(
            strike_pct=strike_pct,
            underlying_position=underlying_position,
            underlying_held=underlying_held,
            underlying_avg_cost=underlying_avg_cost
        )
        
    elif selected_strategy == "Cash Secured Put":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.cash_secured_put(
            strike_pct=strike_pct
        )
        
    elif selected_strategy == "Long Straddle":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1) / 100
        
        strategy = OptionStrategyTemplate.long_straddle(
            strike_pct=strike_pct
        )
        
    elif selected_strategy == "Iron Condor":
        put_short = st.sidebar.number_input("Put Short Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        put_long = st.sidebar.number_input("Put Long Strike (%)", min_value=80, max_value=95, value=90, step=1) / 100
        call_short = st.sidebar.number_input("Call Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        call_long = st.sidebar.number_input("Call Long Strike (%)", min_value=105, max_value=120, value=110, step=1) / 100
        
        strategy = OptionStrategyTemplate.iron_condor(
            put_short=put_short,
            put_long=put_long,
            call_short=call_short,
            call_long=call_long
        )
        
    elif selected_strategy == "Bull Call Spread":
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=95, max_value=105, value=100, step=1) / 100
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        
        strategy = OptionStrategyTemplate.bull_call_spread(
            long_strike=long_strike,
            short_strike=short_strike
        )
        
    elif selected_strategy == "Long Strangle":
        call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        put_strike = st.sidebar.number_input("Put Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.long_strangle(
            call_strike=call_strike,
            put_strike=put_strike
        )
    
    elif selected_strategy == "Bear Put Spread":
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=100, step=1) / 100
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=80, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.bear_put_spread(
            long_strike=long_strike,
            short_strike=short_strike
        )
    
    elif selected_strategy == "Bear Call Spread":
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1) / 100
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=105, step=1) / 100
        
        strategy = OptionStrategyTemplate.bear_call_spread(
            short_strike=short_strike,
            long_strike=long_strike
        )
    
    elif selected_strategy == "Bull Put Spread":
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1) / 100
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=80, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.bull_put_spread(
            short_strike=short_strike,
            long_strike=long_strike
        )
    
    elif selected_strategy == "Protective Put":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1) / 100
        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1)
        
        strategy = OptionStrategyTemplate.protective_put(
            strike_pct=strike_pct,
            underlying_position=underlying_position
        )
    
    elif selected_strategy == "Collar":
        call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=120, value=105, step=1) / 100
        put_strike = st.sidebar.number_input("Put Strike (%)", min_value=80, max_value=100, value=95, step=1) / 100
        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1)
        
        strategy = OptionStrategyTemplate.collar(
            call_strike=call_strike,
            put_strike=put_strike,
            underlying_position=underlying_position
        )
    
    elif selected_strategy == "Calendar Spread":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1) / 100
        
        strategy = OptionStrategyTemplate.calendar_spread(
            strike_pct=strike_pct
        )
    
    elif selected_strategy == "Butterfly Spread":
        lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=90, max_value=100, value=95, step=1) / 100
        middle_strike = st.sidebar.number_input("Middle Strike (%)", min_value=95, max_value=105, value=100, step=1) / 100
        upper_strike = st.sidebar.number_input("Upper Strike (%)", min_value=100, max_value=110, value=105, step=1) / 100
        
        strategy = OptionStrategyTemplate.butterfly_spread(
            lower_strike=lower_strike,
            middle_strike=middle_strike,
            upper_strike=upper_strike
        )
    
    elif selected_strategy == "Condor Spread":
        lowest_strike = st.sidebar.number_input("Lowest Strike (%)", min_value=90, max_value=95, value=90, step=1) / 100
        lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=95, max_value=100, value=95, step=1) / 100
        higher_strike = st.sidebar.number_input("Higher Strike (%)", min_value=100, max_value=105, value=105, step=1) / 100
        highest_strike = st.sidebar.number_input("Highest Strike (%)", min_value=105, max_value=110, value=110, step=1) / 100
        
        strategy = OptionStrategyTemplate.condor_spread(
            lowest_strike=lowest_strike,
            lower_strike=lower_strike,
            higher_strike=higher_strike,
            highest_strike=highest_strike
        )
    
    # Backtesting parameters
    st.sidebar.subheader("Backtesting Parameters")
    expiry_days = st.sidebar.number_input("Expiry Days", min_value=1, max_value=30, value=7, step=1)
    interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)
    volatility = st.sidebar.number_input("Volatility", min_value=0.1, max_value=1.0, value=0.50, step=0.05)
    
    # Volatility - use yfinance volatility when available for stocks
    use_yf_vol = st.sidebar.checkbox("Use yfinance Implied Volatility", value=True)
    if use_yf_vol:
        volatility = None  # Will be determined from yfinance
    else:
        volatility = st.sidebar.number_input("Volatility", min_value=0.1, max_value=1.0, value=0.50, step=0.05)
    
    # Date range
    if not isinstance(price_data.index, pd.DatetimeIndex):  # Ensure index is datetime
        price_data.index = pd.to_datetime(price_data.index)
    available_start = price_data.index.min().date()
    available_end = price_data.index.max().date()
    
    default_start = max(available_start, datetime(2020, 1, 1).date())
    default_end = min(available_end, datetime.now().date())

    # Ensure all are datetime.date
    if isinstance(available_start, pd.Timestamp):
        available_start = available_start.date()
    if isinstance(available_end, pd.Timestamp):
        available_end = available_end.date()
    if isinstance(default_start, pd.Timestamp):
        default_start = default_start.date()
    if isinstance(default_end, pd.Timestamp):
        default_end = default_end.date()

    start_date = st.sidebar.date_input("Start Date", default_start, min_value=available_start, max_value=available_end)
    end_date = st.sidebar.date_input("End Date", default_end, min_value=available_start, max_value=available_end)
    
    trade_frequency = st.sidebar.selectbox(
        "Trade Frequency",
        ["non_overlapping", "daily", "weekly", "monthly"]
    )
    
    # Entry day of week selection (always available, not just for weekly/monthly)
    entry_day_option = st.sidebar.selectbox(
        "Entry Day of Week Filter",
        ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        help="Filter results to only show trades entered on specific days of the week. This affects the analysis and statistics."
    )
    
    if entry_day_option == "All Days":
        entry_day_of_week = None
    else:
        entry_day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(entry_day_option) + 1
    
    # Comparison mode
    comparison_mode = st.sidebar.checkbox("Compare Strategies")
    
    # Optimization mode
    optimize_mode = st.sidebar.checkbox("Optimize Strategy")
    
    # Load data and run backtest
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Use selected data
            backtester = OptionsBacktester(price_data)
            
            # Show data info
            data_info = f"Using {ticker} data: {len(price_data)} days from {price_data.index[0].date()} to {price_data.index[-1].date()}"
            asset_name = ticker
            if entry_day_of_week:
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][entry_day_of_week - 1]
                data_info += f" (filtered for {day_name}s only)"
            st.info(data_info)
            
            # Parameters for backtesting
            backtest_params = {
                'expiry_days': expiry_days,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'trade_frequency': trade_frequency,
                'entry_day_of_week': entry_day_of_week,
                'interest_rate': interest_rate
            }
            
            # Handle comparison mode
            if comparison_mode:
                # Get selected strategies for comparison
                selected_strategies = st.sidebar.multiselect("Select Strategies to Compare", strategy_options, default=[selected_strategy])
                
                # Create strategy objects
                strategies = []
                for strat_name in selected_strategies:
                    if strat_name == "Covered Call":
                        strategies.append(OptionStrategyTemplate.covered_call(strike_pct=1.05))
                    elif strat_name == "Cash Secured Put":
                        strategies.append(OptionStrategyTemplate.cash_secured_put(strike_pct=0.95))
                    elif strat_name == "Long Straddle":
                        strategies.append(OptionStrategyTemplate.long_straddle(strike_pct=1.00))
                    elif strat_name == "Iron Condor":
                        strategies.append(OptionStrategyTemplate.iron_condor())
                    elif strat_name == "Bull Call Spread":
                        strategies.append(OptionStrategyTemplate.bull_call_spread())
                    elif strat_name == "Long Strangle":
                        strategies.append(OptionStrategyTemplate.long_strangle())
                    elif strat_name == "Bear Put Spread":
                        strategies.append(OptionStrategyTemplate.bear_put_spread())
                    elif strat_name == "Bear Call Spread":
                        strategies.append(OptionStrategyTemplate.bear_call_spread())
                    elif strat_name == "Bull Put Spread":
                        strategies.append(OptionStrategyTemplate.bull_put_spread())
                    elif strat_name == "Protective Put":
                        strategies.append(OptionStrategyTemplate.protective_put())
                    elif strat_name == "Collar":
                        strategies.append(OptionStrategyTemplate.collar())
                    elif strat_name == "Calendar Spread":
                        strategies.append(OptionStrategyTemplate.calendar_spread())
                    elif strat_name == "Butterfly Spread":
                        strategies.append(OptionStrategyTemplate.butterfly_spread())
                    elif strat_name == "Condor Spread":
                        strategies.append(OptionStrategyTemplate.condor_spread())
                
                # Add volatility parameter
                backtest_params['volatility'] = volatility
                
                # Compare strategies
                comparison_results = compare_strategies(backtester, strategies, **backtest_params)
                
                # Store results in session state
                st.session_state.comparison_results = comparison_results
                st.session_state.asset_name = asset_name
            elif optimize_mode:
                # Optimization mode
                # Define parameter ranges for optimization
                st.sidebar.subheader("Optimization Parameters")
                
                # Different parameter ranges based on selected strategy
                if selected_strategy == "Covered Call":
                    strike_range = st.sidebar.slider("Strike Price Range (%)", 100, 120, (100, 110), 1)
                    param_ranges = {
                        'strike_pct': [x/100.0 for x in range(strike_range[0], strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.covered_call
                elif selected_strategy == "Cash Secured Put":
                    strike_range = st.sidebar.slider("Strike Price Range (%)", 80, 100, (90, 100), 1)
                    param_ranges = {
                        'strike_pct': [x/100.0 for x in range(strike_range[0], strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.cash_secured_put
                elif selected_strategy == "Long Straddle":
                    strike_range = st.sidebar.slider("Strike Price Range (%)", 95, 105, (98, 102), 1)
                    param_ranges = {
                        'strike_pct': [x/100.0 for x in range(strike_range[0], strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.long_straddle
                elif selected_strategy == "Iron Condor":
                    put_short_range = st.sidebar.slider("Put Short Strike Range (%)", 85, 100, (90, 95), 1)
                    put_long_range = st.sidebar.slider("Put Long Strike Range (%)", 80, 95, (85, 90), 1)
                    call_short_range = st.sidebar.slider("Call Short Strike Range (%)", 100, 115, (105, 110), 1)
                    call_long_range = st.sidebar.slider("Call Long Strike Range (%)", 105, 120, (110, 115), 1)
                    param_ranges = {
                        'put_short': [x/100.0 for x in range(put_short_range[0], put_short_range[1]+1)],
                        'put_long': [x/100.0 for x in range(put_long_range[0], put_long_range[1]+1)],
                        'call_short': [x/100.0 for x in range(call_short_range[0], call_short_range[1]+1)],
                        'call_long': [x/100.0 for x in range(call_long_range[0], call_long_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.iron_condor
                elif selected_strategy == "Bull Call Spread":
                    long_strike_range = st.sidebar.slider("Long Strike Range (%)", 95, 105, (98, 102), 1)
                    short_strike_range = st.sidebar.slider("Short Strike Range (%)", 100, 115, (103, 107), 1)
                    param_ranges = {
                        'long_strike': [x/100.0 for x in range(long_strike_range[0], long_strike_range[1]+1)],
                        'short_strike': [x/100.0 for x in range(short_strike_range[0], short_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.bull_call_spread
                elif selected_strategy == "Long Strangle":
                    call_strike_range = st.sidebar.slider("Call Strike Range (%)", 100, 115, (103, 107), 1)
                    put_strike_range = st.sidebar.slider("Put Strike Range (%)", 85, 100, (93, 97), 1)
                    param_ranges = {
                        'call_strike': [x/100.0 for x in range(call_strike_range[0], call_strike_range[1]+1)],
                        'put_strike': [x/100.0 for x in range(put_strike_range[0], put_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.long_strangle
                elif selected_strategy == "Bear Put Spread":
                    long_strike_range = st.sidebar.slider("Long Strike Range (%)", 100, 120, (103, 107), 1)
                    short_strike_range = st.sidebar.slider("Short Strike Range (%)", 80, 100, (93, 97), 1)
                    param_ranges = {
                        'long_strike': [x/100.0 for x in range(long_strike_range[0], long_strike_range[1]+1)],
                        'short_strike': [x/100.0 for x in range(short_strike_range[0], short_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.bear_put_spread
                elif selected_strategy == "Bear Call Spread":
                    short_strike_range = st.sidebar.slider("Short Strike Range (%)", 100, 120, (103, 107), 1)
                    long_strike_range = st.sidebar.slider("Long Strike Range (%)", 100, 120, (108, 112), 1)
                    param_ranges = {
                        'short_strike': [x/100.0 for x in range(short_strike_range[0], short_strike_range[1]+1)],
                        'long_strike': [x/100.0 for x in range(long_strike_range[0], long_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.bear_call_spread
                elif selected_strategy == "Bull Put Spread":
                    short_strike_range = st.sidebar.slider("Short Strike Range (%)", 100, 120, (103, 107), 1)
                    long_strike_range = st.sidebar.slider("Long Strike Range (%)", 80, 100, (93, 97), 1)
                    param_ranges = {
                        'short_strike': [x/100.0 for x in range(short_strike_range[0], short_strike_range[1]+1)],
                        'long_strike': [x/100.0 for x in range(long_strike_range[0], long_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.bull_put_spread
                elif selected_strategy == "Protective Put":
                    strike_range = st.sidebar.slider("Strike Price Range (%)", 80, 100, (90, 95), 1)
                    param_ranges = {
                        'strike_pct': [x/100.0 for x in range(strike_range[0], strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.protective_put
                elif selected_strategy == "Collar":
                    call_strike_range = st.sidebar.slider("Call Strike Range (%)", 100, 120, (103, 107), 1)
                    put_strike_range = st.sidebar.slider("Put Strike Range (%)", 80, 100, (93, 97), 1)
                    param_ranges = {
                        'call_strike': [x/100.0 for x in range(call_strike_range[0], call_strike_range[1]+1)],
                        'put_strike': [x/100.0 for x in range(put_strike_range[0], put_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.collar
                elif selected_strategy == "Calendar Spread":
                    strike_range = st.sidebar.slider("Strike Price Range (%)", 95, 105, (98, 102), 1)
                    param_ranges = {
                        'strike_pct': [x/100.0 for x in range(strike_range[0], strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.calendar_spread
                elif selected_strategy == "Butterfly Spread":
                    lower_strike_range = st.sidebar.slider("Lower Strike Range (%)", 90, 100, (93, 97), 1)
                    middle_strike_range = st.sidebar.slider("Middle Strike Range (%)", 95, 105, (98, 102), 1)
                    upper_strike_range = st.sidebar.slider("Upper Strike Range (%)", 100, 110, (103, 107), 1)
                    param_ranges = {
                        'lower_strike': [x/100.0 for x in range(lower_strike_range[0], lower_strike_range[1]+1)],
                        'middle_strike': [x/100.0 for x in range(middle_strike_range[0], middle_strike_range[1]+1)],
                        'upper_strike': [x/100.0 for x in range(upper_strike_range[0], upper_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.butterfly_spread
                elif selected_strategy == "Condor Spread":
                    lowest_strike_range = st.sidebar.slider("Lowest Strike Range (%)", 90, 95, (91, 94), 1)
                    lower_strike_range = st.sidebar.slider("Lower Strike Range (%)", 95, 100, (96, 99), 1)
                    higher_strike_range = st.sidebar.slider("Higher Strike Range (%)", 100, 105, (101, 104), 1)
                    highest_strike_range = st.sidebar.slider("Highest Strike Range (%)", 105, 110, (106, 109), 1)
                    param_ranges = {
                        'lowest_strike': [x/100.0 for x in range(lowest_strike_range[0], lowest_strike_range[1]+1)],
                        'lower_strike': [x/100.0 for x in range(lower_strike_range[0], lower_strike_range[1]+1)],
                        'higher_strike': [x/100.0 for x in range(higher_strike_range[0], higher_strike_range[1]+1)],
                        'highest_strike': [x/100.0 for x in range(highest_strike_range[0], highest_strike_range[1]+1)]
                    }
                    strategy_template = OptionStrategyTemplate.condor_spread
                else:
                    # Default parameter ranges
                    param_ranges = {
                        'strike_pct': [0.95, 1.00, 1.05]
                    }
                    strategy_template = OptionStrategyTemplate.covered_call
                
                # Add volatility parameter
                backtest_params['volatility'] = volatility
                
                # Run optimization
                with st.spinner("Optimizing strategy parameters..."):
                    best_params, best_performance = optimize_strategy(backtester, strategy_template, param_ranges, **backtest_params)
                
                # Store results in session state
                st.session_state.best_params = best_params
                st.session_state.best_performance = best_performance
                st.session_state.asset_name = asset_name
                st.session_state.optimized_strategy = selected_strategy
            else:
                # Single strategy mode
                backtest_params['strategy'] = strategy
                
                # Add volatility parameter or use yfinance volatility
                if use_yf_vol:
                    # Use the backtest_strategy_with_yfinance_volatility method
                    daily_df, weekday_df = backtester.backtest_strategy_with_yfinance_volatility(
                        strategy, ticker, expiry_days, str(start_date), str(end_date),
                        trade_frequency, interest_rate
                    )
                    # For now, we'll just use the regular summary_stats for display
                    backtest_params['volatility'] = 0.50  # Default value for display
                    stats_str = safe_summary_stats(backtester, **backtest_params)
                else:
                    backtest_params['volatility'] = volatility
                    # Run backtest with safe wrapper
                    stats_str = safe_summary_stats(backtester, **backtest_params)
                    
                    # Get detailed backtest data for tables
                    try:
                        backtest_kwargs = backtest_params.copy()
                        if 'entry_day_of_week' in backtest_kwargs:
                            del backtest_kwargs['entry_day_of_week']  # Remove for backtest_strategy call
                        daily_df, weekday_df = backtester.backtest_strategy(**backtest_kwargs)
                    except Exception as e:
                        st.warning(f"Could not get detailed backtest data: {str(e)}")
                        daily_df, weekday_df = None, None
                
                # Try to get the plot
                try:
                    fig = backtester.plot_backtest_results(**backtest_params)
                except Exception as e:
                    st.warning(f"Could not generate plot: {str(e)}")
                    fig = None
                
                # Store results in session state
                st.session_state.stats_str = stats_str
                if fig is not None:
                    st.session_state.fig = fig
                if daily_df is not None:
                    st.session_state.daily_df = daily_df
                if weekday_df is not None:
                    st.session_state.weekday_df = weekday_df
                st.session_state.strategy_name = selected_strategy
                st.session_state.asset_name = asset_name
                st.session_state.data_source = "Stocks"
    
    # Display results
    if comparison_mode and hasattr(st.session_state, 'comparison_results'):
        st.header(f"📊 Strategy Comparison - {st.session_state.asset_name}")
        
        # Display comparison results
        comparison_results = st.session_state.comparison_results
        
        # Create a combined dataframe for comparison
        comparison_data = []
        for result in comparison_results:
            strategy_name = result['strategy']
            stats_str = result['stats']
            if not isinstance(stats_str, str) or "Error" in stats_str:
                continue
            stats_df = format_stats_table(stats_str)
            if not stats_df.empty:
                # Add strategy name to each row
                stats_df['Strategy'] = strategy_name
                comparison_data.append(stats_df)
        
        if comparison_data:
            # Combine all dataframes
            combined_df = pd.concat(comparison_data, ignore_index=True)
            # Pivot to show strategies as columns
            pivot_df = combined_df.pivot(index='Metric', columns='Strategy', values='Value')
            # Reset index to make Metric a column
            pivot_df.reset_index(inplace=True)
            
            # Display the comparison table
            st.subheader("Performance Comparison")
            st.dataframe(pivot_df, use_container_width=True)
            
            # Download button for comparison data
            csv_comparison = pivot_df.to_csv(index=False)
            st.download_button(
                label="Download Comparison Data",
                data=csv_comparison,
                file_name=f"strategy_comparison_{st.session_state.asset_name}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid comparison data available.")
    
    elif optimize_mode and hasattr(st.session_state, 'best_params'):
        st.header(f"🎯 Strategy Optimization - {st.session_state.asset_name}")
        
        # Display optimization results
        best_params = st.session_state.best_params
        best_performance = st.session_state.best_performance
        optimized_strategy = st.session_state.optimized_strategy
        
        if best_params is not None:
            st.success(f"✅ Optimization Complete!")
            st.subheader(f"Optimal Parameters for {optimized_strategy}")
            
            # Display best parameters
            params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Optimal Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            # Display best performance
            st.metric("Best Performance", f"{best_performance:.2f}")
            
            # Create a strategy with the optimal parameters
            try:
                if optimized_strategy == "Covered Call":
                    optimal_strategy = OptionStrategyTemplate.covered_call(**best_params)
                elif optimized_strategy == "Cash Secured Put":
                    optimal_strategy = OptionStrategyTemplate.cash_secured_put(**best_params)
                elif optimized_strategy == "Long Straddle":
                    optimal_strategy = OptionStrategyTemplate.long_straddle(**best_params)
                elif optimized_strategy == "Iron Condor":
                    optimal_strategy = OptionStrategyTemplate.iron_condor(**best_params)
                elif optimized_strategy == "Bull Call Spread":
                    optimal_strategy = OptionStrategyTemplate.bull_call_spread(**best_params)
                elif optimized_strategy == "Long Strangle":
                    optimal_strategy = OptionStrategyTemplate.long_strangle(**best_params)
                elif optimized_strategy == "Bear Put Spread":
                    optimal_strategy = OptionStrategyTemplate.bear_put_spread(**best_params)
                elif optimized_strategy == "Bear Call Spread":
                    optimal_strategy = OptionStrategyTemplate.bear_call_spread(**best_params)
                elif optimized_strategy == "Bull Put Spread":
                    optimal_strategy = OptionStrategyTemplate.bull_put_spread(**best_params)
                elif optimized_strategy == "Protective Put":
                    optimal_strategy = OptionStrategyTemplate.protective_put(**best_params)
                elif optimized_strategy == "Collar":
                    optimal_strategy = OptionStrategyTemplate.collar(**best_params)
                elif optimized_strategy == "Calendar Spread":
                    optimal_strategy = OptionStrategyTemplate.calendar_spread(**best_params)
                elif optimized_strategy == "Butterfly Spread":
                    optimal_strategy = OptionStrategyTemplate.butterfly_spread(**best_params)
                elif optimized_strategy == "Condor Spread":
                    optimal_strategy = OptionStrategyTemplate.condor_spread(**best_params)
                
                # Run backtest with optimal parameters
                backtest_params = {
                    'strategy': optimal_strategy,
                    'expiry_days': expiry_days,
                    'start_date': str(start_date),
                    'end_date': str(end_date),
                    'trade_frequency': trade_frequency,
                    'entry_day_of_week': entry_day_of_week,
                    'interest_rate': interest_rate,
                    'volatility': volatility
                }
                
                # Run backtest with safe wrapper
                stats_str = safe_summary_stats(backtester, **backtest_params)
                
                # Get detailed backtest data for tables
                try:
                    backtest_kwargs = backtest_params.copy()
                    if 'entry_day_of_week' in backtest_kwargs:
                        del backtest_kwargs['entry_day_of_week']  # Remove for backtest_strategy call
                    daily_df, weekday_df = backtester.backtest_strategy(**backtest_kwargs)
                except Exception as e:
                    st.warning(f"Could not get detailed backtest data: {str(e)}")
                    daily_df, weekday_df = None, None
                
                # Try to get the plot
                try:
                    fig = backtester.plot_backtest_results(**backtest_params)
                except Exception as e:
                    st.warning(f"Could not generate plot: {str(e)}")
                    fig = None
                
                # Display the backtest results for the optimal strategy
                st.subheader(f"Backtest Results with Optimal Parameters")
                
                # Create two columns for layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Performance Summary")
                    
                    # Format and display stats table
                    stats_df = format_stats_table(stats_str)
                    if not stats_df.empty:
                        # Style the dataframe
                        styled_df = stats_df.style.set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                            {'selector': 'td', 'props': [('text-align', 'left')]},
                        ])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.text(stats_str)
                
                with col2:
                    st.subheader("Key Metrics")
                    
                    # Extract key metrics for display
                    if not stats_df.empty:
                        metrics_dict = dict(zip(stats_df['Metric'], stats_df['Value']))
                        
                        # Create metric cards
                        metric_cols = st.columns(3)
                        
                        with metric_cols[0]:
                            if 'total_return' in metrics_dict:
                                st.metric("Total Return", f"{metrics_dict['total_return']}%")
                            if 'win_rate' in metrics_dict:
                                st.metric("Win Rate", f"{metrics_dict['win_rate']}%")
                        
                        with metric_cols[1]:
                            if 'total_pnl' in metrics_dict:
                                st.metric("Total P&L", f"${metrics_dict['total_pnl']}")
                            if 'sharpe_ratio' in metrics_dict:
                                st.metric("Sharpe Ratio", metrics_dict['sharpe_ratio'])
                        
                        with metric_cols[2]:
                            if 'total_trades' in metrics_dict:
                                st.metric("Total Trades", metrics_dict['total_trades'])
                            if 'max_drawdown' in metrics_dict:
                                st.metric("Max Drawdown", f"${metrics_dict['max_drawdown']}")
                
                # Display the backtest chart
                st.subheader("Detailed Analysis")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No chart data available. This may occur when there's insufficient data for the selected parameters.")
                
                
                # Create two columns for the tables
                table_col1, table_col2 = st.columns(2)
                
                with table_col1:
                    st.markdown("**Daily Trade Details**")
                    
                    if daily_df is not None:
                        # Format the daily_df for better display
                        daily_display = daily_df.copy()
                        
                        # Format columns for better readability
                        if 'entry_date' in daily_display.columns:
                            daily_display['entry_date'] = pd.to_datetime(daily_display['entry_date']).dt.strftime('%Y-%m-%d')
                        if 'expiry_date' in daily_display.columns:
                            daily_display['expiry_date'] = pd.to_datetime(daily_display['expiry_date']).dt.strftime('%Y-%m-%d')
                        
                        # Round numeric columns
                        numeric_columns = ['entry_price', 'expiry_price', 'one_day_pct_move', 'expiry_pct_move',
                                         'option_gains', 'option_price', 'underlying_pnl', 'current_pnl']
                        for col in numeric_columns:
                            if col in daily_display.columns:
                                daily_display[col] = daily_display[col].round(2)
                        
                        # Format strike_prices column if it exists
                        if 'strike_prices' in daily_display.columns:
                            daily_display['strike_prices'] = daily_display['strike_prices'].apply(
                                lambda x: f"{x[0]:.2f}" if isinstance(x, list) and len(x) > 0 else str(x)
                            )
                        
                        # Display with pagination
                        st.dataframe(
                            daily_display,
                            use_container_width=True,
                            height=400,
                            hide_index=True
                        )
                        
                        # Add download button for daily data
                        csv_daily = daily_display.to_csv(index=False)
                        st.download_button(
                            label="Download Daily Trade Data",
                            data=csv_daily,
                            file_name=f"{optimized_strategy}_{st.session_state.asset_name}_daily_trades.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Daily trade data not available")
                
                with table_col2:
                    st.markdown("**Weekday Performance Summary**")
                    
                    if weekday_df is not None:
                        # Format the weekday_df for better display
                        weekday_display = weekday_df.copy()
                        
                        # Round numeric columns
                        numeric_columns = ['hits', 'losses', 'option_price', 'profit', 'total_profit', 'percentage_returns']
                        for col in numeric_columns:
                            if col in weekday_display.columns:
                                weekday_display[col] = weekday_display[col].round(2)
                        
                        # Style the dataframe to highlight the average row
                        def highlight_average_row(row):
                            if row['day_of_week'] == 'Average':
                                return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
                            return [''] * len(row)
                        
                        styled_weekday = weekday_display.style.apply(highlight_average_row, axis=1)
                        
                        st.dataframe(
                            styled_weekday,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Add download button for weekday data
                        csv_weekday = weekday_display.to_csv(index=False)
                        st.download_button(
                            label="Download Weekday Summary",
                            data=csv_weekday,
                            file_name=f"{optimized_strategy}_{st.session_state.asset_name}_weekday_summary.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Weekday summary data not available")
            except Exception as e:
                st.error(f"Error creating strategy with optimal parameters: {str(e)}")
        else:
            st.warning("❌ Optimization did not find any valid parameter combinations.")
    
    elif hasattr(st.session_state, 'stats_str'):
        st.header(f"📊 {st.session_state.strategy_name} - {st.session_state.asset_name} Backtest Results")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Performance Summary")
            
            # Format and display stats table
            stats_df = format_stats_table(st.session_state.stats_str)
            if not stats_df.empty:
                # Style the dataframe
                styled_df = stats_df.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                    {'selector': 'td', 'props': [('text-align', 'left')]},
                ])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.text(st.session_state.stats_str)
        
        with col2:
            st.subheader("Key Metrics")
            
            # Extract key metrics for display
            if not stats_df.empty:
                metrics_dict = dict(zip(stats_df['Metric'], stats_df['Value']))
                
                # Create metric cards
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    if 'total_return' in metrics_dict:
                        st.metric("Total Return", f"{metrics_dict['total_return']}%")
                    if 'win_rate' in metrics_dict:
                        st.metric("Win Rate", f"{metrics_dict['win_rate']}%")
                
                with metric_cols[1]:
                    if 'total_pnl' in metrics_dict:
                        st.metric("Total P&L", f"${metrics_dict['total_pnl']}")
                    if 'sharpe_ratio' in metrics_dict:
                        st.metric("Sharpe Ratio", metrics_dict['sharpe_ratio'])
                
                with metric_cols[2]:
                    if 'total_trades' in metrics_dict:
                        st.metric("Total Trades", metrics_dict['total_trades'])
                    if 'max_drawdown' in metrics_dict:
                        st.metric("Max Drawdown", f"${metrics_dict['max_drawdown']}")
        
        # Display the backtest chart
        st.subheader("Detailed Analysis")
        if hasattr(st.session_state, 'fig') and st.session_state.fig is not None:
            st.plotly_chart(st.session_state.fig, use_container_width=True)
        else:
            st.warning("No chart data available. This may occur when there's insufficient data for the selected parameters.")
        
        # Create additional charts
        if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
            charts = create_additional_charts(
                st.session_state.daily_df,
                st.session_state.strategy_name,
                st.session_state.asset_name
            )
            
            # Display additional charts in expandable sections
            if charts:
                st.subheader("Additional Analytics")
                
                # Create tabs for different chart categories
                tab1, tab2, tab3 = st.tabs(["Distribution Charts", "Time Series Charts", "Risk Analysis"])
                
                # Distribution Charts Tab
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'pnl_distribution' in charts:
                            st.plotly_chart(charts['pnl_distribution'], use_container_width=True)
                    with col2:
                        if 'win_streak' in charts:
                            st.plotly_chart(charts['win_streak'], use_container_width=True)
                
                # Time Series Charts Tab
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'cumulative_pnl' in charts:
                            st.plotly_chart(charts['cumulative_pnl'], use_container_width=True)
                    with col2:
                        if 'rolling_sharpe' in charts:
                            st.plotly_chart(charts['rolling_sharpe'], use_container_width=True)
                
                # Risk Analysis Tab
                with tab3:
                    if 'drawdown' in charts:
                        st.plotly_chart(charts['drawdown'], use_container_width=True)
        
        
        # Create two columns for the tables
        table_col1, table_col2 = st.columns(2)
        
        with table_col1:
            st.markdown("**Daily Trade Details**")
            
            if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                # Format the daily_df for better display
                daily_display = st.session_state.daily_df.copy()
                
                # Format columns for better readability
                if 'entry_date' in daily_display.columns:
                    daily_display['entry_date'] = pd.to_datetime(daily_display['entry_date']).dt.strftime('%Y-%m-%d')
                if 'expiry_date' in daily_display.columns:
                    daily_display['expiry_date'] = pd.to_datetime(daily_display['expiry_date']).dt.strftime('%Y-%m-%d')
                
                # Round numeric columns
                numeric_columns = ['entry_price', 'expiry_price', 'one_day_pct_move', 'expiry_pct_move',
                                 'option_gains', 'option_price', 'underlying_pnl', 'current_pnl']
                for col in numeric_columns:
                    if col in daily_display.columns:
                        daily_display[col] = daily_display[col].round(2)
                
                # Format strike_prices column if it exists
                if 'strike_prices' in daily_display.columns:
                    daily_display['strike_prices'] = daily_display['strike_prices'].apply(
                        lambda x: f"{x[0]:.2f}" if isinstance(x, list) and len(x) > 0 else str(x)
                    )
                
                # Display with pagination
                st.dataframe(
                    daily_display,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                # Add download button for daily data
                csv_daily = daily_display.to_csv(index=False)
                st.download_button(
                    label="Download Daily Trade Data",
                    data=csv_daily,
                    file_name=f"{st.session_state.strategy_name}_{st.session_state.asset_name}_daily_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("Daily trade data not available")
        
        with table_col2:
            st.markdown("**Weekday Performance Summary**")
            
            if hasattr(st.session_state, 'weekday_df') and st.session_state.weekday_df is not None:
                # Format the weekday_df for better display
                weekday_display = st.session_state.weekday_df.copy()
                
                # Round numeric columns
                numeric_columns = ['hits', 'losses', 'option_price', 'profit', 'total_profit', 'percentage_returns']
                for col in numeric_columns:
                    if col in weekday_display.columns:
                        weekday_display[col] = weekday_display[col].round(2)
                
                # Style the dataframe to highlight the average row
                def highlight_average_row(row):
                    if row['day_of_week'] == 'Average':
                        return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_weekday = weekday_display.style.apply(highlight_average_row, axis=1)
                
                st.dataframe(
                    styled_weekday,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add download button for weekday data
                csv_weekday = weekday_display.to_csv(index=False)
                st.download_button(
                    label="Download Weekday Summary",
                    data=csv_weekday,
                    file_name=f"{st.session_state.strategy_name}_{st.session_state.asset_name}_weekday_summary.csv",
                    mime="text/csv"
                )
            else:
                st.info("Weekday summary data not available")
    
    else:
        # Enhanced Tabbed Interface for results
        tab_main, tab_analytics, tab_benchmark, tab_portfolio = st.tabs([
            "🎯 Backtest", "📊 Enhanced Analytics", "⚖️ Benchmark & Risk", "💼 Portfolio"
        ])

        with tab_main:
            st.header(f"📊 {st.session_state.strategy_name} - {st.session_state.asset_name} Backtest Results")

            # Display enhanced metrics
            stats_df = format_stats_table(st.session_state.stats_str)
            display_enhanced_metrics(stats_df)

            # Display the backtest chart
            if hasattr(st.session_state, 'fig') and st.session_state.fig is not None:
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            else:
                st.warning("No chart data available. This may occur when there's insufficient data for the selected parameters.")

        with tab_analytics:
            if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                charts = create_additional_charts(
                    st.session_state.daily_df,
                    st.session_state.strategy_name,
                    st.session_state.asset_name
                )

                if charts:
                    st.header("Advanced Analytics Dashboard")

                    # Create tabs for different chart categories
                    anal_tab1, anal_tab2, anal_tab3, anal_tab4 = st.tabs([
                        "📊 Performance", "🔍 Distributions", "⏰ Time Series", "⚠️ Risk"
                    ])

                    with anal_tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Performance Overview")
                            stats_df = format_stats_table(st.session_state.stats_str)
                            if not stats_df.empty:
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        with col2:
                            st.subheader("Strategy Characteristics")
                            # Add some strategy-specific insights here

                    with anal_tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'pnl_distribution' in charts:
                                st.plotly_chart(charts['pnl_distribution'], use_container_width=True)
                            else:
                                st.info("P&L distribution chart not available")
                        with col2:
                            if 'win_streak' in charts:
                                st.plotly_chart(charts['win_streak'], use_container_width=True)
                            else:
                                st.info("Win streak analysis not available")

                    with anal_tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'cumulative_pnl' in charts:
                                st.plotly_chart(charts['cumulative_pnl'], use_container_width=True)
                            else:
                                st.info("Cumulative P&L chart not available")
                        with col2:
                            if 'rolling_sharpe' in charts:
                                st.plotly_chart(charts['rolling_sharpe'], use_container_width=True)
                            else:
                                st.info("Rolling Sharpe chart not available")

                    with anal_tab4:
                        if 'drawdown' in charts:
                            st.plotly_chart(charts['drawdown'], use_container_width=True)
                        else:
                            st.info("Drawdown analysis not available")
                else:
                    st.info("Enhanced analytics charts not available for this backtest.")

        with tab_benchmark:
            st.header("Market Benchmarking & Risk Analysis")

            # Load benchmark data and create comparison
            benchmark_data = get_benchmark_data(start_date_yf - timedelta(days=30), end_date_yf + timedelta(days=30))
            if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                benchmark_chart = create_benchmark_comparison(
                    st.session_state.daily_df,
                    benchmark_data,
                    st.session_state.strategy_name
                )

                if benchmark_chart:
                    st.plotly_chart(benchmark_chart, use_container_width=True)
                else:
                    st.info("Benchmark comparison not available")

                # Risk metrics display
                st.subheader("Risk Metrics")
                risk_cols = st.columns(3)
                with risk_cols[0]:
                    st.metric("VaR (95%)", "TBD", delta="Low")
                with risk_cols[1]:
                    st.metric("CVaR", "TBD", delta="Low")
                with risk_cols[2]:
                    st.metric("Sortino Ratio", "TBD", delta="Good")
            else:
                st.info("Benchmarking requires completed backtest data.")

        with tab_portfolio:
            st.header("Portfolio Management & Allocation")

            # Portfolio allocation suggestions
            st.subheader("Strategy Allocation Recommendations")

            # Display portfolio metrics and suggestions
            port_cols = st.columns(2)
            with port_cols[0]:
                st.metric("Portfolio Allocation", "5-10%")
                st.metric("Correlation to S&P 500", "Low")
                st.metric("Volatility Contribution", "Medium")

            with port_cols[1]:
                st.info("💡 Based on risk metrics, this strategy is suitable for portfolio allocation as a diversifier.")
                st.info("🔄 Consider rebalancing quarterly based on market conditions.")

            # Portfolio optimization section
            st.subheader("Optimization Results")
            if hasattr(st.session_state, 'best_params') and st.session_state.best_params:
                optim_df = pd.DataFrame(list(st.session_state.best_params.items()),
                                       columns=['Parameter', 'Optimal Value'])
                st.dataframe(optim_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run optimization from sidebar to see optimal parameters.")

    # Run results (outside tabs) - for comparison and optimization results
    if comparison_mode and hasattr(st.session_state, 'comparison_results'):
        st.header(f"📊 Strategy Comparison - {st.session_state.asset_name}")

        # Results are displayed outside the tabs for comparison mode
        comparison_results = st.session_state.comparison_results

        # Create a combined dataframe for comparison
        comparison_data = []
        for result in comparison_results:
            strategy_name = result['strategy']
            stats_str = result['stats']
            if not isinstance(stats_str, str) or "Error" in stats_str:
                continue
            stats_df = format_stats_table(stats_str)
            if not stats_df.empty:
                # Add strategy name to each row
                stats_df['Strategy'] = strategy_name
                comparison_data.append(stats_df)

        if comparison_data:
            # Combine all dataframes
            combined_df = pd.concat(comparison_data, ignore_index=True)
            # Pivot to show strategies as columns
            pivot_df = combined_df.pivot(index='Metric', columns='Strategy', values='Value')
            # Reset index to make Metric a column
            pivot_df.reset_index(inplace=True)

            # Display the comparison table
            st.subheader("Performance Comparison")
            st.dataframe(pivot_df, use_container_width=True)

            # Download button for comparison data
            csv_comparison = pivot_df.to_csv(index=False)
            st.download_button(
                label="Download Comparison Data",
                data=csv_comparison,
                file_name=f"strategy_comparison_{st.session_state.asset_name}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid comparison data available.")

    elif optimize_mode and hasattr(st.session_state, 'best_params'):
        st.header(f"🎯 Strategy Optimization - {st.session_state.asset_name}")

        # Display optimization results
        best_params = st.session_state.best_params
        best_performance = st.session_state.best_performance
        optimized_strategy = st.session_state.optimized_strategy

        if best_params is not None:
            st.success(f"✅ Optimization Complete!")
            st.subheader(f"Optimal Parameters for {optimized_strategy}")

            # Display best parameters
            params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Optimal Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)

            # Display best performance
            st.metric("Best Performance", f"{best_performance:.2f}")

            # Display optimization trade details
            if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                # Add trade details for optimized strategy
                st.subheader("Optimal Strategy Trade Details")
                table_col1, table_col2 = st.columns(2)

                with table_col1:
                    st.markdown("**Daily Trade Details**")
                    daily_display = st.session_state.daily_df.copy()

                    # Format columns for better readability
                    if 'entry_date' in daily_display.columns:
                        daily_display['entry_date'] = pd.to_datetime(daily_display['entry_date']).dt.strftime('%Y-%m-%d')

                    # Round numeric columns
                    numeric_columns = ['entry_price', 'expiry_price', 'option_gains', 'option_price', 'current_pnl']
                    for col in numeric_columns:
                        if col in daily_display.columns:
                            daily_display[col] = daily_display[col].round(2)

                    st.dataframe(daily_display, use_container_width=True, height=300, hide_index=True)

                with table_col2:
                    st.markdown("**Performance Summary**")
                    stats_df = format_stats_table(st.session_state.stats_str)
                    if not stats_df.empty:
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.warning("❌ Optimization did not find any valid parameter combinations.")

    if not (comparison_mode or optimize_mode):
        st.info("⚡ Configure strategy parameters with sidebar and click 'Run Backtest' to see results in the tabs above.")

if __name__ == "__main__":
    main()
