"""
Main application file to run the Stock Options Strategy Backtester Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from backtester import *

# Function to load yfinance data
def load_yfinance_data(ticker, start_date, end_date):
    """Load data from yfinance."""
    import yfinance as yf
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            return None
        # Rename columns to match expected format
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to create strategy diagram (placeholder)
def create_strategy_diagram(strategy_name):
    """Create a placeholder strategy diagram."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=f"Strategy: {strategy_name}",
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200
    )
    return fig

# Quick presets function
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

# Safe wrapper for summary stats
def safe_summary_stats(backtester, **kwargs):
    """Safe wrapper for summary_stats method."""
    try:
        return backtester.summary_stats(**kwargs)
    except Exception as e:
        return f"Error in backtest: {str(e)}"

# Function to create additional charts
def create_additional_charts(daily_df, strategy_name, asset_name):
    """Create additional analytics charts."""
    charts = {}

    # P&L Distribution
    if 'current_pnl' in daily_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=daily_df['current_pnl'],
            nbinsx=30,
            marker_color='blue',
            opacity=0.7
        ))
        fig.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency"
        )
        charts['pnl_distribution'] = fig

    # Win Streak Analysis
    if 'option_hit' in daily_df.columns:
        # Calculate win streaks
        wins = []
        current_streak = 0
        for hit in daily_df['option_hit']:
            if hit == 0:  # Win
                current_streak += 1
            else:
                if current_streak > 0:
                    wins.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            wins.append(current_streak)

        if wins:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=wins,
                marker_color='green',
                opacity=0.7
            ))
            fig.update_layout(
                title="Win Streak Distribution",
                xaxis_title="Win Streak Length",
                yaxis_title="Frequency"
            )
            charts['win_streak'] = fig

    # Cumulative P&L
    if 'current_pnl' in daily_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_df['entry_date'],
            y=daily_df['current_pnl'].cumsum(),
            mode='lines',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)"
        )
        charts['cumulative_pnl'] = fig

    return charts

# Function to format stats table
def format_stats_table(stats_str):
    """Format stats string into a DataFrame."""
    if isinstance(stats_str, str) and "\n" in stats_str:
        lines = stats_str.strip().split('\n')
        data = []
        for line in lines[1:-1]:  # Skip header and bottom border
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    key = parts[1].strip()
                    value = parts[2].strip()
                    data.append([key, value])
        return pd.DataFrame(data, columns=['Metric', 'Value'])
    return pd.DataFrame()

# Function to get benchmark data (placeholder)
def get_benchmark_data(start_date, end_date):
    """Get benchmark data (placeholder implementation)."""
    import yfinance as yf
    try:
        spy = yf.download('^GSPC', start=start_date, end=end_date)
        if not spy.empty:
            return spy['Close']
        return pd.Series()
    except:
        return pd.Series()

# Function to create benchmark comparison chart
def create_benchmark_comparison(daily_df, benchmark_data, strategy_name):
    """Create benchmark comparison chart (placeholder)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df['entry_date'],
        y=daily_df['cumulative_pnl'],
        mode='lines',
        name=strategy_name,
        line=dict(color='blue')
    ))
    return fig

# Main application function
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

    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    selected_ticker = ticker

    # Date range for data loading and backtesting
    from datetime import datetime, timedelta
    default_start = datetime(2020, 1, 1).date()
    default_end = datetime.now().date()
    start_date = st.sidebar.date_input("Start Date", default_start, min_value=datetime(2010, 1, 1).date(), max_value=default_end)
    end_date = st.sidebar.date_input("End Date", default_end, min_value=start_date, max_value=default_end)

    # Load yfinance data
    with st.spinner("Loading stock data..."):
        price_data = load_yfinance_data(ticker, start_date, end_date)

    if price_data is None or price_data.empty:
        st.error("Failed to load stock data. Please check the ticker symbol and try again.")
        return

    # Show data preview section
    with st.expander("📈 Market Data Preview", expanded=False):
        st.subheader(f"📊 {ticker} - Market Data Summary")

        # Display key data statistics
        col_data1, col_data2, col_data3, col_data4 = st.columns(4)

        with col_data1:
            st.metric("📅 Data Points", f"{len(price_data)}")
        with col_data2:
            start_date_formatted = price_data.index.min().strftime('%Y-%m-%d')
            st.metric("📅 Start Date", start_date_formatted)
        with col_data3:
            end_date_formatted = price_data.index.max().strftime('%Y-%m-%d')
            st.metric("📅 End Date", end_date_formatted)
        with col_data4:
            trading_days = len(price_data)
            years = trading_days / 252  # Approximate trading days per year
            st.metric("⏱️ Years of Data", f"{years:.1f}")

        # Price statistics
        col_price1, col_price2, col_price3, col_price4 = st.columns(4)

        with col_price1:
            initial_price = price_data['close'].iloc[0]
            st.metric("💰 Starting Price", f"${initial_price:.2f}")
        with col_price2:
            latest_price = price_data['close'].iloc[-1]
            st.metric("💰 Latest Price", f"${latest_price:.2f}")
        with col_price3:
            price_change_pct = ((latest_price - initial_price) / initial_price) * 100
            color = "normal" if abs(price_change_pct) < 1 else "inverse"
            st.metric("📈 Total Price Change",
                     f"{price_change_pct:+.1f}%",
                     delta=f"+{price_change_pct:+.1f}%" if abs(price_change_pct) >= 1 else f"{price_change_pct:+.1f}%",
                     delta_color=color)
        with col_price4:
            price_change_abs = latest_price - initial_price
            st.metric("💵 Total Price Change", f"${price_change_abs:+.2f}")

        # Volume statistics
        col_vol1, col_vol2, col_vol3 = st.columns(3)

        with col_vol1:
            avg_volume = price_data['volume'].mean()
            st.metric("📊 Avg Daily Volume", f"{avg_volume:,.0f}")
        with col_vol2:
            max_volume = price_data['volume'].max()
            st.metric("📈 Peak Volume", f"{max_volume:,.0f}")
        with col_vol3:
            total_volume = price_data['volume'].sum()
            st.metric("📊 Total Volume", f"{total_volume:,.0f}")

        # Price volatility statistics
        col_vola1, col_vola2, col_vola3 = st.columns(3)

        with col_vola1:
            daily_returns = price_data['close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
            st.metric("📈 Annual Volatility", f"{volatility:.1f}%")
        with col_vola2:
            high = price_data['high'].max()
            low = price_data['low'].min()
            range_pct = ((high - low) / price_data['close'].iloc[0]) * 100
            st.metric("📊 Price Range", f"{range_pct:.1f}%")
        with col_vola3:
            avg_price = price_data['close'].mean()
            st.metric("📝 Average Price", f"${avg_price:.2f}")

        # Simple price chart preview
        st.subheader(f"📈 {ticker} Price Chart")
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=1.5)
        ))

        fig_preview.update_layout(
            title=f"{ticker} Price Movement",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig_preview, use_container_width=True)

        # Raw data section with checkbox to show/hide
        show_raw_data = st.checkbox("📋 Show Raw Data Table", value=False, key="raw_data_toggle")

        if show_raw_data:
            st.subheader("📋 Raw Market Data")

            # Format the data for display
            display_data = price_data.copy()
            display_data.columns = [col.capitalize() for col in display_data.columns]
            display_data.index = display_data.index.strftime('%Y-%m-%d')

            # Convert numeric columns to 2 decimal places
            numeric_cols = ['Open', 'High', 'Low', 'Close']
            for col in numeric_cols:
                if col in display_data.columns:
                    display_data[col] = display_data[col].round(2)

            st.dataframe(display_data, use_container_width=True, height=300)

            # Download button for raw data
            csv_data = display_data.to_csv()
            st.download_button(
                label="💾 Download Raw Data",
                data=csv_data,
                file_name=f"{ticker}_market_data_{start_date_formatted}_to_{end_date_formatted}.csv",
                mime="text/csv"
            )

        st.info(f"✅ **Data Status:** Successfully loaded {len(price_data)} trading days for {ticker} from {start_date_formatted} to {end_date_formatted}")

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

    # Strategy parameter configuration
    strategy_configs = {}

    # Single strategy mode
    if not comparison_mode:
        st.sidebar.subheader("Strategy Parameters")
        strategy_configs[selected_strategy] = None

        # Set parameters for the selected strategy
        if selected_strategy == "Covered Call":
            strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=100, max_value=120, value=105, step=1, key="single_strike_cc")
            underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1, key="single_under_pos_cc")
            underlying_held = st.sidebar.number_input("Already Held", min_value=0, max_value=1000, value=1, step=1, key="single_under_held_cc")
            underlying_avg_cost = st.sidebar.number_input("Average Cost ($)", min_value=0.0, max_value=200000.0, value=45000.0, step=100.0, key="single_avg_cost_cc") if underlying_held > 0 else None

            strategy_configs[selected_strategy] = {
                'strike_pct': strike_pct / 100,
                'underlying_position': underlying_position,
                'underlying_held': underlying_held,
                'underlying_avg_cost': underlying_avg_cost
            }

        elif selected_strategy == "Cash Secured Put":
            strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1, key="single_strike_csp")
            strategy_configs[selected_strategy] = {'strike_pct': strike_pct / 100}

        elif selected_strategy == "Long Straddle":
            strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1, key="single_strike_ls")
            strategy_configs[selected_strategy] = {'strike_pct': strike_pct / 100}

        elif selected_strategy == "Iron Condor":
            put_short = st.sidebar.number_input("Put Short Strike (%)", min_value=85, max_value=100, value=95, step=1, key="single_put_short_ic")
            put_long = st.sidebar.number_input("Put Long Strike (%)", min_value=80, max_value=95, value=90, step=1, key="single_put_long_ic")
            call_short = st.sidebar.number_input("Call Short Strike (%)", min_value=100, max_value=115, value=105, step=1, key="single_call_short_ic")
            call_long = st.sidebar.number_input("Call Long Strike (%)", min_value=105, max_value=120, value=110, step=1, key="single_call_long_ic")

            strategy_configs[selected_strategy] = {
                'put_short': put_short / 100,
                'put_long': put_long / 100,
                'call_short': call_short / 100,
                'call_long': call_long / 100
            }

        elif selected_strategy == "Bull Call Spread":
            long_strike = st.sidebar.number_input("Long Strike (%)", min_value=95, max_value=105, value=100, step=1, key="single_long_bcs")
            short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=115, value=105, step=1, key="single_short_bcs")

            strategy_configs[selected_strategy] = {
                'long_strike': long_strike / 100,
                'short_strike': short_strike / 100
            }

        elif selected_strategy == "Long Strangle":
            call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=115, value=105, step=1, key="single_call_lsg")
            put_strike = st.sidebar.number_input("Put Strike (%)", min_value=85, max_value=100, value=95, step=1, key="single_put_lsg")

            strategy_configs[selected_strategy] = {
                'call_strike': call_strike / 100,
                'put_strike': put_strike / 100
            }

        elif selected_strategy == "Bear Put Spread":
            long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=100, step=1, key="single_long_bps")
            short_strike = st.sidebar.number_input("Short Strike (%)", min_value=80, max_value=100, value=95, step=1, key="single_short_bps")

            strategy_configs[selected_strategy] = {
                'long_strike': long_strike / 100,
                'short_strike': short_strike / 100
            }

        elif selected_strategy == "Bear Call Spread":
            short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1, key="single_short_bcs")
            long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=105, step=1, key="single_long_bcs")

            strategy_configs[selected_strategy] = {
                'short_strike': short_strike / 100,
                'long_strike': long_strike / 100
            }

        elif selected_strategy == "Bull Put Spread":
            short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1, key="single_short_bps")
            long_strike = st.sidebar.number_input("Long Strike (%)", min_value=80, max_value=100, value=95, step=1, key="single_long_bps")

            strategy_configs[selected_strategy] = {
                'short_strike': short_strike / 100,
                'long_strike': long_strike / 100
            }

        elif selected_strategy == "Protective Put":
            strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1, key="single_strike_pp")
            underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1, key="single_under_pos_pp")

            strategy_configs[selected_strategy] = {
                'strike_pct': strike_pct / 100,
                'underlying_position': underlying_position
            }

        elif selected_strategy == "Collar":
            call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=120, value=105, step=1, key="single_call_clr")
            put_strike = st.sidebar.number_input("Put Strike (%)", min_value=80, max_value=100, value=95, step=1, key="single_put_clr")
            underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1, key="single_under_pos_clr")

            strategy_configs[selected_strategy] = {
                'call_strike': call_strike / 100,
                'put_strike': put_strike / 100,
                'underlying_position': underlying_position
            }

        elif selected_strategy == "Calendar Spread":
            strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1, key="single_strike_cs")
            strategy_configs[selected_strategy] = {'strike_pct': strike_pct / 100}

        elif selected_strategy == "Butterfly Spread":
            lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=90, max_value=100, value=95, step=1, key="single_lower_bfs")
            middle_strike = st.sidebar.number_input("Middle Strike (%)", min_value=95, max_value=105, value=100, step=1, key="single_middle_bfs")
            upper_strike = st.sidebar.number_input("Upper Strike (%)", min_value=100, max_value=110, value=105, step=1, key="single_upper_bfs")

            strategy_configs[selected_strategy] = {
                'lower_strike': lower_strike / 100,
                'middle_strike': middle_strike / 100,
                'upper_strike': upper_strike / 100
            }

        elif selected_strategy == "Condor Spread":
            lowest_strike = st.sidebar.number_input("Lowest Strike (%)", min_value=90, max_value=95, value=90, step=1, key="single_lowest_cmd")
            lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=95, max_value=100, value=95, step=1, key="single_lower_cmd")
            higher_strike = st.sidebar.number_input("Higher Strike (%)", min_value=100, max_value=105, value=105, step=1, key="single_higher_cmd")
            highest_strike = st.sidebar.number_input("Highest Strike (%)", min_value=105, max_value=110, value=110, step=1, key="single_highest_cmd")

            strategy_configs[selected_strategy] = {
                'lowest_strike': lowest_strike / 100,
                'lower_strike': lower_strike / 100,
                'higher_strike': higher_strike / 100,
                'highest_strike': highest_strike / 100
            }

    # Comparison mode parameters
    if comparison_mode:
        st.sidebar.subheader("📊 Strategy Comparison")
        selected_strategies = st.sidebar.multiselect(
            "Choose Strategies to Compare",
            strategy_options,
            default=[selected_strategy],
            help="Select multiple strategies to compare their performance"
        )

        if selected_strategies:
            st.sidebar.subheader("🎯 Strategy Parameters for Comparison")

            # Configure parameters for each selected strategy
            for i, strat_name in enumerate(selected_strategies):
                st.sidebar.markdown(f"---")
                st.sidebar.markdown(f"### 🎯 {strat_name} Parameters")
                st.sidebar.markdown(f"**Strategy #{i+1}**")
                strategy_configs[strat_name] = {}

                if strat_name == "Covered Call":
                    # Covered Call Strategy Parameters
                    if strat_name == "Covered Call":
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("**Covered Call Options**")
                        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=100, max_value=120, value=105, step=1, key=f"strike_cc_{i}")
                        st.sidebar.markdown("**Underlying Stock**")
                        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1, key=f"under_pos_cc_{i}")
                        underlying_held = st.sidebar.number_input("Already Held", min_value=0, max_value=1000, value=1, step=1, key=f"under_held_cc_{i}")
                        underlying_avg_cost = st.sidebar.number_input("Average Cost ($)", min_value=0.0, max_value=200000.0, value=45000.0, step=100.0, key=f"avg_cost_cc_{i}") if underlying_held > 0 else None

                        strategy_configs[strat_name] = {
                            'strike_pct': strike_pct / 100,
                            'underlying_position': underlying_position,
                            'underlying_held': underlying_held,
                            'underlying_avg_cost': underlying_avg_cost
                        }

                elif strat_name == "Cash Secured Put":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Put Options**")
                    strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1, key=f"strike_csp_{i}")
                    st.sidebar.markdown("*(Requires cash to cover 100 shares)*")

                    strategy_configs[strat_name] = {'strike_pct': strike_pct / 100}

                elif strat_name == "Long Straddle":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Long Straddle Options**")
                    strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1, key=f"strike_ls_{i}")
                    st.sidebar.markdown("*(Call and Put at same strike)*")

                    strategy_configs[strat_name] = {'strike_pct': strike_pct / 100}

                elif strat_name == "Iron Condor":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Put Wing**")
                    put_short = st.sidebar.number_input("Put Short Strike (%)", min_value=85, max_value=100, value=95, step=1, key=f"put_short_ic_{i}")
                    put_long = st.sidebar.number_input("Put Long Strike (%)", min_value=80, max_value=95, value=90, step=1, key=f"put_long_ic_{i}")
                    st.sidebar.markdown("**Call Wing**")
                    call_short = st.sidebar.number_input("Call Short Strike (%)", min_value=100, max_value=115, value=105, step=1, key=f"call_short_ic_{i}")
                    call_long = st.sidebar.number_input("Call Long Strike (%)", min_value=105, max_value=120, value=110, step=1, key=f"call_long_ic_{i}")

                    strategy_configs[strat_name] = {
                        'put_short': put_short / 100,
                        'put_long': put_long / 100,
                        'call_short': call_short / 100,
                        'call_long': call_long / 100
                    }

                elif strat_name == "Bull Call Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Bull Call Spread Options**")
                    long_strike = st.sidebar.number_input("Long Strike (%)", min_value=95, max_value=105, value=100, step=1, key=f"long_bcs_{i}")
                    short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=115, value=105, step=1, key=f"short_bcs_{i}")
                    st.sidebar.markdown("*(Bullish: Long lower, Short higher)*")

                    strategy_configs[strat_name] = {
                        'long_strike': long_strike / 100,
                        'short_strike': short_strike / 100
                    }

                elif strat_name == "Long Strangle":
                    st.sidebar.markdown("---")
                    call_strike = st.sidebar.number_input("**Call Options** - Strike Price (%)", min_value=100, max_value=115, value=105, step=1, key=f"call_lsg_{i}")
                    put_strike = st.sidebar.number_input("**Put Options** - Strike Price (%)", min_value=85, max_value=100, value=95, step=1, key=f"put_lsg_{i}")
                    st.sidebar.markdown("*(OTM Call and Put)*")

                    strategy_configs[strat_name] = {
                        'call_strike': call_strike / 100,
                        'put_strike': put_strike / 100
                    }

                elif strat_name == "Bear Put Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Bear Put Spread Options**")
                    long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=100, step=1, key=f"long_bps_{i}")
                    short_strike = st.sidebar.number_input("Short Strike (%)", min_value=80, max_value=100, value=95, step=1, key=f"short_bps_{i}")
                    st.sidebar.markdown("*(Bearish: Long higher, Short lower)*")

                    strategy_configs[strat_name] = {
                        'long_strike': long_strike / 100,
                        'short_strike': short_strike / 100
                    }

                elif strat_name == "Bear Call Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Bear Call Spread Options**")
                    short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1, key=f"short_bcs_{i}")
                    long_strike = st.sidebar.number_input("Long Strike (%)", min_value=100, max_value=120, value=105, step=1, key=f"long_bcs_{i}")
                    st.sidebar.markdown("*(Bearish: Short lower, Long higher)*")

                    strategy_configs[strat_name] = {
                        'short_strike': short_strike / 100,
                        'long_strike': long_strike / 100
                    }

                elif strat_name == "Bull Put Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Bull Put Spread Options**")
                    short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=120, value=100, step=1, key=f"short_bps_{i}")
                    long_strike = st.sidebar.number_input("Long Strike (%)", min_value=80, max_value=100, value=95, step=1, key=f"long_bps_{i}")
                    st.sidebar.markdown("*(Bullish: Short higher, Long lower)*")

                    strategy_configs[strat_name] = {
                        'short_strike': short_strike / 100,
                        'long_strike': long_strike / 100
                    }

                elif strat_name == "Protective Put":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Protective Put**")
                    strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1, key=f"strike_pp_{i}")
                    st.sidebar.markdown("**Underlying Stock**")
                    underlying_position = st.sidebar.number_input("Stock Position", min_value=0, max_value=1000, value=1, step=1, key=f"under_pos_pp_{i}")

                    strategy_configs[strat_name] = {
                        'strike_pct': strike_pct / 100,
                        'underlying_position': underlying_position
                    }

                elif strat_name == "Collar":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Collar Options**")
                    call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=120, value=105, step=1, key=f"call_clr_{i}")
                    put_strike = st.sidebar.number_input("Put Strike (%)", min_value=80, max_value=100, value=95, step=1, key=f"put_clr_{i}")
                    st.sidebar.markdown("**Underlying Stock**")
                    underlying_position = st.sidebar.number_input("Stock Position", min_value=0, max_value=1000, value=1, step=1, key=f"under_pos_clr_{i}")

                    strategy_configs[strat_name] = {
                        'call_strike': call_strike / 100,
                        'put_strike': put_strike / 100,
                        'underlying_position': underlying_position
                    }

                elif strat_name == "Calendar Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Calendar Spread**")
                    strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1, key=f"strike_cs_{i}")
                    st.sidebar.markdown("*(Short front month, Long back month)*")

                    strategy_configs[strat_name] = {'strike_pct': strike_pct / 100}

                elif strat_name == "Butterfly Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Butterfly Wings**")
                    lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=90, max_value=100, value=95, step=1, key=f"lower_bfs_{i}")
                    middle_strike = st.sidebar.number_input("Middle Strike (%)", min_value=95, max_value=105, value=100, step=1, key=f"middle_bfs_{i}")
                    upper_strike = st.sidebar.number_input("Upper Strike (%)", min_value=100, max_value=110, value=105, step=1, key=f"upper_bfs_{i}")
                    st.sidebar.markdown("*(Long middle, Short wings)*")

                    strategy_configs[strat_name] = {
                        'lower_strike': lower_strike / 100,
                        'middle_strike': middle_strike / 100,
                        'upper_strike': upper_strike / 100
                    }

                elif strat_name == "Condor Spread":
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**Inner Strikes**")
                    lower_strike = st.sidebar.number_input("Lower Strike (%)", min_value=95, max_value=100, value=95, step=1, key=f"lower_cmd_{i}")
                    higher_strike = st.sidebar.number_input("Higher Strike (%)", min_value=100, max_value=105, value=105, step=1, key=f"higher_cmd_{i}")
                    st.sidebar.markdown("**Outer Strikes**")
                    lowest_strike = st.sidebar.number_input("Lowest Strike (%)", min_value=90, max_value=95, value=90, step=1, key=f"lowest_cmd_{i}")
                    highest_strike = st.sidebar.number_input("Highest Strike (%)", min_value=105, max_value=110, value=110, step=1, key=f"highest_cmd_{i}")

                    strategy_configs[strat_name] = {
                        'lowest_strike': lowest_strike / 100,
                        'lower_strike': lower_strike / 100,
                        'higher_strike': higher_strike / 100,
                        'highest_strike': highest_strike / 100
                    }

        st.sidebar.markdown("---")

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

            if comparison_mode and 'selected_strategies' in locals():
                # Create strategy objects with custom parameters
                strategies = []
                for strat_name in selected_strategies:
                    config = strategy_configs[strat_name]
                    if strat_name == "Covered Call":
                        strategies.append(OptionStrategyTemplate.covered_call(**config))
                    elif strat_name == "Cash Secured Put":
                        strategies.append(OptionStrategyTemplate.cash_secured_put(**config))
                    elif strat_name == "Long Straddle":
                        strategies.append(OptionStrategyTemplate.long_straddle(**config))
                    elif strat_name == "Iron Condor":
                        strategies.append(OptionStrategyTemplate.iron_condor(**config))
                    elif strat_name == "Bull Call Spread":
                        strategies.append(OptionStrategyTemplate.bull_call_spread(**config))
                    elif strat_name == "Long Strangle":
                        strategies.append(OptionStrategyTemplate.long_strangle(**config))
                    elif strat_name == "Bear Put Spread":
                        strategies.append(OptionStrategyTemplate.bear_put_spread(**config))
                    elif strat_name == "Bear Call Spread":
                        strategies.append(OptionStrategyTemplate.bear_call_spread(**config))
                    elif strat_name == "Bull Put Spread":
                        strategies.append(OptionStrategyTemplate.bull_put_spread(**config))
                    elif strat_name == "Protective Put":
                        strategies.append(OptionStrategyTemplate.protective_put(**config))
                    elif strat_name == "Collar":
                        strategies.append(OptionStrategyTemplate.collar(**config))
                    elif strat_name == "Calendar Spread":
                        strategies.append(OptionStrategyTemplate.calendar_spread(**config))
                    elif strat_name == "Butterfly Spread":
                        strategies.append(OptionStrategyTemplate.butterfly_spread(**config))
                    elif strat_name == "Condor Spread":
                        strategies.append(OptionStrategyTemplate.condor_spread(**config))

                # Add volatility parameter
                backtest_params['volatility'] = volatility if not use_yf_vol else 0.5

                # Compare strategies with custom parameters
                comparison_results = backtester.compare_strategies(strategies, **backtest_params)

                # Store results in session state
                st.session_state.comparison_results = comparison_results
                st.session_state.asset_name = asset_name
            elif optimize_mode:
                st.info("Optimization mode is available - use single strategy for now")
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
    if hasattr(st.session_state, 'stats_str'):
        st.header(f"📊 {st.session_state.strategy_name} - {st.session_state.asset_name} Backtest Results")

        # Create two columns for layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Performance Summary")

            # Format and display stats table
            stats_df = format_stats_table(st.session_state.stats_str)
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.text(st.session_state.stats_str)

        with col2:
            st.subheader("📊 Advanced Performance Metrics")

            # Extract key metrics for display
            if not stats_df.empty:
                metrics_dict = dict(zip(stats_df['Metric'], stats_df['Value']))

                # Create tabs for different metric categories
                tab_basic, tab_risk, tab_advanced = st.tabs(["📈 Basic", "⚠️ Risk", "🎯 Advanced"])

                with tab_basic:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'total_return' in metrics_dict:
                            try:
                                value = float(metrics_dict['total_return'])
                                st.metric("Total Return", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("Total Return", f"{metrics_dict['total_return']}%")
                        if 'win_rate' in metrics_dict:
                            try:
                                value = float(metrics_dict['win_rate'])
                                st.metric("Win Rate", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("Win Rate", f"{metrics_dict['win_rate']}%")

                    with col2:
                        if 'total_pnl' in metrics_dict:
                            try:
                                value = float(metrics_dict['total_pnl'])
                                st.metric("Total P&L", f"${value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Total P&L", f"${metrics_dict['total_pnl']}")
                        if 'sharpe_ratio' in metrics_dict:
                            try:
                                value = float(metrics_dict['sharpe_ratio'])
                                st.metric("Sharpe Ratio", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Sharpe Ratio", metrics_dict['sharpe_ratio'])

                    with col3:
                        if 'total_trades' in metrics_dict:
                            st.metric("Total Trades", metrics_dict['total_trades'])
                        if 'payoff_ratio' in metrics_dict:
                            try:
                                value = float(metrics_dict['payoff_ratio'])
                                st.metric("Payoff Ratio", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Payoff Ratio", metrics_dict['payoff_ratio'])

                with tab_risk:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'max_drawdown' in metrics_dict:
                            try:
                                value = float(metrics_dict['max_drawdown'])
                                st.metric("Max Drawdown", f"${value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Max Drawdown", f"${metrics_dict['max_drawdown']}")
                        if 'annual_volatility' in metrics_dict:
                            try:
                                value = float(metrics_dict['annual_volatility'])
                                st.metric("Volatility", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("Volatility", f"{metrics_dict['annual_volatility']}%")

                    with col2:
                        if 'value_at_risk_95' in metrics_dict:
                            try:
                                value = float(metrics_dict['value_at_risk_95'])
                                st.metric("VaR (95%)", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("VaR (95%)", f"{metrics_dict['value_at_risk_95']}%")
                        if 'expected_shortfall' in metrics_dict:
                            try:
                                value = float(metrics_dict['expected_shortfall'])
                                st.metric("Expected Shortfall", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("Expected Shortfall", f"{metrics_dict['expected_shortfall']}%")

                    with col3:
                        if 'recovery_factor' in metrics_dict:
                            try:
                                value = float(metrics_dict['recovery_factor'])
                                st.metric("Recovery Factor", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Recovery Factor", metrics_dict['recovery_factor'])
                        if 'profit_factor' in metrics_dict:
                            try:
                                value = float(metrics_dict['profit_factor'])
                                st.metric("Profit Factor", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Profit Factor", metrics_dict['profit_factor'])

                with tab_advanced:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'sortino_ratio' in metrics_dict:
                            try:
                                value = float(metrics_dict['sortino_ratio'])
                                st.metric("Sortino Ratio", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Sortino Ratio", metrics_dict['sortino_ratio'])
                        if 'calmar_ratio' in metrics_dict:
                            try:
                                value = float(metrics_dict['calmar_ratio'])
                                st.metric("Calmar Ratio", f"{value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Calmar Ratio", metrics_dict['calmar_ratio'])

                    with col2:
                        if 'best_day' in metrics_dict:
                            try:
                                value = float(metrics_dict['best_day'])
                                st.metric("Best Day", f"${value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Best Day", f"${metrics_dict['best_day']}")
                        if 'worst_day' in metrics_dict:
                            try:
                                value = float(metrics_dict['worst_day'])
                                st.metric("Worst Day", f"${value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Worst Day", f"${metrics_dict['worst_day']}")

                    with col3:
                        if 'std_dev_returns' in metrics_dict:
                            try:
                                value = float(metrics_dict['std_dev_returns'])
                                st.metric("Return Std Dev", f"{value:.2f}%")
                            except (ValueError, TypeError):
                                st.metric("Return Std Dev", metrics_dict['std_dev_returns'])
                        if 'std_pnl' in metrics_dict:
                            try:
                                value = float(metrics_dict['std_pnl'])
                                st.metric("P&L Std Dev", f"${value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("P&L Std Dev", metrics_dict['std_pnl'])

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
                        pass  # Placeholder for rolling Sharpe

                # Risk Analysis Tab
                with tab3:
                    pass  # Placeholder

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

                # Round numeric columns
                numeric_columns = ['entry_price', 'expiry_price', 'one_day_pct_move', 'expiry_pct_move',
                                 'option_gains', 'option_price', 'underlying_pnl', 'current_pnl']
                for col in numeric_columns:
                    if col in daily_display.columns:
                        daily_display[col] = daily_display[col].round(2)

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
                st.dataframe(
                    st.session_state.weekday_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Weekday summary data not available")

    elif hasattr(st.session_state, 'comparison_results'):
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

            # Display the comparison table
            st.subheader("📈 Performance Comparison")

            # Style the dataframe with color coding
            def color_negative_red_positive_green(val):
                try:
                    numeric_val = float(val)
                    if numeric_val > 0:
                        return 'color: green'
                    elif numeric_val < 0:
                        return 'color: red'
                    else:
                        return 'color: black'
                except:
                    return 'color: black'

            styled_pivot = pivot_df.style.map(color_negative_red_positive_green)
            st.dataframe(styled_pivot, use_container_width=True)

            # Add visual comparison charts
            st.subheader("📊 Visual Strategy Comparison")

            # Create tabs for different comparison views
            tab_table, tab_bars, tab_radar = st.tabs(["📋 Comparison Table", "📊 Bar Charts", "🕸️ Radar Chart"])

            with tab_table:
                # Enhanced table with rankings
                if 'win_rate' in pivot_df.index:
                    win_rates = pd.to_numeric(pivot_df.loc['win_rate'], errors='coerce')
                    win_rates = win_rates.sort_values(ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"🏆 Highest Win Rate", f"{win_rates.iloc[0]}%", f"+{win_rates.iloc[0] - win_rates.iloc[-1]:.1f}% vs lowest")

                    with col2:
                        total_returns = pd.to_numeric(pivot_df.loc['percentage_returns'], errors='coerce')
                        if not total_returns.empty:
                            best_return = total_returns.max()
                            st.metric(f"💰 Best Total Return", f"{best_return}%")

            with tab_bars:
                # Bar charts for key metrics - expanded with advanced metrics
                key_metrics = [
                    'win_rate', 'percentage_returns', 'sharpe_ratio', 'sortino_ratio',
                    'max_drawdown', 'annual_volatility', 'payoff_ratio', 'recovery_factor'
                ]

                for metric in key_metrics:
                    if metric in pivot_df.index:
                        fig = go.Figure()
                        values = pd.to_numeric(pivot_df.loc[metric], errors='coerce')

                        # For metrics where negative is good (like max_drawdown), reverse color logic
                        if metric in ['max_drawdown', 'annual_volatility']:
                            colors = ['red' if x >= 0 else 'green' for x in values]
                        else:
                            colors = ['green' if x >= 0 else 'red' for x in values]

                        fig.add_trace(go.Bar(
                            x=values.index,
                            y=values.values,
                            marker_color=colors,
                            text=[f"{x:.2f}" for x in values],
                            textposition='auto',
                        ))

                        metric_titles = {
                            'win_rate': 'Win Rate (%)',
                            'percentage_returns': 'Total Return (%)',
                            'sharpe_ratio': 'Sharpe Ratio',
                            'sortino_ratio': 'Sortino Ratio',
                            'max_drawdown': 'Max Drawdown ($)',
                            'annual_volatility': 'Annual Volatility (%)',
                            'payoff_ratio': 'Payoff Ratio',
                            'recovery_factor': 'Recovery Factor'
                        }

                        fig.update_layout(
                            title=f"{metric_titles.get(metric, metric.title())} by Strategy",
                            xaxis_title="Strategy",
                            yaxis_title=metric_titles.get(metric, metric.title()),
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)

            with tab_radar:
                # Radar chart for multi-metric comparison (normalized)
                radar_metrics = ['win_rate', 'sharpe_ratio', 'percentage_returns']

                # Normalize metrics to 0-1 scale for radar chart
                normalized_data = {}
                for metric in radar_metrics:
                    if metric in pivot_df.index:
                        values = pd.to_numeric(pivot_df.loc[metric], errors='coerce')
                        min_val = values.min()
                        max_val = values.max()
                        if max_val > min_val:
                            normalized_values = (values - min_val) / (max_val - min_val)
                            normalized_data[metric] = normalized_values

                if normalized_data:
                    strategy_names = list(normalized_data[list(normalized_data.keys())[0]].index)

                    fig = go.Figure()

                    for strategy in strategy_names:
                        r_values = [normalized_data[metric][strategy] for metric in radar_metrics]
                        fig.add_trace(go.Scatterpolar(
                            r=r_values,
                            theta=radar_metrics,
                            fill='toself',
                            name=strategy
                        ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        title="Strategy Performance Radar Chart",
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.info("💡 **Radar Chart Explanation:** Each axis represents a normalized metric (0-1 scale). Larger area indicates better overall performance across all metrics.")

            # Download button for comparison data
            csv_comparison = pivot_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data",
                data=csv_comparison,
                file_name=f"strategy_comparison_{st.session_state.asset_name}.csv",
                mime="text/csv"
            )

            # Strategy recommendations
            if 'win_rate' in pivot_df.index and 'percentage_returns' in pivot_df.index:
                win_rates = pd.to_numeric(pivot_df.loc['win_rate'], errors='coerce')
                returns = pd.to_numeric(pivot_df.loc['percentage_returns'], errors='coerce')

                if not win_rates.empty and not returns.empty:
                    best_win_rate = win_rates.idxmax()
                    best_return = returns.idxmax()

                    st.subheader("🎯 Strategy Recommendations")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.success(f"🎲 **Highest Win Rate:** {best_win_rate}")
                        st.info(f"Win Rate: {win_rates[best_win_rate]}%")

                    with col2:
                        st.success(f"💰 **Best Total Return:** {best_return}")
                        st.info(f"Total Return: {returns[best_return]}%")

                    with col3:
                        if 'sharpe_ratio' in pivot_df.index:
                            sharpe = pd.to_numeric(pivot_df.loc['sharpe_ratio'], errors='coerce')
                            if not sharpe.empty:
                                best_sharpe = sharpe.idxmax()
                                st.success(f"📊 **Best Risk-Adjusted:** {best_sharpe}")
                                st.info(f"Sharpe Ratio: {sharpe[best_sharpe]}")

            st.markdown("---")
            st.info("💡 **Strategy Comparison Analysis:** Compare multiple options strategies side-by-side to identify the best performing approach for your trading goals. Higher win rates indicate more consistent performance, while higher total returns show better profitability.")
        else:
            st.warning("No valid comparison data available.")
            st.info("Try selecting different strategies or adjusting your backtest parameters.")
    else:
        st.info("⚡ Configure strategy parameters with sidebar and click 'Run Backtest' to see comprehensive results and analysis above.")

if __name__ == "__main__":
    main()
