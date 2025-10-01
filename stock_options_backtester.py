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
                        'upper_strike': [x/100.0 for x in range(upper_strike_range[0], middle_strike_range[1]+1)]
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

        # Enhanced Tabbed Interface for additional analysis after main results
        if not (comparison_mode or optimize_mode):
            st.markdown("---")  # Separator
            additional_tab1, additional_tab2, additional_tab3 = st.tabs([
                "📊 Advanced Analytics", "⚖️ Market Benchmark", "💼 Portfolio Allocation"
            ])

            with additional_tab1:
                st.header("🎯 Advanced Analytics Dashboard")

                if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                    charts = create_additional_charts(
                        st.session_state.daily_df,
                        st.session_state.strategy_name,
                        st.session_state.asset_name
                    )

                    if charts:
                        # Create sub-tabs for different chart categories
                        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                            "📈 Performance Overview", "📊 Distribution Analysis", "📉 Time Series", "🛡️ Risk Metrics"
                        ])

                        with sub_tab1:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Performance Metrics")
                                stats_df = format_stats_table(st.session_state.stats_str)
                                if not stats_df.empty:
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            with col2:
                                st.subheader("Strategy Insights")
                                st.info("📊 Comprehensive performance analysis with multiple perspectives")
                                st.info("🔍 Advanced distribution analysis for trade outcomes")
                                st.info("📈 Time-based performance tracking with risk metrics")

                        with sub_tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'pnl_distribution' in charts:
                                    st.plotly_chart(charts['pnl_distribution'], use_container_width=True)
                                    st.caption("Distribution of Profit & Loss for all trades")
                                else:
                                    st.info("P&L distribution chart not available")
                            with col2:
                                if 'win_streak' in charts:
                                    st.plotly_chart(charts['win_streak'], use_container_width=True)
                                    st.caption("Analysis of winning streak patterns")
                                else:
                                    st.info("Win streak analysis not available")

                        with sub_tab3:
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'cumulative_pnl' in charts:
                                    st.plotly_chart(charts['cumulative_pnl'], use_container_width=True)
                                    st.caption("Cumulative P&L over time")
                                else:
                                    st.info("Cumulative P&L chart not available")
                            with col2:
                                if 'rolling_sharpe' in charts:
                                    st.plotly_chart(charts['rolling_sharpe'], use_container_width=True)
                                    st.caption("30-day rolling Sharpe ratio")
                                else:
                                    st.info("Rolling Sharpe chart not available")

                        with sub_tab4:
                            if 'drawdown' in charts:
                                st.plotly_chart(charts['drawdown'], use_container_width=True)
                                st.caption("Drawdown analysis showing maximum losses over time")
                            else:
                                st.info("Drawdown analysis not available")

                            # Add risk metrics section
                            st.subheader("Advanced Risk Metrics")
                            risk_metrics_cols = st.columns(4)
                            with risk_metrics_cols[0]:
                                st.metric("Value at Risk (95%)", "TBD", delta="Low")
                            with risk_metrics_cols[1]:
                                st.metric("Expected Shortfall", "TBD", delta="Medium")
                            with risk_metrics_cols[2]:
                                st.metric("Sortino Ratio", "TBD", delta="Good")
                            with risk_metrics_cols[3]:
                                st.metric("Calmar Ratio", "TBD", delta="High")
                    else:
                        st.info("Enhanced analytics charts not available for this backtest.")
                else:
                    st.info("Advanced analytics require completed backtest data.")

            with additional_tab2:
                st.header("⚖️ Market Benchmarking & Risk Analysis")

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
                        st.info("💡 This chart compares your strategy's performance against the S&P 500 benchmark, helping you understand relative performance and market timing.")
                    else:
                        st.info("Benchmark comparison not available")

                    # Enhanced risk metrics display
                    st.subheader("Risk-Adjusted Performance Metrics")
                    risk_cols = st.columns(3)
                    with risk_cols[0]:
                        st.metric("Beta to S&P 500", "TBD", delta="Low")
                    with risk_cols[1]:
                        st.metric("Alpha", "TBD", delta="Good")
                    with risk_cols[2]:
                        st.metric("Info Ratio", "TBD", delta="High")

                    st.markdown("---")
                    st.markdown("**📋 Risk Analysis Guide:**")
                    st.markdown("• **Beta**: Measures volatility relative to market (1.0 = same as market)")
                    st.markdown("• **Alpha**: Measures excess returns above market expectations")
                    st.markdown("• **Info Ratio**: Risk-adjusted excess return relative to benchmark")
                    st.markdown("• **VaR (95%)**: Maximum expected loss over a day with 95% confidence")

                else:
                    st.info("Market benchmarking requires completed backtest data.")

            with additional_tab3:
                st.header("💼 Portfolio Management & Allocation")

                # Portfolio allocation suggestions
                st.subheader("Strategy Integration Recommendations")

                # Display portfolio metrics and suggestions
                port_cols = st.columns(2)
                with port_cols[0]:
                    st.metric("Suggested Allocation", "5-15%")
                    st.metric("Max Portfolio Beta Impact", "< 0.05")
                    st.metric("Recommended Rebalancing", "Quarterly")

                with port_cols[1]:
                    st.info("💡 **Conservative Investors**: Start with 5% allocation")
                    st.info("📈 **Balanced Investors**: Consider 10-15% portfolio allocation")
                    st.info("🔥 **Aggressive Investors**: Can go up to 20% if risk tolerance allows")
                    st.info("⚠️ **Always test thoroughly before live implementation**")

                # Strategy diversification matrix
                st.subheader("Strategy Diversification Matrix")

                # Create a simple diversification suggestion table
                diversification_data = {
                    "Strategy Type": ["Covered Call", "Cash Secured Put", "Iron Condor", "Long Strangle", "Bull Call Spread"],
                    "Risk Level": ["Low", "Low", "Medium", "High", "Medium"],
                    "Return Potential": ["Moderate", "Moderate", "Low", "High", "Medium"],
                    "Portfolio Allocation %": ["15%", "15%", "20%", "5%", "10%"],
                    "Best For": ["Income", "Income", "Income", "Speculation", "Directional"]
                }

                div_df = pd.DataFrame(diversification_data)
                st.table(div_df)

                # Capital requirements section
                st.subheader("Capital Requirements & Position Sizing")

                # Mock capital requirements (in thousands)
                capital_reqs = pd.DataFrame({
                    "Strategy": ["Covered Call", "Cash Secured Put", "Iron Condor", "Long Straddle", "Bull Call Spread"],
                    "Min Capital ($K)": [50, 100, 25, 10, 25],
                    "Max Risk per Trade": ["$2-5K", "$10K", "$1-2K", "$10K", "$2-5K"],
                    "Max Positions": [20, 10, 40, 10, 20]
                })

                st.table(capital_reqs)

                st.markdown("---")
                st.markdown("**💡 Portfolio Integration Tips:**")
                st.markdown("• Start small and scale up as you gain confidence")
                st.markdown("• Monitor portfolio correlations carefully")
                st.markdown("• Use stop-losses and position size limits")
                st.markdown("• Consider tax implications of options strategies")

    else:
        st.info("⚡ Configure strategy parameters with sidebar and click 'Run Backtest' to see comprehensive results and analysis above.")

if __name__ == "__main__":
    main()
