# Stock Options Strategy Backtester Dashboard

A comprehensive Python application for backtesting various stock options strategies using historical data with performance metrics and visualizations.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.1-red)

## 📊 Overview

This application allows users to backtest 14 different options strategies against historical stock data to evaluate their performance. It provides detailed analytics, performance metrics, and visualizations to help traders make informed decisions.

Access via Streamlit: https://optionsbacktester-dashboard.streamlit.app

## 🚀 Features

### Supported Strategies

1. **Covered Call** - Long stock + Short call
2. **Cash Secured Put** - Short put with cash backing
3. **Long Straddle** - Long call + Long put at same strike
4. **Iron Condor** - Short put spread + Short call spread
5. **Bull Call Spread** - Long call + Short call at higher strike
6. **Long Strangle** - Long OTM call + Long OTM put
7. **Bear Put Spread** - Long higher strike put + Short lower strike put
8. **Bear Call Spread** - Short lower strike call + Long higher strike call
9. **Bull Put Spread** - Short higher strike put + Long lower strike put
10. **Protective Put** - Long underlying + Long put
11. **Collar** - Long underlying + Short call + Long put
12. **Calendar Spread** - Short near-term option + Long longer-term option
13. **Butterfly Spread** - Long ITM + Short 2 ATM + Long ITM
14. **Condor Spread** - Bull call spread + Bear call spread

### Performance Metrics

- Total Return & P&L
- Win Rate & Trade Count
- Sharpe Ratio & Max Drawdown
- Profit Factor & Volatility
- Monthly performance breakdown
- Rolling performance analysis

### Visualizations

The dashboard provides 8 different analytical charts:
- Cumulative P&L vs benchmark
- Return distribution histograms
- Monthly win rate analysis
- Day-of-week performance
- Price movement correlation
- Rolling performance metrics
- Entry vs expiry price scatter
- Monthly P&L heatmap

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd option-strategy-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. The dashboard will open in your default web browser.

### Dashboard Guide

1. **Select Strategy**: Choose your desired options strategy from the sidebar dropdown
2. **Configure Parameters**: Adjust strategy-specific parameters (strikes, premiums, etc.)
3. **Set Backtesting Parameters**: Configure expiry days, interest rate, and volatility
4. **Select Stock Data**: Enter a stock ticker symbol and date range
5. **Run Backtest**: Click the "Run Backtest" button to see results
6. **Analyze Results**: Review performance metrics and visualizations

### Advanced Features

- **Strategy Comparison**: Compare multiple strategies side-by-side
- **Parameter Optimization**: Find optimal strategy parameters
- **Day-of-Week Filtering**: Analyze performance for specific days of the week
- **Data Export**: Download results as CSV files

## 🧠 How It Works

The backtester uses the Black-Scholes pricing model to calculate option premiums and simulates strategy performance over historical data. Key components include:

1. **Strategy Templates**: Pre-defined templates for all 14 options strategies
2. **Backtesting Engine**: Processes historical data and calculates P&L for each trade
3. **Risk Metrics**: Calculates Sharpe ratio, max drawdown, and other risk metrics
4. **Visualization Engine**: Creates interactive charts using Plotly

### Data Sources

The application uses yfinance to fetch real historical stock data, including:
- Open, High, Low, Close prices
- Trading volume
- Options chain data for implied volatility

## 📈 Performance Analysis

The dashboard provides comprehensive performance analysis including:

- **Cumulative Returns**: Track strategy performance over time
- **Risk Metrics**: Sharpe ratio, maximum drawdown, and volatility
- **Trade Statistics**: Win rate, average profit/loss per trade
- **Distribution Analysis**: P&L distribution histograms
- **Time-Based Analysis**: Monthly and day-of-week performance

## 🛠️ Technical Architecture

### Core Components

1. **Option Strategy Classes**: Define strategy structures and payoff calculations
2. **Backtesting Engine**: Processes historical data and executes strategies
3. **Streamlit Dashboard**: Interactive web interface for strategy configuration and results
4. **Visualization Module**: Creates analytical charts and graphs

### Key Classes

- `OptionLeg`: Represents a single option position
- `OptionStrategy`: Container for multiple option legs
- `OptionStrategyTemplate`: Factory for common strategies
- `OptionsBacktester`: Main backtesting engine
- `Streamlit Dashboard`: Interactive UI components

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Uses yfinance for stock data retrieval
- Built with Streamlit for the web interface
- Implements Black-Scholes option pricing model
- Inspired by options trading strategies from financial literature

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **yfinance API Issues**: Check internet connectivity and yfinance service status
3. **Data Loading Errors**: Verify stock ticker symbols are correct
4. **Performance Issues**: Reduce date range or use a more powerful machine

### Support

For support, please open an issue on the GitHub repository.

## 📅 Future Enhancements

Planned improvements include:
- Additional options strategies
- Real-time data integration
- Advanced risk management features
- Portfolio-level backtesting
- Machine learning optimization techniques
