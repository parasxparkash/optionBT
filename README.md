# 📈 Options Strategy Backtester Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://optionbt.streamlit.app/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/parasxparkash/optionBT.git)

## Overview

**Options Strategy Backtester Pro** is a comprehensive analytical platform for backtesting and comparing options trading strategies. This advanced tool provides professional-grade analysis with extensive metrics, visual comparisons, and detailed performance insights.

> 🌟 **Enhanced Version 1.1.0** - Professional analytics platform with 16 advanced metrics, comprehensive strategy comparisons, and production-ready code.

## 🚀 Key Features

### 📊 **Professional Analytics & Metrics**
- **16 Advanced Performance Metrics** across 3 categories
- **Real-time Data Validation** with comprehensive market statistics
- **Risk-Adjusted Performance** analysis (Sharpe, Sortino, Calmar ratios)
- **Custom Parameter Configurations** for each strategy

### 🎯 **Strategy Comparison Engine**
- **Multi-Strategy Comparison** with individual parameter settings
- **Visual Analytics** - Bar charts, radar charts, and comparative tables
- **Performance Rankings** with clear winner identification
- **Side-by-Side Analysis** for informed strategy selection

### 💾 **Data Management**
- **Market Data Preview** with complete statistics dashboard
- **Export Capabilities** for trade history and comparison results
- **Real-time Validation** of input parameters
- **Error-Free Operation** with robust error handling

### 🔄 **User Experience**
- **Intuitive Interface** with collapsible sections
- **Parameter Organization** with clear visual hierarchy
- **Responsive Design** working on desktop environments
- **Progress Indicators** and loading states

## 🎨 **Enhanced UI Components**

### 1. **Market Data Preview Dashboard**
- 📅 Complete data statistics (points, date range, trading days)
- 💰 Price metrics (starting price, latest price, total change)
- 📊 Volume statistics (average, peak, total volume)
- 📈 Volatility analysis (annual volatility, price range)
- 📋 Interactive price charts and export capabilities

### 2. **Advanced Performance Metrics**
- **📈 Basic Metrics**: Total Return, Win Rate, Sharpe Ratio
- **⚠️ Risk Metrics**: Max Drawdown, Volatility, Value at Risk
- **🎯 Advanced Metrics**: Sortino Ratio, Recovery Factor, Payoff Ratio

### 3. **Strategy Comparison System**
```
📊 Strategy Comparison
├── Multi-Select Strategy Choice
├── Individual Parameter Configuration for Each Strategy
├── Visual Performance Comparisons
├── Strategy Recommendation Engine
└── Exportable Results
```

## 🛠️ **Technical Stack**

- **Frontend**: Streamlit
- **Backend**: Python 3.7+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Market Data**: yfinance API
- **Analytical Engine**: Custom Options Backtester

## ⚡ **Installation & Setup**

### 1. **Prerequisites**
```bash
# Ensure Python 3.7+ is installed
python --version

# Install required packages
pip install streamlit pandas numpy plotly yfinance
```

### 2. **Clone & Run**
```bash
# Clone the repository
git clone https://github.com/parasxparkash/optionBT.git
cd optionBT

# Run the application
streamlit run app.py
```

### 3. **Access**
- Opens automatically at: `http://localhost:8501`
- Works in any modern web browser

## 📖 **How to Use**

### **Step 1: Configure Market Data**
1. **Select Ticker Symbol** (e.g., AAPL, TSLA, SPY)
2. **Choose Date Range** (default: 2020-01-01 to present)
3. **Review Market Data Preview** - automatically displays complete statistics

### **Step 2: Select Strategy**
1. **Choose Strategy Type** from 14 available options:
   - Covered Call, Cash Secured Put, Long Straddle
   - Iron Condor, Bull/Bear Call/Put Spreads
   - Collar, Calendar Spread, Butterfly/Condor
   - Protective Put

2. **Configure Parameters** - customize strike prices, positions, etc.

### **Step 3: Run Backtest**
1. **Set Backtesting Parameters**:
   - Expiry Days (1-30)
   - Interest Rate (0-10%)
   - Trade Frequency (non-overlapping, daily, weekly, monthly)
   - Day-of-week filters

2. **Click "Run Backtest"** for instant analysis

### **Step 4: Analyze Results**
1. **Performance Summary** - 16 comprehensive metrics
2. **Visual Analysis** - charts for P&L distribution, time series
3. **Export Options** - download trade data and results

## 🎯 **Strategy Comparison Mode**

### **Enable Advanced Comparison**
1. **Check "Compare Strategies"** checkbox
2. **Select Multiple Strategies** from the multiselect box
3. **Configure Each Strategy** - individual parameter sections appear

### **Individual Parameter Configuration**
Each strategy gets its own dedicated section:
```
🎯 Strategy #1: Covered Call
├── Covered Call Options [strike price]
└── Underlying Stock [position, cost]

🎯 Strategy #2: Cash Secured Put
├── Put Options [strike price]
└── Cash Requirements [amount]
```

### **Comparative Analysis**
- **Performance Table** with color-coded results
- **Visual Charts**: Bar charts for each metric
- **Radar Chart**: Multi-dimensional performance comparison
- **Strategy Rankings** with clear recommendations

## 📊 **Performance Metrics Deep Dive**

### **📈 Basic Metrics**
- **Total Return (%)**: Overall profitability
- **Win Rate (%)**: Percentage of profitable trades
- **Total P&L ($)**: Absolute profit/loss amount
- **Sharpe Ratio**: Risk-adjusted returns

### **⚠️ Risk Metrics**
- **Max Drawdown ($)**: Largest peak-to-trough decline
- **Annual Volatility (%)**: Price fluctuation intensity
- **Value at Risk (95%)**: Potential loss in adverse scenarios
- **Expected Shortfall**: Average loss beyond VaR

### **🎯 Advanced Metrics**
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Risk-adjusted returns vs drawdown
- **Recovery Factor**: Profit relative to maximum drawdown
- **Profit Factor**: Gross profit divided by gross loss

## 🏗️ **Architecture & Design**

### **Modular Code Structure**
```
├── app.py                 # Main Streamlit application
├── backtester.py          # Core backtesting engine
├── stock_options_backtester.py  # Legacy backtester
├── app_improvements_documentation.md  # Feature documentation
└── README.md             # This documentation
```

### **Key Components**
- **Data Pipeline**: yfinance → pandas → validation
- **Strategy Engine**: Template-based strategy implementations
- **Analytics Engine**: Comprehensive metric calculations
- **UI Framework**: Streamlit with custom styling

## 🔧 **Advanced Configuration**

### **Trade Frequency Options**
- **Non-overlapping**: One trade per expiry period
- **Daily**: New trades every day
- **Weekly**: New trades every week
- **Monthly**: New trades every month

### **Entry Day Filtering**
- **All Days**: No filtering
- **Monday-Friday**: Specific day entry restrictions
- **Weekdays/Weekends**: Time-based filtering

### **Volatility Sources**
- **yfinance Implied Volatility**: Real market data
- **Custom Volatility**: User-defined percentage

## 📈 **Sample Analysis**

### **Strategy Comparison Example**
Compare three strategies on AAPL data:

| Strategy | Win Rate | Total Return | Sharpe | Max Drawdown |
|----------|----------|--------------|--------|--------------|
| Covered Call | 78% | 24.5% | 1.8 | $3,200 |
| Cash Secured Put | 65% | 18.2% | 1.4 | $2,800 |
| Iron Condor | 72% | 21.1% | 1.6 | $4,100 |

### **Market Data Validation**
- **Data Points**: 1,094 trading days
- **Date Range**: 2020-01-01 to 2024-10-01
- **Annual Volatility**: 28.7%
- **Average Volume**: 56.2M shares

## 🐛 **Troubleshooting**

### **Common Issues**
- **"Failed to load stock data"**: Check ticker symbol spelling
- **"No chart data available"**: Insufficient historical data for parameters
- **Slow performance**: Reduce date range or simplify parameters

### **Data Requirements**
- **Minimum Period**: 1 year for meaningful statistics
- **Ticker Symbols**: Valid NASDAQ/NYSE/MFST symbols
- **Volatility Data**: Available for most major assets

## 🚀 **Future Enhancements**

Planned improvements include:
- **Portfolio Management**: Multi-asset position tracking
- **Real-time Data**: Live option chain integration
- **Risk Management**: Advanced stop-loss and position sizing
- **Strategy Optimization**: Automated parameter optimization
- **Machine Learning**: Predictive strategy performance

## 📝 **Documentation**

Detailed improvement documentation available in:
- [`app_improvements_documentation.md`](app_improvements_documentation.md) - Complete feature guide
- Inline code comments for all major functions
- Parameter explanations in UI tooltips

## 🤝 **Contributing**

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ **Support**

For questions, issues, or feature requests:
- Create GitHub issues for bugs/feature requests
- Check existing documentation first
- Include screenshots for UI-related problems

---

**Made with ❤️ for options traders and quantitative analysts**

**🌐 Live Demo**: [https://optionbt.streamlit.app/](https://optionbt.streamlit.app/)

**📦 Repository**: [https://github.com/parasxparkash/optionBT.git](https://github.com/parasxparkash/optionBT.git)

**💻 Local Demo**: Run `streamlit run app.py` locally
