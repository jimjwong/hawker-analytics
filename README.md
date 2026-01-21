# NEA Hawker Tender Analytics Dashboard

## ⚠️ Disclaimer

**This project is for analytics exploration and educational purposes only.** It is a work in progress and may contain errors, bugs, and incomplete features. The data and predictions should not be used as the sole basis for any real-world bidding decisions. Always conduct your own thorough research and due diligence.

---

## Overview

This dashboard provides analytics and insights into NEA (National Environment Agency) hawker stall tender data in Singapore. It's designed to help potential bidders explore historical bidding patterns and estimate fair bid prices using machine learning models.

## Features

### 📊 Data Exploration
- **Overview Metrics**: Total bids, unique hawker centres, trades, and average winning bids
- **Market Dynamics**: Distribution of bids, winning bid trends over time, and top centres by bid value
- **Trade Insights**: Bid levels by trade type and competition analysis

### 🤖 Machine Learning Models
- **Price Prediction**: Estimate expected winning bid prices using Random Forest Regression
- **Win Probability**: Calculate probability of winning with a specific bid amount
- **Bid Strategy Simulator**: Visualize how win probability changes across different bid values

### 🎯 Centre Analysis
- **Segmentation**: K-means clustering of hawker centres based on bid patterns
- **Scorecard**: Detailed metrics for individual hawker centres
- **Historical Trends**: Track winning bids over time for specific locations

### 📋 Data Tables
- Browse all bids (winners and losers)
- Filter and download winning bids
- Summary statistics by hawker centre and trade type

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jimjwong/hawker-analytics.git
cd hawker-analytics
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run hawker_dashboard-v1_0.py
```

The app will open in your browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- pandas
- numpy
- streamlit
- matplotlib
- plotly
- scikit-learn

See `requirements.txt` for complete list of dependencies.

## Data

The dashboard uses `nea_bids_cleaned.csv` which contains historical NEA hawker stall tender data including:
- Hawker centre names
- Trade types
- Bid values
- Number of bidders
- Winner information
- Tender dates

## Project Status

🚧 **Work in Progress** 🚧

This project is actively being developed. Known issues and planned improvements:
- Model accuracy improvements
- Additional visualization features
- Enhanced error handling
- Data validation and cleaning
- Performance optimization

## Contributing

This is a personal learning project, but suggestions and feedback are welcome!

## License

This project is open source and available for educational purposes.

## Contact

For questions or feedback, please open an issue on GitHub.

---

*Last updated: January 2026*

