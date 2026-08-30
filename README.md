# 🏡 House Price Prediction Web Application

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-brightgreen?style=flat-square&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An interactive, production-ready machine learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** for exploring housing market data and predicting residential real estate prices using **Multiple Linear Regression**.

## 🎯 Overview

This project combines data science and web development to create an intuitive interface for real estate valuation. The app features a trained multiple linear regression model with comprehensive data preprocessing, interactive visualizations, and batch prediction capabilities.

---

## ✨ Key Features

### 🔮 **Interactive Valuation Estimator**
- Customize property features with an intuitive form interface
- Supported features: living area, lot size, bedrooms, bathrooms, floors, waterfront status, view rating, condition, construction/renovation year, city, and state/zip code
- **Instant valuation** calculation with:
  - Estimated market price
  - Price per square foot breakdown
  - Confidence interval (±MAE)
  - Comparison vs. city median and regional average

**Quick Presets:**
- 🏡 Suburban Family Home
- 🌊 Waterfront Luxury Villa  
- 🏙️ Modern Urban Condo
- 🔨 Starter / Fixer-Upper

### 📊 **Exploratory Data Analysis (EDA)**
Four interactive analysis tabs:
1. **💰 Price Distributions** - Histogram with log scale option, outlier filtering
2. **📈 Feature Relationships** - Scatter plots with OLS trendlines, color-coded by conditions
3. **🏙️ City & Geographic Analysis** - Box plots of top 15 cities with median/mean price tables
4. **🔥 Correlation Matrix** - Heatmap showing feature correlations

### 📈 **Model Performance & Diagnostics**
- **Performance Metrics Dashboard:**
  - Test R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Train/Test dataset split info
- **Visualizations:**
  - Actual vs. Predicted prices scatter plot with perfect prediction baseline
  - Residual error distribution histogram
  - Top 20 feature coefficients bar chart (shows feature impact on price)

### 📁 **Batch CSV Predictor**
- Upload CSV files with multiple property records
- Download sample template CSV
- Instant bulk predictions
- Results include all input features + predicted prices
- CSV download with results

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.30+ |
| **ML Engine** | Scikit-Learn 1.3+ |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ |
| **Visualizations** | Plotly 5.18+, Matplotlib 3.7+, Seaborn 0.12+ |
| **Language** | Python 3.8+ |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
```bash
git clone <repository-url>
cd areeb's\ work
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure data.csv is in the project directory**
The app looks for `data.csv` in the following order:
- Current directory
- Parent directory  
- Script directory

### Running the Application

Start the Streamlit local server:

```bash
streamlit run app.py
```

The web app will automatically open at:
- **Local:** `http://localhost:8501`
- **Network:** `http://<your-ip>:8501`

### Stopping the Application
Press `Ctrl+C` in your terminal to stop the server.

---

## 📂 Project Structure

```text
├── data.csv                                      # Housing dataset (required)
├── house_price_multiple_linear_regression.ipynb  # Original Jupyter notebook
├── app.py                                        # Main Streamlit application
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

---

## 📊 Model Details

### Algorithm: Multiple Linear Regression
- **Type:** Supervised Learning, Regression
- **Train/Test Split:** 80/20
- **Feature Preprocessing:**
  - Numeric features: Standard scaling (Z-score normalization)
  - Categorical features: One-Hot Encoding
  - Missing values: Imputation (median for numeric, most frequent for categorical)

### Performance Metrics
The model achieves strong performance on real estate valuation tasks:
- **R² Score:** Measures goodness-of-fit (1.0 = perfect, 0.0 = no correlation)
- **MAE:** Average absolute prediction error in dollars
- **RMSE:** Penalizes larger errors more heavily

### Feature Importance
The app displays the top 20 most impactful features:
- **Positive coefficients** → increase estimated price
- **Negative coefficients** → decrease estimated price
- **Magnitude** → relative importance (per standard deviation for scaled features)

---

## 📖 Usage Examples

### Single Property Valuation
1. Navigate to "🔮 Interactive Price Predictor" tab
2. Click a preset template or manually adjust values
3. Click "💰 Calculate Valuation"
4. View estimated price with confidence bounds and market comparisons

### Batch Processing
1. Navigate to "📁 Batch CSV Predictor" tab
2. Download the sample CSV template
3. Fill in your property data (must match column names)
4. Upload the CSV
5. Download predictions with estimated prices

### Data Exploration
1. Navigate to "📊 Exploratory Data Analysis" tab
2. Choose analysis type (Price Distribution, Features, Geographic, Correlation)
3. Interact with Plotly charts (hover, zoom, pan)
4. Toggle log scale or filter outliers as needed

---

## 🔧 Configuration

### Data Loading
The app automatically searches for `data.csv` in multiple locations. To use a custom dataset:

1. Name it `data.csv` and place in the project directory, OR
2. Modify the `file_path` parameter in `load_and_clean_data()` function (line ~101)

### Required CSV Columns
- `price` - Target variable (required)
- `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, etc. - Features
- `city`, `statezip` - Location fields (optional but recommended)
- `date` - Sales date (optional, auto-engineered to sale_year, sale_month)

---

## 🎨 UI/UX Features

- **Dark theme** with professional gradient cards
- **Responsive design** that works on desktop and tablet
- **Real-time calculations** with cached model for performance
- **Interactive Plotly charts** with zoom, pan, and hover tooltips
- **Currency formatting** for all price displays
- **Confidence intervals** for prediction uncertainty quantification

---

## 📝 Key Files Explained

### `app.py`
Main Streamlit application containing:
- Data loading & preprocessing functions
- Model training & caching pipeline
- Page configuration & custom CSS styling
- Four app modules (predictor, EDA, diagnostics, batch predictor)
- Navigation sidebar

### `house_price_multiple_linear_regression.ipynb`
Jupyter notebook showing:
- Exploratory data analysis process
- Feature engineering steps
- Model development and evaluation
- Visualization and insights

### `requirements.txt`
Python package dependencies with pinned versions for reproducibility.

### `data.csv`
Housing dataset containing:
- Property features (size, rooms, age, condition)
- Location information (city, state, zip)
- Sales data (date, price)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "data.csv not found" | Ensure `data.csv` is in the same directory as `app.py` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Slow predictions | Model is cached; first run caches the trained pipeline |
| CSV upload fails | Verify column names match dataset (case-sensitive) |
| Missing values in upload | The pipeline auto-imputes missing values |

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Visit [share.streamlit.io]([https://share.streamlit.io](http://localhost:8501/))
3. Connect GitHub repo → Deploy in seconds
4. App is live at public URL

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Local Server / VPS
```bash
# Install systemd service or use screen/tmux for persistent running
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📚 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-Learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#regression)
- [Plotly Python Visualization](https://plotly.com/python/)
- [Multiple Linear Regression Explained](https://en.wikipedia.org/wiki/Linear_regression)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

Built as part of Data Crumbs Bootcamp - Class 9

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and suggest features via issues
- Fork and submit pull requests
- Improve documentation

---

## 📞 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the [Streamlit documentation](https://docs.streamlit.io)
3. Examine the original Jupyter notebook for model insights

---

**Happy predicting! 🎯**
