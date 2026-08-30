import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Real Estate House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-card h3 {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #60a5fa;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .metric-card .subtitle {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 5px;
    }

    /* Prediction Banner */
    .prediction-box {
        background: linear-gradient(135deg, #1e3a8a, #0f172a);
        border: 1px solid #3b82f6;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
    }
    .prediction-box h2 {
        color: #93c5fd;
        margin-bottom: 5px;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .prediction-box .price-display {
        font-size: 3.2rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
    }
    .prediction-box .subtext {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    
    /* Preset Button Container */
    .preset-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path="data.csv"):
    """
    Loads raw housing dataset and prepares features and target.
    """
    if not os.path.exists(file_path):
        # Check parent or current directory
        alt_paths = ["data.csv", "data(1).csv", os.path.join(os.path.dirname(__file__), "data.csv")]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                file_path = p
                found = True
                break
        if not found:
            return None, None, None, None, None, None

    df = pd.read_csv(file_path)
    clean_df = df.copy()

    # Engineer date features
    if "date" in clean_df.columns:
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
        clean_df["sale_year"] = clean_df["date"].dt.year
        clean_df["sale_month"] = clean_df["date"].dt.month
        clean_df.drop(columns=["date"], inplace=True)

    # Drop identifier & high-cardinality columns
    drop_cols = [c for c in ["street", "country"] if c in clean_df.columns]
    clean_df.drop(columns=drop_cols, inplace=True)

    # Target cleaning
    clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce")
    clean_df = clean_df.dropna(subset=["price"])

    # Separate target & features
    X = clean_df.drop(columns=["price"])
    y = clean_df["price"]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, clean_df, X, y, numeric_features, categorical_features


# ==============================================================================
# 3. MODEL TRAINING & PIPELINE CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def train_pipeline(X, y, numeric_features, categorical_features):
    """
    Trains the scikit-learn preprocessing & Multiple Linear Regression pipeline.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model_pipeline.fit(X_train, y_train)

    # Predictions & metrics
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)

    train_metrics = {
        "mae": mean_absolute_error(y_train, y_train_pred),
        "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "r2": r2_score(y_train, y_train_pred)
    }

    test_metrics = {
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "r2": r2_score(y_test, y_test_pred)
    }

    # Extract coefficients
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = model_pipeline.named_steps["regressor"].coef_

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    return {
        "pipeline": model_pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coef_df": coef_df
    }


# ==============================================================================
# 4. INITIALIZE DATA & MODEL
# ==============================================================================
raw_df, clean_df, X, y, numeric_cols, categorical_cols = load_and_clean_data()

if clean_df is None:
    st.error("⚠️ `data.csv` was not found in the project directory. Please ensure `data.csv` exists in the same folder.")
    st.stop()

train_results = train_pipeline(X, y, numeric_cols, categorical_cols)
model_pipeline = train_results["pipeline"]


# ==============================================================================
# 5. SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================
with st.sidebar:
    st.title("🏡 House Price AI")
    st.markdown("Multiple Linear Regression Model & Analytics")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        [
            "🔮 Interactive Price Predictor",
            "📊 Exploratory Data Analysis",
            "📈 Model Performance & Coefficients",
            "📁 Batch CSV Predictor"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📌 Model Summary")
    st.markdown(f"- **Algorithm**: Multiple Linear Regression")
    st.markdown(f"- **Total Properties**: `{len(clean_df):,}`")
    st.markdown(f"- **Test R² Score**: `{train_results['test_metrics']['r2']:.4f}`")
    st.markdown(f"- **Test MAE**: `${train_results['test_metrics']['mae']:,.0f}`")
    st.markdown(f"- **Test RMSE**: `${train_results['test_metrics']['rmse']:,.0f}`")

    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn")


# ==============================================================================
# 6. APP MODULE 1: INTERACTIVE PRICE PREDICTOR
# ==============================================================================
if app_mode == "🔮 Interactive Price Predictor":
    st.title("🔮 House Price Valuation Estimator")
    st.markdown(
        "Adjust the property characteristics below or select a preset template to estimate the market valuation using our trained Multiple Linear Regression model."
    )

    # Preset Templates
    st.markdown("<p class='preset-title'>⚡ Quick Presets (Click to Auto-fill):</p>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_name = None
    if col_p1.button("🏡 Suburban Family Home", use_container_width=True):
        preset_name = "suburban"
    if col_p2.button("🌊 Waterfront Luxury Villa", use_container_width=True):
        preset_name = "luxury"
    if col_p3.button("🏙️ Modern Urban Condo", use_container_width=True):
        preset_name = "urban"
    if col_p4.button("🔨 Starter / Fixer-Upper", use_container_width=True):
        preset_name = "starter"

    # Default values dictionary
    defaults = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1500,
        "sqft_basement": 300,
        "yr_built": 1995,
        "yr_renovated": 0,
        "city": "Seattle",
        "statezip": "WA 98115",
        "sale_year": 2014,
        "sale_month": 6
    }

    if preset_name == "suburban":
        defaults.update({
            "bedrooms": 4, "bathrooms": 2.5, "sqft_living": 2600, "sqft_lot": 8500,
            "floors": 2.0, "waterfront": 0, "view": 1, "condition": 4,
            "sqft_above": 2200, "sqft_basement": 400, "yr_built": 2005,
            "yr_renovated": 0, "city": "Redmond", "statezip": "WA 98052"
        })
    elif preset_name == "luxury":
        defaults.update({
            "bedrooms": 5, "bathrooms": 4.5, "sqft_living": 5200, "sqft_lot": 20000,
            "floors": 2.5, "waterfront": 1, "view": 4, "condition": 5,
            "sqft_above": 4200, "sqft_basement": 1000, "yr_built": 2012,
            "yr_renovated": 2014, "city": "Bellevue", "statezip": "WA 98004"
        })
    elif preset_name == "urban":
        defaults.update({
            "bedrooms": 2, "bathrooms": 1.5, "sqft_living": 1100, "sqft_lot": 2200,
            "floors": 2.0, "waterfront": 0, "view": 2, "condition": 4,
            "sqft_above": 1100, "sqft_basement": 0, "yr_built": 2010,
            "yr_renovated": 0, "city": "Seattle", "statezip": "WA 98101"
        })
    elif preset_name == "starter":
        defaults.update({
            "bedrooms": 2, "bathrooms": 1.0, "sqft_living": 950, "sqft_lot": 4500,
            "floors": 1.0, "waterfront": 0, "view": 0, "condition": 2,
            "sqft_above": 950, "sqft_basement": 0, "yr_built": 1955,
            "yr_renovated": 0, "city": "Kent", "statezip": "WA 98042"
        })

    # Available cities and statezips from dataset
    city_options = sorted(clean_df["city"].dropna().unique().tolist()) if "city" in clean_df.columns else ["Seattle"]
    statezip_options = sorted(clean_df["statezip"].dropna().unique().tolist()) if "statezip" in clean_df.columns else ["WA 98103"]

    city_default_idx = city_options.index(defaults["city"]) if defaults["city"] in city_options else 0
    statezip_default_idx = statezip_options.index(defaults["statezip"]) if defaults["statezip"] in statezip_options else 0

    st.markdown("---")
    # Form input layout
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📐 Living Dimensions")
            sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=15000, value=defaults["sqft_living"], step=50)
            sqft_above = st.number_input("Above Ground Area (sqft)", min_value=200, max_value=15000, value=defaults["sqft_above"], step=50)
            sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=6000, value=defaults["sqft_basement"], step=50)
            sqft_lot = st.number_input("Lot Size (sqft)", min_value=400, max_value=1500000, value=defaults["sqft_lot"], step=250)

        with col2:
            st.subheader("🛏️ Rooms & Structure")
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=int(defaults["bedrooms"]), step=1)
            bathrooms = st.slider("Bathrooms", min_value=0.75, max_value=8.0, value=float(defaults["bathrooms"]), step=0.25)
            floors = st.selectbox("Floors / Stories", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5].index(defaults["floors"]) if defaults["floors"] in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] else 0)
            condition = st.slider("Condition Rating (1 = Poor, 5 = Pristine)", min_value=1, max_value=5, value=int(defaults["condition"]), step=1)

        with col3:
            st.subheader("📍 Location & Features")
            city = st.selectbox("City", options=city_options, index=city_default_idx)
            statezip = st.selectbox("State & Zip Code", options=statezip_options, index=statezip_default_idx)
            waterfront = st.selectbox("Waterfront Property?", options=[0, 1], format_func=lambda x: "Yes 🌊" if x == 1 else "No 🚫", index=defaults["waterfront"])
            view = st.slider("View Quality Score (0 = None, 4 = Panoramic)", min_value=0, max_value=4, value=int(defaults["view"]), step=1)

        st.markdown("#### 📅 Age & Renovation")
        col_age1, col_age2, col_age3, col_age4 = st.columns(4)
        with col_age1:
            yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=int(defaults["yr_built"]), step=1)
        with col_age2:
            is_renovated = st.checkbox("Has been renovated?", value=defaults["yr_renovated"] > 0)
            if is_renovated:
                yr_renovated = st.number_input("Year Renovated", min_value=1950, max_value=2026, value=max(defaults["yr_renovated"], 2000), step=1)
            else:
                yr_renovated = 0
        with col_age3:
            sale_year = st.selectbox("Sale Year", options=[2014, 2015, 2024, 2025, 2026], index=0)
        with col_age4:
            sale_month = st.slider("Sale Month", min_value=1, max_value=12, value=int(defaults["sale_month"]))

        submit_btn = st.form_submit_button("💰 Calculate Valuation", use_container_width=True, type="primary")

    # Prediction Calculation
    input_row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "city": city,
        "statezip": statezip,
        "sale_year": sale_year,
        "sale_month": sale_month
    }

    input_df = pd.DataFrame([input_row])
    # Ensure all model columns are present
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[X.columns]

    predicted_price = model_pipeline.predict(input_df)[0]
    predicted_price = max(0.0, float(predicted_price))

    price_per_sqft = predicted_price / sqft_living if sqft_living > 0 else 0
    mae = train_results["test_metrics"]["mae"]

    # Display Valuation Banner
    st.markdown(f"""
    <div class="prediction-box">
        <h2>ESTIMATED MARKET VALUE</h2>
        <div class="price-display">${predicted_price:,.0f}</div>
        <div class="subtext">
            Confidence Interval: <strong>${max(0, predicted_price - mae):,.0f}</strong> – <strong>${predicted_price + mae:,.0f}</strong> 
            (±${mae:,.0f} Test MAE)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Price per Sq. Ft.</h3>
            <div class="value">${price_per_sqft:.1f}</div>
            <div class="subtitle">Based on {sqft_living:,} sqft</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        city_median = clean_df[clean_df["city"] == city]["price"].median() if city in clean_df["city"].values else clean_df["price"].median()
        diff_city = ((predicted_price - city_median) / city_median) * 100 if city_median > 0 else 0
        diff_str = f"+{diff_city:.1f}%" if diff_city >= 0 else f"{diff_city:.1f}%"
        st.markdown(f"""
        <div class="metric-card">
            <h3>City Median Price</h3>
            <div class="value">${city_median:,.0f}</div>
            <div class="subtitle">{diff_str} vs. {city} median</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        overall_median = clean_df["price"].median()
        diff_ov = ((predicted_price - overall_median) / overall_median) * 100 if overall_median > 0 else 0
        diff_ov_str = f"+{diff_ov:.1f}%" if diff_ov >= 0 else f"{diff_ov:.1f}%"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Market Median</h3>
            <div class="value">${overall_median:,.0f}</div>
            <div class="subtitle">{diff_ov_str} vs. all regional sales</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        age = sale_year - yr_built
        st.markdown(f"""
        <div class="metric-card">
            <h3>Property Age</h3>
            <div class="value">{age} yrs</div>
            <div class="subtitle">Built in {yr_built}</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 7. APP MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif app_mode == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis & Market Trends")
    st.markdown("Interact with distributions, geographic price differences, and feature correlations across the dataset.")

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "💰 Price Distributions",
        "📈 Feature Relationships",
        "🏙️ City & Geographic Analysis",
        "🔥 Correlation Matrix"
    ])

    with eda_tab1:
        st.subheader("Price Distribution & Outliers")
        col_t1, col_t2 = st.columns([3, 1])
        with col_t2:
            log_scale = st.checkbox("Logarithmic Price Scale", value=False)
            filter_outliers = st.checkbox("Exclude Extreme Outliers (Top 1%)", value=True)

        plot_df = clean_df.copy()
        if filter_outliers:
            p99 = plot_df["price"].quantile(0.99)
            plot_df = plot_df[plot_df["price"] <= p99]

        fig_hist = px.histogram(
            plot_df,
            x="price",
            nbins=60,
            log_x=log_scale,
            title="House Price Frequency Distribution",
            color_discrete_sequence=["#38bdf8"],
            marginal="box"
        )
        fig_hist.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Count",
            template="plotly_dark",
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown(f"**Dataset Summary**: Median Price = **${clean_df['price'].median():,.0f}** | Mean Price = **${clean_df['price'].mean():,.0f}** | Max Price = **${clean_df['price'].max():,.0f}**")

    with eda_tab2:
        st.subheader("House Price vs. Key Numerical Attributes")
        col_x, col_c = st.columns(2)
        with col_x:
            scatter_x = st.selectbox(
                "Select X-axis Feature",
                options=["sqft_living", "sqft_above", "sqft_lot", "bathrooms", "bedrooms", "yr_built", "condition"],
                index=0
            )
        with col_c:
            color_by = st.selectbox(
                "Color Points By",
                options=["waterfront", "view", "condition", "floors"],
                index=1
            )

        scatter_df = clean_df.sample(min(1500, len(clean_df)), random_state=42)
        fig_scatter = px.scatter(
            scatter_df,
            x=scatter_x,
            y="price",
            color=color_by,
            hover_data=["city", "bedrooms", "bathrooms"],
            trendline="ols",
            title=f"Price vs. {scatter_x.replace('_', ' ').title()}",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_scatter.update_layout(yaxis_title="Price ($)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with eda_tab3:
        st.subheader("Price by City")
        # Top 15 cities by listing count
        top_cities = clean_df["city"].value_counts().nlargest(15).index.tolist()
        city_df = clean_df[clean_df["city"].isin(top_cities)]

        fig_city = px.box(
            city_df,
            x="city",
            y="price",
            color="city",
            title="Price Distribution Across Top 15 Cities",
            template="plotly_dark"
        )
        fig_city.update_layout(xaxis_title="City", yaxis_title="Price ($)", showlegend=False)
        st.plotly_chart(fig_city, use_container_width=True)

        # Median Price Table
        median_city_df = city_df.groupby("city")["price"].agg(["median", "mean", "count"]).reset_index()
        median_city_df.columns = ["City", "Median Price ($)", "Mean Price ($)", "Total Listings"]
        median_city_df = median_city_df.sort_values("Median Price ($)", ascending=False)
        st.dataframe(median_city_df.style.format({
            "Median Price ($)": "${:,.0f}",
            "Mean Price ($)": "${:,.0f}",
            "Total Listings": "{:,}"
        }), use_container_width=True)

    with eda_tab4:
        st.subheader("Numeric Feature Correlation Matrix")
        corr_matrix = clean_df.select_dtypes(include=np.number).corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
            template="plotly_dark"
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# 8. APP MODULE 3: MODEL PERFORMANCE & COEFFICIENTS
# ==============================================================================
elif app_mode == "📈 Model Performance & Coefficients":
    st.title("📈 Model Evaluation & Diagnostic Insights")
    st.markdown("Detailed breakdown of Multiple Linear Regression performance metrics, residual diagnostics, and feature coefficients.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Test R² Score</h3>
            <div class="value">{train_results['test_metrics']['r2']:.3f}</div>
            <div class="subtitle">Train R²: {train_results['train_metrics']['r2']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Test MAE</h3>
            <div class="value">${train_results['test_metrics']['mae']:,.0f}</div>
            <div class="subtitle">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Test RMSE</h3>
            <div class="value">${train_results['test_metrics']['rmse']:,.0f}</div>
            <div class="subtitle">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Dataset Split</h3>
            <div class="value">80 / 20</div>
            <div class="subtitle">{len(train_results['X_train']):,} Train / {len(train_results['X_test']):,} Test</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.subheader("🎯 Actual vs. Predicted Prices")
        y_test = train_results["y_test"]
        y_test_pred = train_results["y_test_pred"]

        min_val = min(y_test.min(), y_test_pred.min())
        max_val = min(3000000, max(y_test.max(), y_test_pred.max()))

        fig_act_pred = go.Figure()
        fig_act_pred.add_trace(go.Scatter(
            x=y_test,
            y=y_test_pred,
            mode='markers',
            marker=dict(color='#38bdf8', opacity=0.5, size=6),
            name='Test Samples'
        ))
        fig_act_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#f87171', dash='dash', width=2),
            name='Perfect Prediction (y=x)'
        ))
        fig_act_pred.update_layout(
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_act_pred, use_container_width=True)

    with col_eval2:
        st.subheader("📉 Residual Error Distribution")
        residuals = y_test - y_test_pred
        fig_res = px.histogram(
            x=residuals,
            nbins=50,
            title="Residuals (Actual - Predicted)",
            color_discrete_sequence=["#a78bfa"],
            template="plotly_dark"
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig_res.update_layout(xaxis_title="Residual ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature Coefficients
    st.subheader("⚖️ Top 20 Feature Coefficients (Standardized Scale)")
    st.markdown(
        "Positive coefficients increase the estimated house price, while negative coefficients decrease it. "
        "Because numerical features are standardized ($Z$-score scaled), their relative magnitude reflects their impact per standard deviation."
    )

    top_coef = train_results["coef_df"].head(20).sort_values("coefficient", ascending=True)
    # Color positive green/blue and negative red
    top_coef["color"] = np.where(top_coef["coefficient"] >= 0, "#34d399", "#f87171")

    fig_coef = go.Figure(go.Bar(
        x=top_coef["coefficient"],
        y=top_coef["feature"].str.replace("num__", "").str.replace("cat__", ""),
        orientation='h',
        marker=dict(color=top_coef["color"])
    ))
    fig_coef.update_layout(
        xaxis_title="Coefficient Value ($)",
        yaxis_title="Feature Name",
        template="plotly_dark",
        height=550
    )
    st.plotly_chart(fig_coef, use_container_width=True)


# ==============================================================================
# 9. APP MODULE 4: BATCH PREDICTOR
# ==============================================================================
elif app_mode == "📁 Batch CSV Predictor":
    st.title("📁 Bulk House Price Predictor")
    st.markdown(
        "Upload a `.csv` file containing multiple property records to run instant valuations in bulk and download the results."
    )

    # Download Template Option
    sample_df = clean_df.drop(columns=["price"]).head(5)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download CSV Sample Template",
        data=csv_buffer.getvalue(),
        file_name="sample_house_listings.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV file with house features", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} records!")

            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Preprocess date if present
            proc_batch = batch_df.copy()
            if "date" in proc_batch.columns:
                proc_batch["date"] = pd.to_datetime(proc_batch["date"], errors="coerce")
                proc_batch["sale_year"] = proc_batch["date"].dt.year
                proc_batch["sale_month"] = proc_batch["date"].dt.month
                proc_batch.drop(columns=["date"], inplace=True)

            # Align columns
            for col in X.columns:
                if col not in proc_batch.columns:
                    proc_batch[col] = np.nan
            proc_batch = proc_batch[X.columns]

            # Run Predictions
            batch_preds = model_pipeline.predict(proc_batch)
            batch_preds = np.maximum(0, batch_preds)

            results_df = batch_df.copy()
            results_df.insert(0, "predicted_price", batch_preds.round(2))

            st.subheader("🎯 Batch Predictions Summary")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Total Predictions", f"{len(results_df):,}")
            with col_b2:
                st.metric("Mean Valuation", f"${results_df['predicted_price'].mean():,.0f}")
            with col_b3:
                st.metric("Median Valuation", f"${results_df['predicted_price'].median():,.0f}")

            st.dataframe(results_df.style.format({"predicted_price": "${:,.2f}"}), use_container_width=True)

            # Download Predictions Button
            out_buffer = io.StringIO()
            results_df.to_csv(out_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=out_buffer.getvalue(),
                file_name="predicted_house_prices.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
