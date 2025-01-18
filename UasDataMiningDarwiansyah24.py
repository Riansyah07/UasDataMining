import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('Regression.csv')
    except FileNotFoundError:
        st.error("File 'Regression.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi ini.")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika file tidak ditemukan

# Fungsi untuk preprocess data
def preprocess_data(data):
    return pd.get_dummies(data, drop_first=True)

# Inisialisasi session state
if "model" not in st.session_state:
    st.session_state["model"] = None
if "columns" not in st.session_state:
    st.session_state["columns"] = None

# Tampilan Header
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        color: #4CAF50;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #6C757D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Regression Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore, visualize, and build regression models interactively 📊</div>', unsafe_allow_html=True)

# Load data
data = load_data()
if data.empty:
    st.stop()  # Hentikan aplikasi jika data kosong

# Tabs for navigation
tabs = st.tabs(["📄 View Data", "📈 Visualize Data", "🤖 Train Model", "🔮 Make Prediction"])

# Tab 1: View Data
with tabs[0]:
    st.header("📄 View Data")
    with st.expander("🔍 Dataset Preview"):
        st.dataframe(data.head(10))
    with st.expander("📊 Descriptive Statistics"):
        st.write(data.describe())

# Tab 2: Visualize Data
with tabs[1]:
    st.header("📈 Visualize Data")
    st.markdown("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=["float64", "int64"])  # Hanya kolom numerik

    if numeric_data.empty or numeric_data.shape[1] < 2:
        st.warning("Tidak cukup kolom numerik untuk membuat heatmap.")
    else:
        # Tangani NaN jika ada
        numeric_data = numeric_data.dropna()
        corr = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    if "charges" in data.columns:
        st.markdown("### Distribution of Target (Charges)")
        fig, ax = plt.subplots()
        sns.histplot(data["charges"].dropna(), kde=True, bins=30, color="#FF6F61", ax=ax)
        ax.set_title("Distribution of Charges")
        st.pyplot(fig)

# Tab 3: Train Model
with tabs[2]:
    st.header("🤖 Train Model")
    if "charges" in data.columns:
        st.markdown("#### Select a regression model to train:")
        data = preprocess_data(data)
        X = data.drop("charges", axis=1)
        y = data["charges"]

        # Validasi apakah X atau y kosong
        if X.empty or y.empty:
            st.error("Data untuk training tidak valid. Pastikan dataset memiliki data numerik dan target (charges).")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Linear Regression**")
                lr = LinearRegression().fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                st.write(f"Mean Squared Error: `{mean_squared_error(y_test, lr_pred):.2f}`")
                st.write(f"R-squared: `{r2_score(y_test, lr_pred):.2f}`")

            with col2:
                st.markdown("**Random Forest Regressor**")
                rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                st.write(f"Mean Squared Error: `{mean_squared_error(y_test, rf_pred):.2f}`")
                st.write(f"R-squared: `{r2_score(y_test, rf_pred):.2f}`")

            with col3:
                st.markdown("**Decision Tree Regressor**")
                dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
                dt_pred = dt.predict(X_test)
                st.write(f"Mean Squared Error: `{mean_squared_error(y_test, dt_pred):.2f}`")
                st.write(f"R-squared: `{r2_score(y_test, dt_pred):.2f}`")

            # Simpan model ke session state
            st.session_state["model"] = dt
            st.session_state["columns"] = X.columns
    else:
        st.error("Kolom target 'charges' tidak ditemukan dalam dataset.")

# Tab 4: Make Prediction
with tabs[3]:
    st.header("🔮 Make Prediction")
    if st.session_state["model"] is not None and st.session_state["columns"] is not None:
        model = st.session_state["model"]
        columns = st.session_state["columns"]

        st.markdown("### Input Features")
        input_data = {}
        for col in columns:
            if col in data.columns:
                if data[col].dtype in ["float64", "int64"]:
                    input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)
                else:
                    input_data[col] = st.selectbox(f"Select value for {col}", data[col].unique())
            else:
                input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=columns, fill_value=0)
            prediction = model.predict(input_df)
            st.success(f"🎉 Predicted Charges: **${prediction[0]:.2f}**")
    else:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
