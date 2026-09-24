import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 17px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1b1e27;
    border: 1px solid #30333d;
    margin-bottom: 15px;
}

.card-title {
    font-size: 15px;
    color: #aaaaaa;
}

.card-value {
    font-size: 30px;
    font-weight: bold;
}

.warning {
    padding: 15px;
    border-radius: 10px;
    background-color: #3a2815;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🔍 Anomaly Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Transaction Monitoring & Risk Analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GENERATE SAMPLE DATA
# =========================================================

np.random.seed(42)

normal_transactions = 950
anomaly_transactions = 50


# -----------------------------
# Normal transactions
# -----------------------------

normal_amount = np.random.normal(
    loc=2500,
    scale=1000,
    size=normal_transactions
)

normal_amount = np.clip(
    normal_amount,
    200,
    7000
)

normal_duration = np.random.normal(
    loc=8,
    scale=2,
    size=normal_transactions
)

normal_duration = np.clip(
    normal_duration,
    2,
    18
)

normal_frequency = np.random.poisson(
    lam=4,
    size=normal_transactions
)

normal_distance = np.random.normal(
    loc=15,
    scale=8,
    size=normal_transactions
)

normal_distance = np.clip(
    normal_distance,
    1,
    50
)


# -----------------------------
# Anomalous transactions
# -----------------------------

anomaly_amount = np.random.uniform(
    15000,
    50000,
    anomaly_transactions
)

anomaly_duration = np.random.uniform(
    25,
    60,
    anomaly_transactions
)

anomaly_frequency = np.random.randint(
    15,
    40,
    anomaly_transactions
)

anomaly_distance = np.random.uniform(
    100,
    500,
    anomaly_transactions
)


# =========================================================
# CREATE DATAFRAME
# =========================================================

normal_df = pd.DataFrame({
    "Transaction_Amount": normal_amount,
    "Transaction_Duration": normal_duration,
    "Transaction_Frequency": normal_frequency,
    "Distance_From_Home": normal_distance
})

anomaly_df = pd.DataFrame({
    "Transaction_Amount": anomaly_amount,
    "Transaction_Duration": anomaly_duration,
    "Transaction_Frequency": anomaly_frequency,
    "Distance_From_Home": anomaly_distance
})


df = pd.concat(
    [normal_df, anomaly_df],
    ignore_index=True
)


# =========================================================
# ADD TRANSACTION INFORMATION
# =========================================================

df.insert(
    0,
    "Transaction_ID",
    [
        f"TXN-{100001 + i}"
        for i in range(len(df))
    ]
)


np.random.seed(42)

names = [
    "Ali Khan",
    "Ahmed Raza",
    "Sara Ahmed",
    "Ayesha Malik",
    "Usman Tariq",
    "Hassan Ali",
    "Fatima Noor",
    "Hamza Sheikh",
    "Zainab Khan",
    "Bilal Ahmed"
]

locations = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Multan",
    "Faisalabad",
    "Rawalpindi",
    "Peshawar",
    "Quetta"
]

df["Customer"] = np.random.choice(
    names,
    len(df)
)

df["Location"] = np.random.choice(
    locations,
    len(df)
)

df["Transaction_Type"] = np.random.choice(
    [
        "Online Payment",
        "ATM Withdrawal",
        "Card Payment",
        "Bank Transfer",
        "Mobile Payment"
    ],
    len(df)
)


# =========================================================
# RANDOMIZE DATA
# =========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================================================
# MACHINE LEARNING FEATURES
# =========================================================

features = [
    "Transaction_Amount",
    "Transaction_Duration",
    "Transaction_Frequency",
    "Distance_From_Home"
]


X = df[features]


# =========================================================
# STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# ISOLATION FOREST MODEL
# =========================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


predictions = model.fit_predict(
    X_scaled
)


# =========================================================
# ANOMALY SCORES
# =========================================================

raw_scores = model.decision_function(
    X_scaled
)

df["Anomaly_Score"] = raw_scores


# Convert prediction
df["Prediction"] = np.where(
    predictions == -1,
    "Anomaly",
    "Normal"
)


# =========================================================
# RISK LEVEL
# =========================================================

def risk_level(score):

    if score < -0.15:
        return "Critical"

    elif score < -0.05:
        return "High"

    elif score < 0.05:
        return "Medium"

    else:
        return "Low"


df["Risk_Level"] = df["Anomaly_Score"].apply(
    risk_level
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Detection Controls")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    ["Low", "Medium", "High", "Critical"],
    default=["Low", "Medium", "High", "Critical"]
)


type_filter = st.sidebar.multiselect(
    "Transaction Type",
    sorted(df["Transaction_Type"].unique()),
    default=sorted(df["Transaction_Type"].unique())
)


location_filter = st.sidebar.multiselect(
    "Location",
    sorted(df["Location"].unique()),
    default=sorted(df["Location"].unique())
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["Risk_Level"].isin(risk_filter))
    &
    (df["Transaction_Type"].isin(type_filter))
    &
    (df["Location"].isin(location_filter))
]


# =========================================================
# DASHBOARD METRICS
# =========================================================

total_transactions = len(df)

total_anomalies = len(
    df[df["Prediction"] == "Anomaly"]
)

total_normal = len(
    df[df["Prediction"] == "Normal"]
)

anomaly_percentage = (
    total_anomalies / total_transactions
) * 100


# =========================================================
# METRIC CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">Total Transactions</div>
        <div class="card-value">{total_transactions:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">Normal Transactions</div>
        <div class="card-value">{total_normal:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">Anomalies Detected</div>
        <div class="card-value">{total_anomalies:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">Anomaly Rate</div>
        <div class="card-value">{anomaly_percentage:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# ALERT
# =========================================================

if total_anomalies > 0:

    st.warning(
        f"⚠️ Machine Learning model detected "
        f"{total_anomalies} potentially anomalous transactions."
    )


# =========================================================
# CHART SECTION
# =========================================================

st.subheader("📊 Transaction Analysis")


chart_col1, chart_col2 = st.columns(2)


# =========================================================
# CHART 1 - NORMAL VS ANOMALY
# =========================================================

with chart_col1:

    st.write("### Normal vs Anomaly")

    status_counts = df["Prediction"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        status_counts.index,
        status_counts.values
    )

    ax.set_xlabel("Transaction Status")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Classification")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# CHART 2 - RISK LEVEL
# =========================================================

with chart_col2:

    st.write("### Risk Level Distribution")

    risk_counts = df["Risk_Level"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        risk_counts.index,
        risk_counts.values
    )

    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Risk Distribution")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# AMOUNT ANALYSIS
# =========================================================

st.subheader("💰 Transaction Amount Analysis")


fig, ax = plt.subplots()

normal_amounts = df[
    df["Prediction"] == "Normal"
]["Transaction_Amount"]

anomaly_amounts = df[
    df["Prediction"] == "Anomaly"
]["Transaction_Amount"]

ax.hist(
    normal_amounts,
    bins=30,
    alpha=0.7,
    label="Normal"
)

ax.hist(
    anomaly_amounts,
    bins=20,
    alpha=0.7,
    label="Anomaly"
)

ax.set_xlabel("Transaction Amount")
ax.set_ylabel("Frequency")
ax.set_title("Transaction Amount Distribution")

ax.legend()

st.pyplot(fig)

plt.close(fig)


# =========================================================
# ANOMALY SCORE ANALYSIS
# =========================================================

st.subheader("📈 Anomaly Score Distribution")


fig, ax = plt.subplots()

ax.hist(
    df["Anomaly_Score"],
    bins=40
)

ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Frequency")
ax.set_title("Isolation Forest Anomaly Score")

st.pyplot(fig)

plt.close(fig)


# =========================================================
# ANOMALY TRANSACTIONS
# =========================================================

st.subheader("🚨 Detected Anomalies")


anomalies = filtered_df[
    filtered_df["Prediction"] == "Anomaly"
].copy()


if len(anomalies) > 0:

    display_anomalies = anomalies[
        [
            "Transaction_ID",
            "Customer",
            "Location",
            "Transaction_Type",
            "Transaction_Amount",
            "Transaction_Duration",
            "Transaction_Frequency",
            "Distance_From_Home",
            "Anomaly_Score",
            "Risk_Level"
        ]
    ].sort_values(
        by="Anomaly_Score"
    )

    st.dataframe(
        display_anomalies,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No anomalies match the selected filters."
    )


# =========================================================
# HIGH-RISK TRANSACTIONS
# =========================================================

st.subheader("🔴 High-Risk Transactions")


high_risk = filtered_df[
    filtered_df["Risk_Level"].isin(
        ["High", "Critical"]
    )
].copy()


if len(high_risk) > 0:

    st.dataframe(
        high_risk[
            [
                "Transaction_ID",
                "Customer",
                "Location",
                "Transaction_Type",
                "Transaction_Amount",
                "Risk_Level",
                "Anomaly_Score"
            ]
        ].sort_values(
            by="Anomaly_Score"
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No high-risk transactions found."
    )


# =========================================================
# TOP ANOMALOUS TRANSACTIONS
# =========================================================

st.subheader("🏆 Most Suspicious Transactions")


top_anomalies = df.sort_values(
    by="Anomaly_Score"
).head(10)


for _, row in top_anomalies.iterrows():

    st.markdown(
        f"""
        <div class="card">

        <b>Transaction:</b>
        {row['Transaction_ID']}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Customer:</b>
        {row['Customer']}

        <br><br>

        <b>Amount:</b>
        Rs. {row['Transaction_Amount']:,.2f}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Location:</b>
        {row['Location']}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Risk:</b>
        {row['Risk_Level']}

        <br><br>

        <b>Anomaly Score:</b>
        {row['Anomaly_Score']:.4f}

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 About the Machine Learning Model"):

    st.write("""
    ### Isolation Forest

    This project uses the **Isolation Forest** machine learning
    algorithm for unsupervised anomaly detection.

    The model identifies transactions that behave differently
    from the majority of transactions.

    ### Features Used

    - Transaction Amount
    - Transaction Duration
    - Transaction Frequency
    - Distance From Home

    ### How it works

    Isolation Forest randomly splits the data and isolates unusual
    observations. Transactions that are easier to isolate are more
    likely to be anomalies.

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib
    """)


# =========================================================
# DOWNLOAD RESULTS
# =========================================================

st.subheader("📥 Export Results")


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Complete Detection Report",
    data=csv_data,
    file_name="anomaly_detection_report.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>

    🔍 <b>Anomaly Detection System</b><br>

    Machine Learning • Isolation Forest • Streamlit

    </center>
    """,
    unsafe_allow_html=True)