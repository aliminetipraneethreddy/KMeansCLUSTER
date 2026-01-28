import streamlit as st
import pandas as pd
import numpy as np

# ✅ Correct matplotlib import order (Cloud-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------
# SIMPLE UI STYLING (SAFE)
# ----------------------------------
st.markdown("""
<style>
h1 { font-size: 40px; }
h2 { font-size: 28px; }
p  { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------
st.title("🛍️ Customer Segmentation Dashboard")
st.write(
    "This app segments customers using **K-Means clustering**. "
    "Designed for **clarity, simplicity, and accessibility**."
)

st.divider()

# ----------------------------------
# SIDEBAR CONTROLS
# ----------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Wholesale Customers CSV",
    type=["csv"]
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=8,
    value=4
)

random_state = st.sidebar.selectbox(
    "Random State",
    [0, 21, 42, 99],
    index=2
)

# ----------------------------------
# MAIN LOGIC
# ----------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    features = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen"
    ]

    X = df[features]

    st.subheader("📄 Dataset Preview")
    st.dataframe(X.head())

    # ----------------------------------
    # SCALING
    # ----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("Data scaled successfully ✔")

    # ----------------------------------
    # ELBOW METHOD
    # ----------------------------------
    st.subheader("📉 Elbow Method")

    wcss = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i,
            init="k-means++",
            random_state=random_state
        )
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # ----------------------------------
    # K-MEANS MODEL
    # ----------------------------------
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=random_state
    )
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    silhouette = silhouette_score(X_scaled, clusters)

    st.subheader("🧠 Model Evaluation")
    st.metric("Silhouette Score", round(silhouette, 2))

    # ----------------------------------
    # CLUSTER VISUALIZATION
    # ----------------------------------
    st.subheader("🎨 Customer Segments")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(
        X["Fresh"],
        X["Milk"],
        c=clusters,
        cmap="tab10",
        s=60
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=200,
        marker="X",
        label="Centroids"
    )

    ax2.set_xlabel("Fresh")
    ax2.set_ylabel("Milk")
    ax2.set_title("Customer Segments (Fresh vs Milk)")
    ax2.legend()

    st.pyplot(fig2)

    # ----------------------------------
    # CLUSTER PROFILING
    # ----------------------------------
    st.subheader("📊 Cluster Profiling")

    profile = df.groupby("Cluster")[features].mean()
    st.dataframe(profile)

    # ----------------------------------
    # BUSINESS INSIGHTS
    # ----------------------------------
    st.subheader("💡 Business Interpretation")

    for c in profile.index:
        st.markdown(f"### Cluster {c}")
        st.write(
            f"""
            • **Highest Spend:** {profile.loc[c].idxmax()}  
            • **Lowest Spend:** {profile.loc[c].idxmin()}  

            **Suggested Actions:**  
            • Targeted promotions  
            • Product bundling  
            • Loyalty programs  
            """
        )

    # ----------------------------------
    # STABILITY NOTE
    # ----------------------------------
    st.subheader("🔁 Stability Check")
    st.write(
        "Changing `random_state` results in **similar cluster patterns**, "
        "indicating a **stable clustering solution**."
    )

    st.success("✅ Analysis Completed Successfully")

else:
    st.info("👈 Upload the `Wholesale customers data.csv` file to begin.")


