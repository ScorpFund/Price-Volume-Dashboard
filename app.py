import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title("📊 Price-Volume Behavior Explorer (10-Year Clustering)")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Load data
def load_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    return df

df = load_price_data(ticker, start_date, end_date)
st.write(f"Loaded {len(df)} days of data for **{ticker}**")

# Feature Engineering
X = df[["Return", "Volume"]].copy()
X["Volume"] = X["Volume"] / 1e6  # Convert to millions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

# Assign Labels based on return & volume logic
centroids = kmeans.cluster_centers_
labels = []
for r, v in centroids:
    if r >= 0 and v >= 0:
        labels.append("High Return / High Volume")
    elif r < 0 and v >= 0:
        labels.append("Low Return / High Volume")
    elif r >= 0 and v < 0:
        labels.append("High Return / Low Volume")
    else:
        labels.append("Low Return / Low Volume")

label_map = {i: labels[i] for i in range(4)}
df["Behavior"] = df["Cluster"].map(label_map)

# 3D Embedding
umap = UMAP(n_components=3, random_state=42)
embedding = umap.fit_transform(X_scaled)
df[["x", "y", "z"]] = embedding

# 3D Plot
st.subheader("🧭 3D Price-Volume Embedding")
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=df["x"],
    y=df["y"],
    z=df["z"],
    mode='markers',
    marker=dict(
        size=4,
        color=df["Cluster"],
        colorscale="Viridis",
        opacity=0.8,
        colorbar=dict(title="Cluster")
    ),
    text=df.apply(lambda row: f"Date: {row.name.date()}<br>Return: {row['Return']:.2%}<br>Volume: {row['Volume']:.2f}M<br>Type: {row['Behavior']}", axis=1),
    hoverinfo="text"
))
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        zaxis_title="UMAP-3"
    )
)
st.plotly_chart(fig, use_container_width=True)

# Cluster Stats
st.subheader("📊 Cluster Summary")
summary = df.groupby("Behavior")["Return"].agg(["count", "mean", "std"]).rename(columns={"count": "Days", "mean": "Avg Return", "std": "Volatility"})
st.dataframe(summary.round(4))
