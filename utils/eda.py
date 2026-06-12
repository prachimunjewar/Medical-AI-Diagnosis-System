import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data(filepath):
    return pd.read_csv(filepath)


def get_basic_stats(df):
    return {
        "total_records": len(df),
        "features": len(df.columns) - 1,
        "diabetic": int(df["Outcome"].sum()),
        "non_diabetic": int((df["Outcome"] == 0).sum()),
        "missing_values": int(df.isnull().sum().sum()),
    }


def plot_outcome_distribution(df):
    counts = df["Outcome"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    counts["Label"] = counts["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
    fig = px.pie(
        counts,
        names="Label",
        values="Count",
        title="Patient Outcome Distribution",
        color_discrete_sequence=["#1D9E75", "#E24B4A"],
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def plot_feature_distributions(df):
    features = ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "SkinThickness"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=features)
    colors = ["#378ADD", "#1D9E75", "#E24B4A", "#BA7517", "#7F77DD", "#D4537E"]
    for i, feat in enumerate(features):
        row, col = divmod(i, 3)
        fig.add_trace(
            go.Histogram(
                x=df[feat],
                name=feat,
                marker_color=colors[i],
                opacity=0.8,
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_layout(title_text="Feature Distributions", height=450, margin=dict(t=60, b=20))
    return fig


def plot_correlation_heatmap(df):
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=50, b=20), height=450)
    return fig


def plot_feature_vs_outcome(df):
    fig = px.box(
        df.melt(id_vars="Outcome", var_name="Feature", value_name="Value"),
        x="Feature",
        y="Value",
        color="Outcome",
        color_discrete_map={0: "#1D9E75", 1: "#E24B4A"},
        title="Feature Values by Outcome (0 = Non-Diabetic, 1 = Diabetic)",
    )
    fig.update_layout(margin=dict(t=50, b=20), height=400)
    return fig
