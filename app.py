import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_openml
import umap

st.set_page_config(page_title = "Fashion MNIST 3D Explorer", layout = "wide")

CLASS_NAMES = [
    "T-Shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

@st.cache_data
def load_data():
    mnist = fetch_openml("Fashion-MNIST", version = 1, as_frame = False)
    X = mnist.data.astype(np.float32)
    Y = mnist.target.astype(int)
    return X, Y

@st.cache_data
def run_umap(X, Y, n_samples = 3000):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), n_samples, replace = False)
    X_sample = X[indices] / 255.0
    Y_sample = Y[indices]

    reducer = umap.UMAP(n_components = 3, random_state = 42)
    embedding = reducer.fit_transform(X_sample)

    return embedding, Y_sample

def build_figure(embedding, Y_sample):
    df = pd.DataFrame ({
        'x' : embedding[: , 0],
        'y' : embedding[: , 1],
        'z' : embedding[: , 2],
        "label" : [CLASS_NAMES[i] for i in Y_sample]
    })
    fig = px.scatter_3d(
        df, x = 'x', y = 'y', z = 'z',
        color = "label",
        title = "Fashion MNIST - 3D UMAP Embedding",
        opacity = 0.7,
        height = 750
    )
    fig.update_traces(marker = dict(size = 2))
    fig.update_layout(legend_title_text = "Category")

    return fig

def main():
    st.title("Fashion MNIST 3D Explorer")
    st.markdown("Visualizing clothing categories using **UMAP** dimensioanlity reduction - from 784 pixel dimensions down to 3.")

    with st.spinner("Loading Fashion MNIST dataset ..."):
        X, Y = load_data()

    with st.spinner("Running UMAP (this may take a moment)..."):
        embedding, Y_sample = run_umap(X, Y)

    fig = build_figure(embedding, Y_sample)
    st.plotly_chart(fig, use_container_width = True)

    st.markdown("---")
    st.markdown("**How it works:** Each point represents a clothing item. UMAP compresses 784 pixel values into 3 dimensions while preserving structure — similar items cluster together.")

if __name__ == "__main__":
    main()
