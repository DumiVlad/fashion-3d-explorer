# Fashion MNIST 3D Explorer

A small project I built to get hands-on with dimensionality reduction and 3D visualization; two things I've been curious about for a while.

## What it does

Fashion MNIST is a dataset of 70,000 grayscale clothing images across 10 categories (t-shirts, sneakers, bags, etc.). Each image is 28×28 pixels, which means each item is represented by 784 numbers.

The problem is you can't visualize 784 dimensions. So I used **UMAP** to compress those 784 numbers down to 3, and then plotted the result in an interactive 3D scatter plot.

The idea is simple: if two clothing items look similar, they should end up close to each other in 3D space. And for the most part, that's exactly what happens.

## How to run it

```bash
# Clone the repo
git clone https://github.com/DumiVlad/fashion-3d-explorer.git
cd fashion-3d-explorer

# Install dependencies
pip install streamlit umap-learn plotly scikit-learn numpy

# Run
streamlit run app.py
```

The first run takes a couple of minutes — it downloads the dataset and runs UMAP. After that it's cached, so subsequent runs are instant.

## Stack

- **UMAP** — dimensionality reduction (784D → 3D)
- **Plotly** — interactive 3D visualization
- **Streamlit** — turns the Python script into a web app
- **scikit-learn** — dataset loading and preprocessing

## Why I built this

I came across Leap Tools and got genuinely interested in the idea of using ML to represent and visualize products in 3D space. This project is my first step into that direction — I wanted to understand the fundamentals before anything else.

I'm still learning. But I learn best by building things.
