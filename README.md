# svd-algebra

**svd-algebra** is a Python tool designed to analyze applications of **Singular Value Decomposition (SVD)**. It provides functionality for solving linear systems, generating matrices with specific ranks, and performing reduced rank approximation for tasks like image compression.

## Features

* **Matrix Generation**: Create random matrices with specified dimensions ($M \times N$) and rank ($K$).
* **Linear Systems Solver**: Solve systems of the form $Ax = b$ using SVD decomposition, calculating the pseudo-inverse of the singular value matrix.
* **Reduced Rank Approximation**: Compress matrices (images) by keeping only the top $k$ singular values.
* **Analysis Tools**: Calculate Frobenius norms to measure approximation error.
* **Visualization**: Helper tools for plotting scatter graphs and displaying images.

## Installation

This project requires Python 3.9+ and the dependencies listed in `pyproject.toml` (`numpy`, `matplotlib`).

To install the package locally:

```bash
pip install .
```

## Usage

The package exposes the `SVDTools` and `Visualizer` classes directly.

### 1. Solving Linear Systems

You can generate random matrices and solve for $x$ in $Ax=b$. This is useful for analyzing numerical stability and the effect of matrix rank.

```python
import numpy as np
from svd_algebra import SVDTools

# 1. Generate a random 10x10 matrix with rank 5
 M, N, K = 10, 10, 5
A = SVDTools.generate_matrix(M, N, K)
b = SVDTools.generate_vector(M)

# 2. Solve the system
# Returns: ((A, Reconstructed A), (b, Reconstructed b))
results = SVDTools.solve_linear_system(A, b, verbose=True)

# 3. Check the calculated vector x
# (Printed to stdout if verbose=True)
```

### 2. Image Compression (Reduced Rank Approximation)

You can load an image, convert it to grayscale, and approximate it using a lower rank $k$. This effectively compresses the image information.

```python
import matplotlib.image as mpimg
from svd_algebra import SVDTools, Visualizer

# 1. Load image and convert to grayscale
img = mpimg.imread('lena.png')
img_gray = Visualizer.rgb_to_gray(img)

# 2. Compress using the top 50 singular values
k = 50
compressed_img = SVDTools.reduced_rank_approximation(img_gray, k)

# 3. Visualize the result
Visualizer.show_image(compressed_img)
```

## Project Structure

* `src/svd_algebra/`: Contains the core library source code (`core.py`).
* `exemples/`: Contains sample scripts for analyzing linear systems and compressing images.
* `Relatorio.ipynb`: A detailed Jupyter Notebook report explaining the theory and results of the SVD experiments.

## Authors

* **Luís Rafael Sena**
