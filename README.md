# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

## Overview

This project explores the application of Convolutional Neural Networks (CNNs) to graph-structured data using spectral graph theory. The report discusses the challenges of applying traditional CNNs to irregular graph data and presents a method developed by Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst to address these challenges. The approach introduces spectral filters localized within a radius of $K$ hops from a central vertex, improving interpretability and efficiency.

## Key Concepts

### Spectral Graph Theory

Spectral graph theory is used to redefine convolution operations on graphs using the graph Laplacian. The Laplacian matrix $L$ is defined as:

```math
L = D - W = I_n - D^{-1/2} W D^{-1/2}
```

where $W$ is the weighted adjacency matrix, and $D$ is the diagonal degree matrix. The Laplacian is symmetric and positive semidefinite, providing a complete set of orthonormal eigenvectors, which are used for the Graph Fourier Transform (GFT).

### Localized Spectral Filtering

The convolution of a signal $x$ with a filter $g_\theta$ on a graph is expressed in the Fourier domain as:

```math
y = g_\theta(L) x = U g_\theta(\Lambda) U^T x
```

The filter $g_\theta$ is localized to ensure it only considers vertices within $K$ hops:

```math
g_\theta(\Lambda) = \sum_{k=0}^{K-1} \theta_k \Lambda^k
```

### Efficient Computation

To avoid the computational complexity of eigenvalue decomposition, the filter is computed directly using Chebyshev polynomials:

```math
g_\theta(L) = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L})
```

where $\tilde{L}$ is the scaled Laplacian. This approach achieves linear evaluation complexity relative to the filter's support size and the number of edges.

### Graph Coarsening and Pooling

The method employs a multi-level clustering approach using the Graclus algorithm to perform pooling operations, preserving local geometric structures. The coarsened graphs are represented as binary trees, allowing for efficient hierarchical feature aggregation.

## Experiments

The method was tested on the Cora dataset, which contains 2,708 nodes representing articles and edges representing citations. The goal was to classify articles into one of seven themes based on their feature vectors and graph structure.

### Implementation

The implementation used a single Chebyshev convolution followed by max-pooling, with a fully connected layer for classification. The model was trained using the torch geometric library in Python.

### Results

The model achieved a test accuracy of 79.2%, comparable to state-of-the-art performance in 2016. Training was efficient, taking only 1.5 minutes on a CPU.

This project demonstrates the effectiveness of spectral graph theory in applying CNNs to graph-structured data. The method provides a scalable and efficient solution for graph-based learning tasks, although further research is needed to address limitations related to dense graphs and dynamic environments.

## References

- Defferrard, M., Bresson, X., & Vandergheynst, P. (2017). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. arXiv:1606.09375.
- Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. arXiv:1903.02428.