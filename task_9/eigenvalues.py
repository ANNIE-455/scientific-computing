import numpy as np

# Adjacency matrix for a small undirected network.
# Nodes: A, B, C, D
A = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

# Identify dominant eigenvalue/eigenvector
index = np.argmax(np.real(eigenvalues))
dominant_value = np.real(eigenvalues[index])
dominant_vector = np.abs(np.real(eigenvectors[:, index]))

# Normalize to form centrality measure
centrality = dominant_vector / dominant_vector.sum()
nodes = ["A", "B", "C", "D"]

print(f"Dominant Eigenvalue: {dominant_value:.4f}")
print("Eigenvector-based Network Importance (Centrality):")
for node, score in zip(nodes, centrality):
    print(f"  Node {node}: {score:.4f}")

# Verify Av = lambda * v
v = np.real(eigenvectors[:, index])
residual = np.linalg.norm(A @ v - dominant_value * v)
print(f"Verification Error ||Av - λv||: {residual:.2e}")