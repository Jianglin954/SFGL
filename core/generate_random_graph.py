import torch
import ipdb

def generate_random_graph_adjacency_matrix_pytorch(n, p):
    """
    Generate a random undirected graph G(n, p) using PyTorch and return its adjacency matrix.

    Parameters:
        n (int): Number of nodes in the graph.
        p (float): Probability of edge creation between nodes (0 <= p <= 1).

    Returns:
        adjacency_matrix (torch.Tensor): Adjacency matrix of the random graph (values 0 or 1).
    """
    # Generate a random matrix with values between 0 and 1
    random_matrix = torch.rand((n, n))


    # Create an upper triangular adjacency matrix where random values < p are 1
    upper_triangular = torch.triu((random_matrix < p).int(), diagonal=1)
    
    # Make the matrix symmetric to represent an undirected graph
    adjacency_matrix = upper_triangular + upper_triangular.T
    
    return adjacency_matrix

# Parameters
n = 100  # Number of nodes
p = 0.3  # Probability of edge creation

# Generate the adjacency matrix
adj_matrix = generate_random_graph_adjacency_matrix_pytorch(n, p)

# Print the adjacency matrix
print("Adjacency Matrix of the Random Undirected Graph G(n, p):")
print(adj_matrix)
