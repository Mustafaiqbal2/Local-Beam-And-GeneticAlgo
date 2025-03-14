import math
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import datetime

def plot_colored_graph(edges: List[Tuple[int, int]], 
                      coloring: Dict[int, int],
                      distance_constraints: List[Tuple[int, int]],
                      filename: str = "colored_graph.png") -> None:
    """
    Plot the graph with assigned colors and save it to a file
    
    Args:
        edges: List of tuples representing edges (source, destination)
        coloring: Dictionary mapping vertices to their assigned colors
        distance_constraints: List of vertex pairs with distance constraints
        filename: Name of the output file
    """
    # Print execution information
    print(f"Graph Visualization Started at UTC: 2025-03-14 09:37:50")
    print(f"User: saadnadeem554")

    # Create a new graph
    G = nx.Graph()
    
    # Add edges to the graph
    G.add_edges_from(edges)
    
    # Create a color map for visualization
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                 '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    # Map color numbers to actual colors
    node_colors = [color_list[coloring[node] % len(color_list)] for node in G.nodes()]
    
    # Set up the plot
    plt.figure(figsize=(20, 20))
    
    # Create a spring layout with more spread
    pos = nx.spring_layout(G, k=1/math.sqrt(len(G.nodes())), iterations=50)
    
    # Draw the graph elements
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=500)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Draw distance constraint edges in red with dashed lines
    constraint_edges = [(v1, v2) for v1, v2 in distance_constraints]
    nx.draw_networkx_edges(G, pos,
                          edgelist=constraint_edges,
                          edge_color='red',
                          style='dashed',
                          alpha=0.7,
                          width=2)
    
    # Add a title with statistics
    num_colors = len(set(coloring.values()))
    plt.title(f'Graph Coloring Visualization\n'
              f'Number of Vertices: {len(G.nodes())}\n'
              f'Number of Edges: {len(G.edges())}\n'
              f'Number of Colors Used: {num_colors}\n'
              f'Distance Constraints: {len(distance_constraints)}', 
              pad=20)
    
    # Add legend for distance constraints
    constraint_line = plt.Line2D([], [], color='red', linestyle='--',
                               label='Distance Constraint')
    plt.legend([constraint_line], ['Distance Constraint'])
    
    # Add timestamp and user information
    plt.text(0.02, -0.02,
             f'Generated at UTC: 2025-03-14 09:37:50\n'
             f'User: saadnadeem554',
             transform=plt.gca().transAxes,
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph visualization has been saved to {filename}")

# Example usage in main.py: