from collections import defaultdict
import random
from typing import List, Dict, Set, Tuple
from graph_visualization import plot_colored_graph
import heapq

class GraphColoring:
    def __init__(self, edges: List[Tuple[int, int]], preassigned_colors: Dict[int, int] = None,
                 distance_constraints: List[Tuple[int, int]] = None):
        """
        Initialize the graph coloring problem
        
        Args:
            edges: List of tuples representing edges (source, destination)
            preassigned_colors: Dictionary of vertex to color mappings that cannot be changed
            distance_constraints: List of vertex pairs that must have different colors (2-hop constraint)
        """
        self.edges = edges
        self.vertices = self._get_vertices()
        self.adj_list = self._create_adj_list()
        self.vertex_degrees = self._calculate_degrees()
        self.preassigned_colors = preassigned_colors or {}
        self.distance_constraints = distance_constraints or []
        self.two_hop_neighbors = self._calculate_two_hop_neighbors()
        
    def _get_vertices(self) -> Set[int]:
        """Extract unique vertices from edges"""
        vertices = set()
        for src, dst in self.edges:
            vertices.add(src)
            vertices.add(dst)
        return vertices
    
    def _create_adj_list(self) -> Dict[int, Set[int]]:
        """Create adjacency list representation of the graph"""
        adj_list = defaultdict(set)
        for src, dst in self.edges:
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        return adj_list
    
    def _calculate_degrees(self) -> Dict[int, int]:
        """Calculate degree of each vertex"""
        return {v: len(self.adj_list[v]) for v in self.vertices}
    
    def _calculate_two_hop_neighbors(self) -> Dict[int, Set[int]]:
        """Calculate two-hop neighbors for each vertex"""
        two_hop = defaultdict(set)
        for v in self.vertices:
            # Get direct neighbors
            neighbors = self.adj_list[v]
            # Get neighbors of neighbors
            for neighbor in neighbors:
                two_hop[v].update(self.adj_list[neighbor])
            # Remove the vertex itself and direct neighbors
            two_hop[v] = two_hop[v] - {v} - neighbors
        return two_hop

    def _is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Check if the coloring is valid according to all constraints
        """
        # Check adjacent vertex constraint
        for v in self.vertices:
            if v in coloring:
                for neighbor in self.adj_list[v]:
                    if neighbor in coloring and coloring[v] == coloring[neighbor]:
                        return False
        
        # Check preassigned colors
        for v, color in self.preassigned_colors.items():
            if v in coloring and coloring[v] != color:
                return False
        
        # Check distance constraints (two-hop)
        for v1, v2 in self.distance_constraints:
            if v1 in coloring and v2 in coloring and coloring[v1] == coloring[v2]:
                return False
            
        return True

    def _calculate_color_balance(self, coloring: Dict[int, int]) -> float:
        """
        Calculate the balance score of color usage
        Returns a value between 0 and 1, where 1 is perfectly balanced
        """
        if not coloring:
            return 0
            
        color_counts = defaultdict(int)
        for color in coloring.values():
            color_counts[color] += 1
            
        min_count = min(color_counts.values())
        max_count = max(color_counts.values())
        
        if max_count == min_count:
            return 1.0
        return 1.0 - (max_count - min_count) / len(self.vertices)

    def _generate_neighbor(self, current_state: Dict[int, int]) -> Dict[int, int]:
        """
        Generate a neighboring state by changing one vertex's color
        """
        neighbor = current_state.copy()
        
        # Sort vertices by degree for modification priority
        vertices = sorted(self.vertices, key=lambda x: self.vertex_degrees[x], reverse=True)
        
        # Try to modify a random vertex that's not preassigned
        for _ in range(len(vertices)):
            vertex = random.choice(vertices)
            if vertex not in self.preassigned_colors:
                # Get current colors of neighbors
                neighbor_colors = {neighbor[n] for n in self.adj_list[vertex] if n in neighbor}
                two_hop_colors = {neighbor[n] for n in self.two_hop_neighbors[vertex] if n in neighbor}
                
                # Try existing colors first
                existing_colors = set(neighbor.values())
                available_colors = [c for c in existing_colors if c not in neighbor_colors and 
                                 (vertex not in self.distance_constraints or c not in two_hop_colors)]
                
                if not available_colors:  # If no existing color works, add a new one
                    new_color = max(existing_colors, default=-1) + 1
                    available_colors = [new_color]
                
                neighbor[vertex] = random.choice(available_colors)
                if self._is_valid_coloring(neighbor):
                    return neighbor
        
        return current_state
def local_beam_search(graph: GraphColoring, k: int = 5, max_iterations: int = 100) -> Dict[int, int]:
    """
    Implement Local Beam Search for graph coloring
    
    Args:
        graph: GraphColoring instance
        k: Number of states to maintain
        max_iterations: Maximum number of iterations
        
    Returns:
        Best coloring found
    """
    def generate_initial_state() -> Dict[int, int]:
        """Generate a valid initial state considering specific distance constraints"""
        # Start with preassigned colors
        coloring = graph.preassigned_colors.copy()
        
        # Get vertices sorted by degree (highest to lowest)
        vertices = sorted(graph.vertices, key=lambda x: graph.vertex_degrees[x], reverse=True)
        
        # Create a mapping of constrained pairs for quick lookup
        constrained_pairs = defaultdict(set)
        for v1, v2 in graph.distance_constraints:
            constrained_pairs[v1].add(v2)
            constrained_pairs[v2].add(v1)
        
        for vertex in vertices:
            if vertex not in coloring:
                # Get colors of adjacent vertices
                neighbor_colors = {coloring[n] for n in graph.adj_list[vertex] if n in coloring}
                
                # Only check distance constraints if this vertex is in constrained pairs
                forbidden_colors = set(neighbor_colors)
                if vertex in constrained_pairs:
                    # Add colors of constrained vertices to forbidden colors
                    for constrained_vertex in constrained_pairs[vertex]:
                        if constrained_vertex in coloring:
                            forbidden_colors.add(coloring[constrained_vertex])
                
                # Find the first available color
                color = 0
                while color in forbidden_colors:
                    color += 1
                coloring[vertex] = color
        
        return coloring


    def evaluate_state(state: Dict[int, int]) -> float:
        """
        Evaluate a state based on number of colors and balance
        Returns a score where higher is better
        """
        if not graph._is_valid_coloring(state):
            return float('-inf')
            
        num_colors = len(set(state.values()))
        balance_score = graph._calculate_color_balance(state)
        
        # Combine metrics (weighted sum)
        return -0.7 * num_colors + 0.3 * balance_score

    # Generate k initial states
    states = [generate_initial_state() for _ in range(k)]
    states = sorted(states, key=evaluate_state, reverse=True)

    for _ in range(max_iterations):
        # Generate neighbors for all current states
        neighbors = []
        for state in states:
            for _ in range(k):  # Generate k neighbors for each state
                neighbor = graph._generate_neighbor(state)
                neighbors.append(neighbor)
        
        # Select k best states among current states and their neighbors
        all_states = states + neighbors
        states = sorted(all_states, key=evaluate_state, reverse=True)[:k]
        
        # If the best state hasn't improved in several iterations, we can stop
        if len(set(tuple(state.items()) for state in states)) == 1:
            break

    return states[0]  # Return the best state found
def read_hypercube_dataset(filename: str) -> List[Tuple[int, int]]:
    """Read the hypercube dataset from file"""
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            src, dst, _ = line.strip().split()
            edges.append((int(src), int(dst)))
    return edges

def main():
    # Read the hypercube dataset
    edges = read_hypercube_dataset('hypercube_dataset.txt')
    
    # Define preassigned colors to vertices
    preassigned_colors = {
        0: 0,   
        512: 1,  
        256: 2,   
        1023: 3,
        1022: 2,
        511: 4,
    }
    
    # distance constraints: add 2 hop vertices that must have different colors
    distance_constraints = [
        (945,951),   
        (995, 1019), 
        (913,1009),
        (271,415),
        (134,647), 
        (1013,1023),
        (779,827)
    ]
    
    # Create graph coloring instance
    graph = GraphColoring(edges, preassigned_colors, distance_constraints)
    
    # Run local beam search
    solution = local_beam_search(graph, k=5, max_iterations=100)
     
    # check if the solution is valid
    if not graph._is_valid_coloring(solution):
        print("Solution is invalid!")
        exit(1)
    
    # Print results
    print("Final coloring:")
    print("Vertex -> Color")
    for vertex in sorted(solution.keys()):
        print(f"{vertex} -> {solution[vertex]}")
    
    print("\nNumber of colors used:", len(set(solution.values())))
    print("Color balance score:", graph._calculate_color_balance(solution))
    # Plot the graph
    plot_colored_graph(
        edges=edges,
        coloring=solution,
        distance_constraints=distance_constraints,
        filename="graph_coloring_result.png"
    )

if __name__ == "__main__":
    main()