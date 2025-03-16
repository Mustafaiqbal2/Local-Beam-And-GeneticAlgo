from collections import defaultdict
import random
from typing import List, Dict, Set, Tuple
import math

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
        self.vertices = self._extract_vertices()
        self.adjacency = self._build_adjacency()
        self.vertex_order = self._compute_vertex_order()
        self.preassigned = preassigned_colors or {}
        self.distance_constraints = distance_constraints or []
        self.constraint_map = self._build_constraint_map()
        
    def _extract_vertices(self) -> Set[int]:
        """Extract unique vertices from edges"""
        vertices = set()
        for src, dst in self.edges:
            vertices.add(src)
            vertices.add(dst)
        return vertices
    
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """Create adjacency list representation of the graph"""
        adjacency = defaultdict(set)
        for src, dst in self.edges:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        return adjacency
    
    def _compute_vertex_order(self) -> List[int]:
        """Order vertices by degree (highest to lowest) as required"""
        degree = {v: len(self.adjacency[v]) for v in self.vertices}
        return sorted(self.vertices, key=lambda v: degree[v], reverse=True)
    
    def _build_constraint_map(self) -> Dict[int, Set[int]]:
        """Create a map of all constraints (adjacency + distance)"""
        constraint_map = defaultdict(set)
        
        # Add adjacent vertices
        for vertex, neighbors in self.adjacency.items():
            constraint_map[vertex].update(neighbors)
        
        # Add distance constraints
        for v1, v2 in self.distance_constraints:
            constraint_map[v1].add(v2)
            constraint_map[v2].add(v1)
            
        return constraint_map
    
    def _is_valid(self, coloring: Dict[int, int]) -> bool:
        """Check if the coloring is valid according to all constraints"""
        # Check adjacent vertex constraint and distance constraints
        for vertex, color in coloring.items():
            for constrained_vertex in self.constraint_map[vertex]:
                if constrained_vertex in coloring and coloring[constrained_vertex] == color:
                    return False
        
        # Check preassigned colors
        for vertex, color in self.preassigned.items():
            if vertex in coloring and coloring[vertex] != color:
                return False
                
        return True
    
    def _calculate_balance_score(self, coloring: Dict[int, int]) -> float:
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

    def _generate_initial_state(self) -> Dict[int, int]:
        """Generate an initial state with minimum colors possible"""
        # Start with preassigned colors
        coloring = self.preassigned.copy()
        
        # Track highest color used
        highest_color = max(coloring.values()) if coloring else -1
        
        # Process vertices strictly by degree order (highest first)
        for vertex in self.vertex_order:
            if vertex in coloring:  # Skip preassigned vertices
                continue
                
            # Get forbidden colors from constraints
            forbidden_colors = set()
            for constrained_vertex in self.constraint_map[vertex]:
                if constrained_vertex in coloring:
                    forbidden_colors.add(coloring[constrained_vertex])
            
            # Try to use the lowest possible color
            for color in range(highest_color + 2):  # +2 to allow one new color if needed
                if color not in forbidden_colors:
                    coloring[vertex] = color
                    highest_color = max(highest_color, color)
                    break
                    
        return coloring
    def _generate_successor(self, state: Dict[int, int]) -> Dict[int, int]:
        """
        Generate a neighboring state by changing one vertex's color
        With strong emphasis on color reduction
        """
        new_state = state.copy()
        current_colors = sorted(set(state.values()))
        max_color = max(current_colors)
        
        # 40% chance to try color reduction
        if random.random() < 0.4:
            # Get vertices using the highest color (that aren't preassigned)
            max_color_vertices = [v for v, c in state.items() if c == max_color and v not in self.preassigned]
            
            if max_color_vertices:
                # Try each vertex with the max color
                random.shuffle(max_color_vertices)
                for vertex in max_color_vertices:
                    # Get forbidden colors
                    forbidden_colors = set()
                    for constrained_vertex in self.constraint_map[vertex]:
                        if constrained_vertex in state:
                            forbidden_colors.add(state[constrained_vertex])
                    
                    # Try to use only existing colors below max_color
                    available_colors = [c for c in current_colors if c < max_color and c not in forbidden_colors]
                    
                    if available_colors:
                        new_state[vertex] = min(available_colors)  # Use lowest available color
                        return new_state
        
        # Regular successor generation if color reduction failed
        # Try vertices in high degree order
        for vertex in self.vertex_order:
            # Skip preassigned vertices
            if vertex in self.preassigned:
                continue
                
            current_color = state[vertex]
            
            # Get forbidden colors
            forbidden_colors = set()
            for constrained_vertex in self.constraint_map[vertex]:
                if constrained_vertex in state:
                    forbidden_colors.add(state[constrained_vertex])
            
            # Try existing colors first (strongly prefer lower color numbers)
            available_colors = [c for c in current_colors if c != current_color and c not in forbidden_colors]
            
            if available_colors:
                # 70% chance to choose the lowest color number (minimizing)
                # 30% chance to choose based on balance
                if random.random() < 0.7:
                    new_state[vertex] = min(available_colors)
                else:
                    color_counts = defaultdict(int)
                    for c in state.values():
                        color_counts[c] += 1
                    
                    # Temporarily remove this vertex's contribution
                    color_counts[current_color] -= 1
                    
                    # Choose color with minimum usage
                    new_state[vertex] = min(available_colors, key=lambda c: color_counts[c])
                
                return new_state
        
        # If we couldn't find a valid move, return the original state
        return state

def local_beam_search(graph: GraphColoring, k: int = 10, max_iterations: int = 150) -> Dict[int, int]:
    """
    Implement Local Beam Search for graph coloring with strong emphasis on color minimization
    """
    def evaluate_state(state: Dict[int, int]) -> float:
        """
        Evaluate a state based on number of colors with very strong penalty for more colors
        """
        if not graph._is_valid(state):
            return float('-inf')
            
        num_colors = len(set(state.values()))
        balance_score = graph._calculate_balance_score(state)
        
        # Very heavy weight on minimizing colors (0.9) vs balance (0.1)
        return -0.9 * num_colors + 0.1 * balance_score
    
    # Function to attempt color reduction
    def try_reduce_colors(state: Dict[int, int]) -> Dict[int, int]:
        """Try to eliminate the highest color number"""
        colors_used = set(state.values())
        if len(colors_used) <= 5:  # Already at target or below
            return state
            
        max_color = max(colors_used)
        new_state = state.copy()
        
        # Get vertices using the highest color
        max_color_vertices = [v for v, c in state.items() if c == max_color and v not in graph.preassigned]
        
        # Try to reassign each vertex with max_color to a lower color
        for vertex in max_color_vertices:
            # Get forbidden colors
            forbidden_colors = set()
            for neighbor in graph.constraint_map[vertex]:
                if neighbor in new_state:
                    forbidden_colors.add(new_state[neighbor])
            
            # Try to use only existing colors below max_color
            available_colors = [c for c in colors_used if c < max_color and c not in forbidden_colors]
            
            if available_colors:
                new_state[vertex] = min(available_colors)  # Use lowest available color
            else:
                # If we can't reassign even one vertex, abort
                return state
        
        return new_state if graph._is_valid(new_state) else state
    
    # Generate k initial states
    states = []
    for _ in range(k):
        state = graph._generate_initial_state()
        
        # Apply color reduction immediately
        reduced_state = try_reduce_colors(state)
        states.append(reduced_state)
    
    # Sort states by evaluation score (best first)
    states.sort(key=evaluate_state, reverse=True)
    
    # Keep track of best state found
    best_state = states[0]
    best_score = evaluate_state(best_state)
    best_num_colors = len(set(best_state.values()))
    
    print(f"Starting local beam search...")
    print(f"Initial best state: {best_num_colors} colors, balance: {graph._calculate_balance_score(best_state):.4f}")
    
    iterations_without_improvement = 0
    
    for iteration in range(max_iterations):
        # Periodically attempt aggressive color reduction
        if iteration % 5 == 0:
            for i in range(len(states)):
                states[i] = try_reduce_colors(states[i])
        
        # Generate successors for all current states
        all_successors = []
        for state in states:
            # Generate multiple successors for each state
            for _ in range(3):
                neighbor = graph._generate_successor(state)
                # Add only if not already in successors
                all_successors.append(neighbor)
        
        # Apply color reduction to some successors
        for i in range(len(all_successors)):
            if random.random() < 0.3:  # 30% chance
                all_successors[i] = try_reduce_colors(all_successors[i])
        
        # Combine current states with successors
        candidates = states + all_successors
        
        # Remove duplicates
        unique_candidates = []
        seen = set()
        for state in candidates:
            state_tuple = tuple(sorted(state.items()))
            if state_tuple not in seen:
                unique_candidates.append(state)
                seen.add(state_tuple)
        
        # Sort by evaluation score and select k best
        states = sorted(unique_candidates, key=evaluate_state, reverse=True)[:k]
        
        # Update best state if improved
        current_best = states[0]
        current_score = evaluate_state(current_best)
        current_num_colors = len(set(current_best.values()))
        
        if current_score > best_score:
            best_state = current_best.copy()
            best_score = current_score
            best_num_colors = current_num_colors
            iterations_without_improvement = 0
            print(f"Iteration {iteration}: New best score = {best_score:.4f}, Colors = {best_num_colors}")
            
            # Early exit if we hit the target
            if best_num_colors <= 5:
                print(f"Target color count achieved!")
                break
        else:
            iterations_without_improvement += 1
            
        # Early stopping if no improvement for a while
        if iterations_without_improvement >= 20:
            print(f"Early stopping at iteration {iteration} - No improvement for 20 iterations")
            break
    
    # After beam search, try one final aggressive optimization for the best state
    for _ in range(10):  # Try multiple times
        best_state = try_reduce_colors(best_state)
    
    # If we still have too many colors, try to balance what we have
    if len(set(best_state.values())) > 5:
        print("Could not reduce to 5 colors. Optimizing balance with current color count.")
        best_state = optimize_balance(best_state, graph)
    
    return best_state

def optimize_balance(state: Dict[int, int], graph: GraphColoring) -> Dict[int, int]:
    """Optimize color balance without changing the number of colors"""
    # Get current color distribution
    color_counts = defaultdict(int)
    for color in state.values():
        color_counts[color] += 1
    
    # Get vertices in order of degree
    vertices = sorted([v for v in state.keys() if v not in graph.preassigned],
                      key=lambda v: len(graph.adjacency[v]), reverse=True)
    
    # Try to move vertices from most used to least used colors
    iterations = 100
    for _ in range(iterations):
        # Find most and least used colors
        most_used = max(color_counts.items(), key=lambda x: x[1])
        least_used = min(color_counts.items(), key=lambda x: x[1])
        
        if most_used[1] - least_used[1] <= 1:
            # Already balanced
            break
        
        # Try to move a vertex from most_used to least_used color
        for vertex in random.sample(vertices, min(100, len(vertices))):
            if state[vertex] != most_used[0] or vertex in graph.preassigned:
                continue
                
            # Check if least_used color is valid for this vertex
            is_valid = True
            for neighbor in graph.constraint_map[vertex]:
                if neighbor in state and state[neighbor] == least_used[0]:
                    is_valid = False
                    break
            
            if is_valid:
                # Update state and color counts
                state[vertex] = least_used[0]
                color_counts[most_used[0]] -= 1
                color_counts[least_used[0]] += 1
                break
                
    return state

def read_hypercube_dataset(filename: str) -> List[Tuple[int, int]]:
    """Read the hypercube dataset from file"""
    edges = []
    with open(filename, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                edges.append((src, dst))
    return edges

def balance_minimal_coloring(graph: GraphColoring, solution: Dict[int, int]) -> Dict[int, int]:
    """
    Optimize color balance for a solution with minimal colors
    This function preserves the number of colors used but improves balance
    """
    # Create a copy to work with
    balanced_solution = solution.copy()
    
    # Get current color counts
    color_counts = defaultdict(int)
    for color in balanced_solution.values():
        color_counts[color] += 1
    
    # Calculate target count per color (ideal balance)
    num_colors = len(set(balanced_solution.values()))
    num_vertices = len(balanced_solution)
    target_per_color = num_vertices / num_colors
    
    # Process vertices by degree (high to low)
    for vertex in graph.vertex_order:
        # Skip preassigned vertices
        if vertex in graph.preassigned:
            continue
            
        current_color = balanced_solution[vertex]
        
        # Only consider moving vertices from overcrowded colors
        if color_counts[current_color] <= target_per_color:
            continue
            
        # Get forbidden colors from constraints
        forbidden_colors = set()
        for constrained_vertex in graph.constraint_map[vertex]:
            if constrained_vertex in balanced_solution:
                forbidden_colors.add(balanced_solution[constrained_vertex])
        
        # Find available colors, focusing on underutilized ones
        available_colors = []
        for color in color_counts.keys():
            if color != current_color and color not in forbidden_colors:
                available_colors.append(color)
        
        if available_colors:
            # Choose the most underutilized color
            best_color = min(available_colors, key=lambda c: color_counts[c])
            
            # Only move if it improves balance
            if color_counts[best_color] < color_counts[current_color] - 1:
                color_counts[current_color] -= 1
                color_counts[best_color] += 1
                balanced_solution[vertex] = best_color
    
    return balanced_solution

def main():
    # Read dataset
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
    
    # Distance constraints
    distance_constraints = [
        (945, 951),   
        (995, 1019), 
        (913, 1009),
        (271, 415),
        (134, 647), 
        (1013, 1023),
        (779, 827)
    ]
    
    # Create graph coloring instance
    graph = GraphColoring(edges, preassigned_colors, distance_constraints)
    
    # Run local beam search as required by the assignment
    # Find minimal coloring solution
    solution = local_beam_search(graph, k=5, max_iterations=100)

    # If we found a 5-color solution, optimize its balance
    if len(set(solution.values())) <= 5:
        print("Found 5-color solution! Now optimizing balance...")
        balanced_solution = balance_minimal_coloring(graph, solution)
        solution = balanced_solution

    # Check if the solution is valid
    if graph._is_valid(solution):
        print("Solution is valid.")
    else:
        print("Solution is invalid!")
        
    # Print results
    print("\nFinal coloring statistics:")
    print(f"Number of colors used: {len(set(solution.values()))}")
    print(f"Color balance score: {graph._calculate_balance_score(solution):.4f}")
    
    # Print color distribution
    color_counts = defaultdict(int)
    for color in solution.values():
        color_counts[color] += 1
    
    print("\nColor distribution:")
    for color in sorted(color_counts.keys()):
        print(f"Color {color}: {color_counts[color]} vertices")

if __name__ == "__main__":
    main()