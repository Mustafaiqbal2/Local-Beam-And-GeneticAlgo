# Local Beam Search and Genetic Algorithm

This repository contains implementations of Local Beam Search and Genetic Algorithm for solving optimization problems. The primary focus is on two main problems:

1. **Shelf Optimization using Genetic Algorithm (GA)**
2. **Graph Coloring using Local Beam Search**

## Shelf Optimization using Genetic Algorithm (GA)

The goal of this problem is to optimize the arrangement of products on shelves to maximize various constraints such as weight capacity, perishable items, high-demand products, and more.

### Key Features:
- **Product and Shelf Classes**: Define the properties of products and shelves.
- **Genetic Algorithm**: An enhanced genetic algorithm with adaptive parameters to find the optimal shelf arrangement.
- **Fitness Calculation**: Evaluate the fitness of each individual based on various constraints.
- **Mutation and Crossover**: Implement mutation and crossover operations for generating new populations.
- **Excel Export**: Save the optimization results to an Excel file for further analysis.

### Example Usage:

```python
from shelf_optimization_ga import create_enhanced_example_data, ShelfOptimizationGA

# Get enhanced example data
products, shelves = create_enhanced_example_data()

# Initialize and run GA with improved parameters
optimizer = ShelfOptimizationGA(
    products, 
    shelves, 
    population_size=50, 
    max_generations=300,
    mutation_rate=0.15,
    crossover_rate=0.85
)

print("Starting Shelf Optimization...")
solution, fitness = optimizer.run()

# Save results to Excel
optimizer.save_results_to_excel(solution, "shelf_optimization_results.xlsx")
```

## Graph Coloring using Local Beam Search

The goal of this problem is to color the vertices of a graph such that no two adjacent vertices share the same color while minimizing the number of colors used.

### Key Features:
- **GraphColoring Class**: Define the properties of the graph, including edges, preassigned colors, and distance constraints.
- **Local Beam Search**: Implement Local Beam Search with strong emphasis on color minimization.
- **Color Balance Optimization**: Optimize the balance of color usage while preserving the number of colors.

### Example Usage:

```python
from graph_coloring import GraphColoring, local_beam_search, balance_minimal_coloring

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

# Run local beam search
solution = local_beam_search(graph, k=5, max_iterations=100)

# Optimize balance if 5-color solution is found
if len(set(solution.values())) <= 5:
    balanced_solution = balance_minimal_coloring(graph, solution)
    solution = balanced_solution

# Check the solution validity
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
```

## Repository Structure

```
.
├── q1.py                       # Graph Coloring using Local Beam Search
├── q2.py                       # Shelf Optimization using Genetic Algorithm
├── README.md                   # This README file
└── hypercube_dataset.txt       # Sample dataset for graph coloring
```

## Requirements

- Python 3.x
- pandas
- numpy

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
