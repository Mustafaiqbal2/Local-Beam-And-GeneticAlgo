import random
import numpy as np
import pandas as pd
from collections import defaultdict

class Product:
    def __init__(self, id, name, weight, category, perishable=False, high_demand=False, 
                 hazardous=False, promotional=False, high_value=False, bulky=False, 
                 price=0.0, related_products=None):
        self.id = id
        self.name = name
        self.weight = weight
        self.category = category
        self.perishable = perishable
        self.high_demand = high_demand
        self.hazardous = hazardous
        self.promotional = promotional
        self.high_value = high_value  # For theft risk items
        self.bulky = bulky
        self.price = price
        self.related_products = related_products or []
        
    def __repr__(self):
        return f"{self.name} ({self.weight}kg)"

class Shelf:
    def __init__(self, id, name, capacity, is_refrigerated=False, is_eye_level=False, 
                 is_lower=False, is_hazardous_zone=False, is_secure=False, is_high_visibility=False):
        self.id = id
        self.name = name
        self.capacity = capacity
        self.is_refrigerated = is_refrigerated
        self.is_eye_level = is_eye_level
        self.is_lower = is_lower
        self.is_hazardous_zone = is_hazardous_zone
        self.is_secure = is_secure
        self.is_high_visibility = is_high_visibility
        
    def __repr__(self):
        return f"{self.name} ({self.capacity}kg)"

class ShelfOptimizationGA:
    def __init__(self, products, shelves, population_size=100, max_generations=500, 
                 mutation_rate=0.1, crossover_rate=0.8):
        self.products = products
        self.shelves = shelves
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        
    def _define_complementary_pairs(self):
        """Define pairs of products that complement each other"""
        pairs = []
        
        # Process explicit related products
        for product in self.products:
            for related_id in product.related_products:
                related_product = next((p for p in self.products if p.id == related_id), None)
                if related_product:
                    pairs.append((product.id, related_id))
        
        # Add category-based pairs
        pasta_products = [p for p in self.products if "Pasta" in p.name]
        pasta_sauce_products = [p for p in self.products if "Sauce" in p.name]
        
        for pasta in pasta_products:
            for sauce in pasta_sauce_products:
                pairs.append((pasta.id, sauce.id))
                
        return pairs
    
    def generate_individual(self):
        """Generate a random shelf assignment with constraint awareness"""
        assignment = {}
        
        # First pass: assign products to compatible shelves
        for product in self.products:
            valid_shelves = []
            
            for shelf in self.shelves:
                # Apply hard constraints
                if product.perishable and not shelf.is_refrigerated:
                    continue
                if product.hazardous and not shelf.is_hazardous_zone:
                    continue
                if product.high_value and not shelf.is_secure:
                    continue
                
                valid_shelves.append(shelf.id)
            
            if valid_shelves:
                assignment[product.id] = random.choice(valid_shelves)
            else:
                # If no valid shelf found, assign to any shelf
                assignment[product.id] = random.choice([s.id for s in self.shelves])
        
        # Second pass: try to put related products together
        complementary_pairs = self._define_complementary_pairs()
        for product1_id, product2_id in complementary_pairs:
            # 30% chance to place complementary products together
            if random.random() < 0.3 and product1_id in assignment and product2_id in assignment:
                product1 = next(p for p in self.products if p.id == product1_id)
                product2 = next(p for p in self.products if p.id == product2_id)
                
                # Check if we can place them together (if constraints allow)
                if not (product1.perishable != product2.perishable and 
                        (product1.perishable or product2.perishable)):
                    # Choose one of their shelves randomly
                    shelf_id = random.choice([assignment[product1_id], assignment[product2_id]])
                    assignment[product1_id] = shelf_id
                    assignment[product2_id] = shelf_id
                
        return assignment
    
    def generate_initial_population(self):
        """Generate initial population of shelf assignments"""
        return [self.generate_individual() for _ in range(self.population_size)]
    
    def calculate_fitness(self, individual):
        """Calculate fitness of an individual (lower is better)"""
        # Start with perfect fitness (0 penalties)
        fitness = 0
        
        # Create shelf assignments and weights
        shelf_assignments = {shelf.id: [] for shelf in self.shelves}
        shelf_weights = defaultdict(float)
        
        for product_id, shelf_id in individual.items():
            product = next(p for p in self.products if p.id == product_id)
            shelf_weights[shelf_id] += product.weight
            shelf_assignments[shelf_id].append(product)
        
        # Check constraints for each shelf
        for shelf_id, products in shelf_assignments.items():
            shelf = next(s for s in self.shelves if s.id == shelf_id)
            
            # 1. Weight capacity constraint
            if shelf_weights[shelf_id] > shelf.capacity:
                fitness += 50 * (shelf_weights[shelf_id] - shelf.capacity)
            
            # 2. Category grouping
            categories = set(p.category for p in products)
            if len(categories) > 1:
                fitness += 10 * (len(categories) - 1)  # Less severe penalty than reference
            
            # 3. Promotional items in checkout
            if shelf.is_high_visibility and not any(p.promotional for p in products):
                fitness += 15  # Penalize unused promotional space
        
        # Check constraints for each product
        for product in self.products:
            shelf_id = individual[product.id]
            shelf = next(s for s in self.shelves if s.id == shelf_id)
            
            # 4. High-demand product accessibility
            if product.high_demand and not shelf.is_eye_level:
                fitness += 20
            
            # 5. Perishable items constraint
            if product.perishable and not shelf.is_refrigerated:
                fitness += 100  # Major penalty
            
            # 6. Hazardous items constraint
            if product.hazardous and not shelf.is_hazardous_zone:
                fitness += 100  # Major penalty
            
            # 7. Bulky items constraint
            if product.bulky and not shelf.is_lower:
                fitness += 15  # Minor penalty
            
            # 8. Promotional items visibility
            if product.promotional and not shelf.is_high_visibility:
                fitness += 25
            
            # 9. Theft prevention
            if product.high_value and not shelf.is_secure:
                fitness += 40
        
        # 10. Complementary products
        complementary_pairs = self._define_complementary_pairs()
        for product1_id, product2_id in complementary_pairs:
            if product1_id in individual and product2_id in individual:
                if individual[product1_id] != individual[product2_id]:
                    fitness += 10  # Penalize separated complementary products
        
        # 11. Refrigeration efficiency
        refrigerated_shelves = [s.id for s in self.shelves if s.is_refrigerated]
        for shelf_id in refrigerated_shelves:
            # Count non-perishable items in refrigerated shelves
            non_perishable = sum(1 for p in shelf_assignments[shelf_id] if not p.perishable)
            if non_perishable > 0:
                fitness += 10 * non_perishable
        
        return fitness
    
    def calculate_diversity(self, population):
        """Calculate diversity of a population (0-1)"""
        # Convert each individual to a hashable representation
        unique_individuals = set(
            tuple(sorted((pid, sid) for pid, sid in ind.items()))
            for ind in population
        )
        return len(unique_individuals) / len(population)
    
    def select_parents(self, population, fitnesses):
        """Select parents using tournament selection"""
        tournament_size = 3
        selected_parents = []
        
        for _ in range(2):  # Select 2 parents
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected_parents.append(population[winner_idx])
            
        return selected_parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Two-point crossover
        product_ids = list(parent1.keys())
        if len(product_ids) <= 2:
            return parent1.copy(), parent2.copy()
            
        points = sorted(random.sample(range(1, len(product_ids)), 2))
        point1, point2 = points
        
        child1 = {}
        child2 = {}
        
        for i, pid in enumerate(product_ids):
            if i < point1 or i >= point2:
                child1[pid] = parent1[pid]
                child2[pid] = parent2[pid]
            else:
                child1[pid] = parent2[pid]
                child2[pid] = parent1[pid]
                
        return child1, child2
    
    def mutate(self, individual, mutation_rate=None):
        """Randomly mutate an individual with adaptive mutation"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
            
        mutated = individual.copy()
        
        # Determine number of mutations based on rate and product count
        num_mutations = max(1, int(mutation_rate * len(self.products) * 0.3))
        products_to_mutate = random.sample(list(mutated.keys()), min(num_mutations, len(mutated)))
        
        for product_id in products_to_mutate:
            product = next(p for p in self.products if p.id == product_id)
            valid_shelves = []
            
            for shelf in self.shelves:
                # Check basic constraints
                if product.perishable and not shelf.is_refrigerated:
                    continue
                if product.hazardous and not shelf.is_hazardous_zone:
                    continue
                if product.high_value and not shelf.is_secure:
                    continue
                
                valid_shelves.append(shelf.id)
            
            if valid_shelves:
                # Avoid assigning to the same shelf
                current_shelf = mutated[product_id]
                possible_shelves = [s for s in valid_shelves if s != current_shelf]
                
                if possible_shelves:
                    mutated[product_id] = random.choice(possible_shelves)
                else:
                    mutated[product_id] = random.choice(valid_shelves)
                    
        return mutated
    
    def run(self):
        """Run the enhanced genetic algorithm with adaptive parameters"""
        population = self.generate_initial_population()
        
        best_fitness = float('inf')
        best_individual = None
        
        # Parameters for adaptive GA
        stagnation_counter = 0
        diversity_history = []
        adaptive_mutation_rate = self.mutation_rate
        
        for generation in range(self.max_generations):
            # Calculate fitness for each individual
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Find the best individual
            gen_best_idx = fitnesses.index(min(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_individual = population[gen_best_idx]
            
            # Calculate population diversity
            diversity = self.calculate_diversity(population)
            diversity_history.append(diversity)
            
            # Update best solution if improved
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best_individual.copy()
                print(f"Generation {generation}: New best fitness = {best_fitness}")
                stagnation_counter = 0
                
                # If we found a perfect solution, stop
                if best_fitness == 0:
                    break
            else:
                stagnation_counter += 1
            
            # Adaptive mutation rate based on diversity and stagnation
            if diversity < 0.3:  # Low diversity
                adaptive_mutation_rate = min(0.5, adaptive_mutation_rate * 1.2)
            elif stagnation_counter > 20:  # Stagnation
                adaptive_mutation_rate = min(0.5, adaptive_mutation_rate * 1.1)
            else:
                adaptive_mutation_rate = max(0.05, adaptive_mutation_rate * 0.9)
                
            # Check for complete stagnation
            if stagnation_counter >= 50:
                print(f"Stagnation detected at generation {generation}. Resetting population...")
                
                # Keep best 20% individuals
                elite_size = max(1, self.population_size // 5)
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
                elite = [population[i] for i in elite_indices]
                
                # Generate new individuals for the rest
                new_individuals = [self.generate_individual() for _ in range(self.population_size - elite_size)]
                
                # Reset population
                population = elite + new_individuals
                stagnation_counter = 0
                adaptive_mutation_rate = 0.2  # Reset mutation rate
                continue
            
            # Create a new population with elitism
            new_population = []
            
            # Elitism - keep the best individuals
            elite_size = max(1, int(self.population_size * 0.1))
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
            elite = [population[i].copy() for i in elite_indices]
            new_population.extend(elite)
            
            # Fill the rest of the population
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Perform crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Perform mutation with adaptive rate
                child1 = self.mutate(child1, adaptive_mutation_rate)
                child2 = self.mutate(child2, adaptive_mutation_rate)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace population
            population = new_population
            
            # Status update
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}, Diversity = {diversity:.2f}, Mutation rate = {adaptive_mutation_rate:.2f}")
        
        # Store best solution for later use
        self.best_individual = best_individual
        return best_individual, best_fitness
    
    def save_results_to_excel(self, solution, filename="shelf_optimization_results.xlsx"):
        """Save the optimization results to an Excel file"""
        # Create a DataFrame with the results
        results = []
        
        for product_id, shelf_id in solution.items():
            product = next(p for p in self.products if p.id == product_id)
            shelf = next(s for s in self.shelves if s.id == shelf_id)
            
            results.append({
                "Product ID": product_id,
                "Product Name": product.name,
                "Weight (kg)": product.weight,
                "Category": product.category,
                "Price": getattr(product, "price", "N/A"),
                "Perishable": "Yes" if product.perishable else "No",
                "High Demand": "Yes" if product.high_demand else "No",
                "Hazardous": "Yes" if product.hazardous else "No",
                "Promotional": "Yes" if product.promotional else "No",
                "High Value": "Yes" if product.high_value else "No",
                "Bulky": "Yes" if product.bulky else "No",
                "Assigned Shelf": shelf.name,
                "Shelf Capacity (kg)": shelf.capacity
            })
        
        df = pd.DataFrame(results)
        
        # Calculate shelf weight utilization
        shelf_utilization = defaultdict(float)
        shelf_assignments = {shelf.id: [] for shelf in self.shelves}
        
        for product_id, shelf_id in solution.items():
            product = next(p for p in self.products if p.id == product_id)
            shelf_utilization[shelf_id] += product.weight
            shelf_assignments[shelf_id].append(product)
        
        shelf_stats = []
        for shelf in self.shelves:
            shelf_stats.append({
                "Shelf ID": shelf.id,
                "Shelf Name": shelf.name,
                "Capacity (kg)": shelf.capacity,
                "Used Capacity (kg)": shelf_utilization[shelf.id],
                "Utilization (%)": (shelf_utilization[shelf.id] / shelf.capacity) * 100 if shelf.capacity > 0 else 0,
                "Number of Products": len(shelf_assignments[shelf.id]),
                "Categories": ", ".join(set(p.category for p in shelf_assignments[shelf.id]))
            })
        
        df_shelves = pd.DataFrame(shelf_stats)
        
        # Save to Excel
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name="Product Assignments", index=False)
            df_shelves.to_excel(writer, sheet_name="Shelf Utilization", index=False)
            
        print(f"Results saved to {filename}")

def create_enhanced_example_data():
    """Create an enhanced set of example products and shelves"""
    shelves = [
        Shelf("S1", "Checkout Display", 8, is_high_visibility=True),
        Shelf("S2", "Lower Shelf", 25, is_lower=True),
        Shelf("S3", "Secure Shelf", 15, is_secure=True),
        Shelf("S4", "Eye-Level Shelf", 15, is_eye_level=True, is_high_visibility=True),
        Shelf("S5", "General Aisle Shelf", 20),
        Shelf("R1", "Refrigerator Zone", 20, is_refrigerated=True),
        Shelf("H1", "Hazardous Item Zone", 10, is_hazardous_zone=True)
    ]
    
    products = [
        Product("P1", "Milk", 5, "Dairy", perishable=True, high_demand=True, price=3.99, related_products=["P4"]),
        Product("P2", "Rice Bag", 12, "Grains", bulky=True, price=15.99),
        Product("P3", "Frozen Nuggets", 5, "Frozen", perishable=True, price=8.99),
        Product("P4", "Cereal", 3, "Breakfast", high_demand=True, price=4.99, related_products=["P1"]),
        Product("P5", "Pasta", 2, "Grains", price=2.99, related_products=["P6"]),
        Product("P6", "Pasta Sauce", 3, "Condiments", price=3.99, related_products=["P5"]),
        Product("P7", "Detergent", 4, "Cleaning", hazardous=True, price=9.99, related_products=["P8","P9"]),
        Product("P8", "Glass Cleaner", 5, "Cleaning", hazardous=True, price=6.99, related_products=["P7"]),
        Product("P9", "Drain Cleaner", 1, "Cleaning", hazardous=True, price=6.99, related_products=["P8", "P7"]),
        Product("P10", "Chips", 2, "Snacks", promotional=True, price=1.99),
        Product("P11", "Cheese", 4, "Dairy", perishable=True, high_demand=True, price=2.99),
        Product("P12", "Luxury Perfume", 1, "Cosmetics", high_value=True, price=99.99)
    ]
    
    return products, shelves

def main():
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
    print(f"Products: {len(products)}, Shelves: {len(shelves)}")
    solution, fitness = optimizer.run()
    
    print("\nOptimized Shelf Allocation:")
    shelf_assignments = defaultdict(list)
    for product_id, shelf_id in solution.items():
        product = next(p for p in products if p.id == product_id)
        shelf = next(s for s in shelves if s.id == shelf_id)
        shelf_assignments[shelf.name].append(product)
    
    # Print nicely formatted results by shelf
    for shelf_name, assigned_products in shelf_assignments.items():
        shelf = next(s for s in shelves if s.name == shelf_name)
        total_weight = sum(p.weight for p in assigned_products)
        print(f"\n{shelf_name} (Capacity: {shelf.capacity}kg, Used: {total_weight}kg):")
        for product in assigned_products:
            attributes = []
            if product.perishable: attributes.append("Perishable")
            if product.high_demand: attributes.append("High Demand")
            if product.hazardous: attributes.append("Hazardous")
            if product.high_value: attributes.append("High Value")
            if product.promotional: attributes.append("Promotional")
            
            attr_str = f" [{', '.join(attributes)}]" if attributes else ""
            print(f"  - {product.name} ({product.weight}kg, ${product.price}){attr_str}")
    
    print(f"\nFinal Fitness Score: {fitness}")
    
    # Save results to Excel
    optimizer.save_results_to_excel(solution, "shelf_optimization_results.xlsx")

if __name__ == "__main__":
    main()


