import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import random
# Author: saadnadeem554
# Created: 2025-03-09 08:50:02 UTC

# -------------------- Data Models --------------------

class ProductCategory(Enum):
    DAIRY = "DAIRY"
    GRAIN = "GRAIN"
    FROZEN = "FROZEN"
    CEREAL = "CEREAL"
    PASTA = "PASTA"
    SAUCE = "SAUCE"
    CLEANING = "CLEANING"
    SNACKS = "SNACKS"
    PRODUCE = "PRODUCE"
    BEVERAGES = "BEVERAGES"
    MEAT = "MEAT"
    CONDIMENTS = "CONDIMENTS"
    COSMETICS = "COSMETICS"

class StorageType(Enum):
    REGULAR = "REGULAR"
    REFRIGERATED = "REFRIGERATED"
    HAZARDOUS = "HAZARDOUS"
    PROMOTIONAL = "PROMOTIONAL"
    SECURE = "SECURE"

@dataclass
class Product:
    id: str
    name: str
    weight: float
    category: ProductCategory
    storage_type: StorageType
    is_high_demand: bool
    is_perishable: bool
    is_hazardous: bool
    theft_risk: bool
    price: float
    related_products: List[str]
    is_promotional: bool

@dataclass
class Shelf:
    id: str
    name: str
    capacity: float
    storage_type: StorageType
    is_eye_level: bool
    is_bottom_level: bool
    is_high_visibility: bool
    is_secure: bool
    current_weight: float = 0
    products: List[Product] = None

    def __post_init__(self):
        if self.products is None:
            self.products = []

    def can_add_product(self, product: Product) -> bool:
        return (self.current_weight + product.weight <= self.capacity and
                self.storage_type == product.storage_type)

    def add_product(self, product: Product) -> bool:
        if self.can_add_product(product):
            self.products.append(product)
            self.current_weight += product.weight
            return True
        return False

    def remove_product(self, product: Product) -> bool:
        if product in self.products:
            self.products.remove(product)
            self.current_weight -= product.weight
            return True
        return False

# -------------------- Genetic Algorithm --------------------
class GroceryStoreGA:
    def __init__(self, products: List[Product], shelves: List[Shelf], 
                 population_size: int = 100,
                 generations: int = 1000):
        self.products = products
        self.shelves = shelves
        self.population_size = population_size
        self.generations = generations
        
    def generate_individual(self) -> List[int]:
        individual = []
        for product in self.products:
            # Get compatible shelves
            compatible_shelves = []
            for i, shelf in enumerate(self.shelves):
                if (shelf.storage_type == product.storage_type and 
                    shelf.capacity >= product.weight):
                    compatible_shelves.append(i)
            
            if compatible_shelves:
                individual.append(random.choice(compatible_shelves))
            else:
                individual.append(random.randint(0, len(self.shelves)-1))
        return individual
    def generate_individual2(self) -> List[int]:
        return [random.randint(0, len(self.shelves)-1) for _ in range(len(self.products))]
    
    def is_valid_solution(self, individual: List[int]) -> bool:
        shelf_weights = {shelf.id: 0 for shelf in self.shelves}
        
        for prod_idx, shelf_idx in enumerate(individual):
            product = self.products[prod_idx]
            shelf = self.shelves[shelf_idx]
            
            # Check storage type compatibility
            if product.storage_type != shelf.storage_type:
                return False
            
            # Check weight limit
            shelf_weights[shelf.id] += product.weight
            if shelf_weights[shelf.id] > shelf.capacity:
                return False
            
        return True

    def calculate_fitness(self, individual: List[int]) -> float:
        #if not self.is_valid_solution(individual):
         #   return 0.0

        score = 5000.0  # Start with base score
        shelf_assignments = {shelf.id: [] for shelf in self.shelves}
        shelf_weights = {shelf.id: 0 for shelf in self.shelves}

        # Assign products to shelves
        for prod_idx, shelf_idx in enumerate(individual):
            product = self.products[prod_idx]
            shelf = self.shelves[shelf_idx]
            shelf_assignments[shelf.id].append(product)
            shelf_weights[shelf.id] += product.weight  # Track weight per shelf

        # Evaluate each shelf assignment
        for shelf_id, assigned_products in shelf_assignments.items():
            shelf = next(s for s in self.shelves if s.id == shelf_id)

            # 🛑 1. Check shelf capacity
            if shelf_weights[shelf_id] > shelf.capacity:
                score -= (shelf_weights[shelf_id] - shelf.capacity) * 10  # Heavy penalty for overloading

            # ✅ 2. Check complementary products placement
            for product in assigned_products:
                for related_prod_id in product.related_products:
                    related_prod = next((p for p in self.products if p.id == related_prod_id), None)
                    if related_prod:
                        related_prod_shelf_idx = individual[self.products.index(related_prod)]
                        if related_prod_shelf_idx != self.shelves.index(shelf):
                            score -= 200  # Heavy penalty for separating related products

            # 👁️ 3. High-Demand Product Accessibility
            for product in assigned_products:
                if product.is_high_demand and not shelf.is_eye_level:
                    score -= 100  # Penalize high-demand products in hard-to-reach areas
                if not product.is_high_demand and shelf.is_eye_level:
                    score -= 50  # Penalize low-demand products at eye level

            # 🏪 4. Checkout display optimization for promotional items
            if shelf.storage_type == StorageType.PROMOTIONAL:
                if not any(p.is_promotional for p in assigned_products):
                    score -= 150  # Penalize if checkout display lacks promotional items

            # 📌 5. Category grouping
            categories = set(p.category for p in assigned_products)
            if len(categories) > 1:
                score -= 50 * (len(categories) - 1)  # Penalize multiple categories in one shelf

            # ❄️ 6. Perishable vs. Non-Perishable Separation
            for product in assigned_products:
                if product.is_perishable and not shelf.storage_type == StorageType.REFRIGERATED:
                    score -= 200  # Heavy penalty for storing perishable items incorrectly
            # penalize if the product is not stored in the correct storage type
            for product in assigned_products:
                if product.storage_type != shelf.storage_type:
                    score -= 100
            # ☣️ 7. Hazardous and Allergen-Free Zones
            for product in assigned_products:
                if product.is_hazardous and not shelf.storage_type == StorageType.HAZARDOUS:
                    score -= 200  # Heavy penalty for hazardous products outside designated area

            # 📦 8. Restocking Efficiency
            for product in assigned_products:
                if product.weight > 10 and not shelf.is_bottom_level:
                    score -= 100  # Heavy products should be on lower shelves
                if product.weight < 5 and shelf.is_bottom_level:
                    score -= 50

            # 🌡️ 9. Refrigeration Efficiency
            refrigerated_products = [p for p in assigned_products if p.is_perishable]
            if len(refrigerated_products) > 0 and len(set(individual[self.products.index(p)] for p in refrigerated_products)) > 1:
                score -= 100  # Penalize spreading refrigerated products across multiple units
            # 🔐 11. Theft Prevention
            for product in assigned_products:
                if product.theft_risk and not shelf.storage_type == StorageType.SECURE:
                    score -= 150  # Penalize high-value items in insecure locations

        return score
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        return parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    
    def mutate(self, individual: List[int]) -> List[int]:
        mutated = individual.copy()
        
        # Randomly select 1-3 positions to mutate
        num_mutations = random.randint(1, max(3, len(mutated) // 10))  # Mutate up to 10% of genes
        positions = random.sample(range(len(mutated)), num_mutations)
        
        for pos in positions:
            product = self.products[pos]
            compatible_shelves = []
            
            # Find compatible shelves
            for i, shelf in enumerate(self.shelves):
                if (shelf.storage_type == product.storage_type and 
                    shelf.capacity >= product.weight):
                    compatible_shelves.append(i)
            
            if compatible_shelves:
                mutated[pos] = random.choice(compatible_shelves)
            else:
                mutated[pos] = random.choice(range(len(self.shelves))) if compatible_shelves else mutated[pos]

                
        return mutated
    '''
    def evolve(self) -> Tuple[List[int], float]:
        # Initialize population
        population = [self.generate_individual2() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = -9999
        generations_without_improvement = 0

        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            current_best = fitness_scores[0]
            
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_solution = current_best[0]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early stopping if no improvement for 1000 generations
            if generations_without_improvement >= 1000:
                print(f"\nEarly stopping at generation {generation} - No improvement for 1000 generations")
                break
            
            # Status update
            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best 10% solutions
            elite_size = max(2, self.population_size // 10)
            new_population.extend([score[0] for score in fitness_scores[:elite_size]])
            
            # Fill rest of population with children by tournament selection
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                parent1 = max(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
                parent2 = max(random.sample(fitness_scores, tournament_size), key=lambda x: x[1])[0]
                
                # Create and mutate child
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        return best_solution, best_fitness '''
    def evolve(self) -> Tuple[List[int], float]:
        """
        Enhanced genetic algorithm evolution process with:
        - Dynamic population management
        - Adaptive mutation rates
        - Population diversity maintenance
        - Restart mechanism
        """
        # Initialize population
        population = [self.generate_individual() for _ in range(self.population_size)]
        best_solution = None
        best_fitness = -9999
        generations_without_improvement = 0
        mutation_rate = 0.1  # Initial mutation rate
        restart_count = 0
        max_restarts = 5

        # Track population diversity
        def calculate_diversity(pop):
            return len(set(tuple(ind) for ind in pop)) / len(pop)

        while restart_count < max_restarts:
            for generation in range(self.generations):
                # Evaluate fitness for all individuals
                fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
                fitness_scores.sort(key=lambda x: x[1], reverse=True)
                
                current_best = fitness_scores[0]
                current_diversity = calculate_diversity(population)
                
                # Update best solution
                if current_best[1] > best_fitness:
                    best_fitness = current_best[1]
                    best_solution = current_best[0].copy()  # Make sure to copy the solution
                    generations_without_improvement = 0
                    print(f"New best fitness found: {best_fitness}")
                else:
                    generations_without_improvement += 1
                
                # Adaptive mutation rate based on diversity and stagnation
                if current_diversity < 0.3:  # Low diversity
                    mutation_rate = min(0.5, mutation_rate * 1.5)
                elif generations_without_improvement > 50:
                    mutation_rate = min(0.5, mutation_rate * 1.2)
                else:
                    mutation_rate = max(0.1, mutation_rate * 0.95)
                
                # Early stopping check with restart mechanism
                if generations_without_improvement >= 200:
                    print(f"\nStagnation detected at generation {generation}")
                    if restart_count < max_restarts - 1:
                        print("Initiating restart with partial population refresh")
                        restart_count += 1
                        
                        # Keep best solutions but reinitialize rest
                        elite_size = max(2, self.population_size // 5)
                        new_population = [score[0] for score in fitness_scores[:elite_size]]
                        
                        # Add some completely new solutions
                        new_population.extend([self.generate_individual() 
                                            for _ in range(self.population_size - elite_size)])
                        
                        population = new_population
                        generations_without_improvement = 0
                        mutation_rate = 0.2  # Reset mutation rate
                        break
                    else:
                        print("\nMaximum restarts reached - stopping evolution")
                        return best_solution, best_fitness
                
                # Status update
                if generation % 100 == 0:
                    print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, "
                        f"Diversity = {current_diversity:.2f}, "
                        f"Mutation Rate = {mutation_rate:.2f}")
                
                # Create new population with enhanced selection pressure
                new_population = []
                
                # Adaptive elitism based on population diversity
                elite_size = max(2, int(self.population_size * (0.1 if current_diversity > 0.3 else 0.05)))
                new_population.extend([score[0].copy() for score in fitness_scores[:elite_size]])
                
                # Fill rest of population with children
                while len(new_population) < self.population_size:
                    # Dynamic tournament size based on diversity
                    tournament_size = 3 if current_diversity > 0.3 else 4
                    
                    # Tournament selection
                    tournament1 = random.sample(fitness_scores, tournament_size)
                    tournament2 = random.sample(fitness_scores, tournament_size)
                    parent1 = max(tournament1, key=lambda x: x[1])[0]
                    parent2 = max(tournament2, key=lambda x: x[1])[0]
                    
                    # Create child
                    if random.random() < 0.8:  # 80% chance of crossover
                        child = self.crossover(parent1, parent2)
                    else:  # Sometimes use single parent to maintain diversity
                        child = parent1.copy()
                    
                    # Adaptive mutation
                    if random.random() < mutation_rate:
                        child = self.mutate(child)
                    
                    new_population.append(child)
                
                # Ensure population size remains constant
                population = new_population[:self.population_size]
                
                # Optional: Inject new random solutions if diversity is very low
                if current_diversity < 0.1:
                    num_random = max(2, self.population_size // 20)
                    random_indices = random.sample(range(elite_size, len(population)), num_random)
                    for idx in random_indices:
                        population[idx] = self.generate_individual2()

        return best_solution, best_fitness

# -------------------- Excel Output Functions --------------------

def create_shelf_allocation_excel(filename: str, 
                                solution: List[int],
                                products: List[Product],
                                shelves: List[Shelf]):
    shelf_assignments = {shelf.id: [] for shelf in shelves}
    for prod_idx, shelf_idx in enumerate(solution):
        product = products[prod_idx]
        shelf = shelves[shelf_idx]
        shelf_assignments[shelf.id].append(product)

    data = []
    for shelf_id, assigned_products in shelf_assignments.items():
        shelf = next(s for s in shelves if s.id == shelf_id)
        total_weight = sum(p.weight for p in assigned_products)
        
        for product in assigned_products:
            data.append({
                'Shelf ID': shelf.id,
                'Shelf Name': shelf.name,
                'Shelf Capacity': shelf.capacity,
                'Current Weight': total_weight,
                'Storage Type': shelf.storage_type.value,
                'Product ID': product.id,
                'Product Name': product.name,
                'Product Weight': product.weight,
                'Category': product.category.value,
                'Is High Demand': product.is_high_demand,
                'Is Perishable': product.is_perishable,
                'Is Hazardous': product.is_hazardous,
                'Theft Risk': product.theft_risk,
                'Price': product.price,
                'is_promotional': product.is_promotional
            })

    df = pd.DataFrame(data)
    
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name='Shelf Allocation', index=False)
        
        summary_data = []
        for shelf_id, assigned_products in shelf_assignments.items():
            shelf = next(s for s in shelves if s.id == shelf_id)
            total_weight = sum(p.weight for p in assigned_products)
            summary_data.append({
                'Shelf ID': shelf.id,
                'Shelf Name': shelf.name,
                'Capacity': shelf.capacity,
                'Used Capacity': total_weight,
                'Remaining Capacity': shelf.capacity - total_weight,
                'Number of Products': len(assigned_products)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

# -------------------- Main Execution --------------------

def create_sample_data():
    # Create shelves
    shelves = [
        Shelf("S1", "Checkout Display", 8, StorageType.PROMOTIONAL, False, False, True, False),
        Shelf("S2", "Lower Shelf",25, StorageType.REGULAR, False, True, False, False),
        Shelf("S3", "Secure Shelf", 15, StorageType.SECURE, False, False, False, True),
        Shelf("S4", "Eye-Level Shelf",15, StorageType.REGULAR, True, False, True, False),
        Shelf("S5", "General Aisle Shelf",20, StorageType.REGULAR, False, False, False, False),
        Shelf("R1", "Refrigerator Zone",20, StorageType.REFRIGERATED, False, False, False, False),
        Shelf("H1", "Hazardous Item Zone",10, StorageType.HAZARDOUS, False, False, True, False)
    ]

    # Create products
    products = [
        Product("P1", "Milk", 5, ProductCategory.DAIRY, StorageType.REFRIGERATED, 
               True, True, False, False, 3.99, ["P4"],False),
        Product("P2", "Rice Bag", 12, ProductCategory.GRAIN, StorageType.REGULAR,
               False, False, False, False, 15.99, [],False),
        Product("P3", "Frozen Nuggets", 5, ProductCategory.FROZEN, StorageType.REFRIGERATED,
               False, True, False, False, 8.99, [],False),
        Product("P4", "Cereal", 3, ProductCategory.CEREAL, StorageType.REGULAR,
               True, False, False, False, 4.99, ["P1"],False),
        Product("P5", "Pasta", 2, ProductCategory.PASTA, StorageType.REGULAR,
               False, False, False, False, 2.99, ["P6"],False),
        Product("P6", "Pasta Sauce", 3, ProductCategory.SAUCE, StorageType.REGULAR,
               False, False, False, False, 3.99, ["P5"],False),
        Product("P7", "Detergent", 4, ProductCategory.CLEANING, StorageType.HAZARDOUS,
               False, False, True, False, 9.99, ["P8,p9"],False),
        Product("P8", "Glass Cleaner", 5, ProductCategory.CLEANING, StorageType.HAZARDOUS,
               False, False, True, False, 6.99, ["P7,p9"],False),
        Product("P9", "Drain cleaner", 1, ProductCategory.CLEANING, StorageType.HAZARDOUS,
               False, False, True, False, 6.99, ["P8,p7"],False),
        Product("P10", "Chips", 2, ProductCategory.SNACKS, StorageType.REGULAR,
               False, False, False, False, 1.99, [],True),
        Product("P11","Cheese", 4, ProductCategory.DAIRY, StorageType.REFRIGERATED,
               True, True, False, False, 2.99, [],False),
        Product("P12","Luxury Perfume", 1, ProductCategory.COSMETICS, StorageType.SECURE,
               False, False, False, True, 99.99, [],False),
    ]

    return products, shelves

def main():
    print("Grocery Store Shelf Optimization")
    print("Author:", "saadnadeem554")
    print("Created:", "2025-03-09 08:50:02 UTC")
    print("\nInitializing optimization...\n")

    # Create sample data
    products, shelves = create_sample_data()

    # Initialize and run genetic algorithm
    ga = GroceryStoreGA(products, shelves, population_size=10, generations=2000)
    best_solution, best_fitness = ga.evolve()

    print(f"\nBest Solution Fitness: {best_fitness}")

    # Create Excel output
    #create_shelf_allocation_excel("shelf_allocation.xlsx", best_solution, products, shelves)

    # Print solution
    shelf_assignments = {shelf.id: [] for shelf in shelves}
    for prod_idx, shelf_idx in enumerate(best_solution):
        product = products[prod_idx]
        shelf = shelves[shelf_idx]
        shelf_assignments[shelf.id].append(product)

    print("\nOptimal Shelf Allocation:")
    for shelf_id, assigned_products in shelf_assignments.items():
        if assigned_products:
            shelf = next(s for s in shelves if s.id == shelf_id)
            total_weight = sum(p.weight for p in assigned_products)
            print(f"\n{shelf.name} (Capacity: {shelf.capacity}kg, Current: {total_weight}kg):")
            for product in assigned_products:
                print(f"  - {product.name} ({product.weight}kg)")

    print("\nDetailed results have been saved to 'shelf_allocation.xlsx'")

if __name__ == "__main__":
    main()