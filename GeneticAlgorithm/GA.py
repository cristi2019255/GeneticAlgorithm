import copy
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm.BitArrayGenome import BitArrayGenome

class GeneticAlgorithm:            

    def __init__(self,genome_config = {'type': BitArrayGenome}, 
                 pop_size = 100,
                 generations_unchanged = 5, 
                 optimum = 40,
                 fitness_function = lambda x: np.sum(x)):
        """
        :param pop_size: population size                                
        :param mutation_rate: mutation rate
        :param max_iter: maximum number of iterations
        :param elitism_rate: elitism rate
        """        
        self.pop_size = pop_size        
        self.population = []                        
        self.genome_class = genome_config['type']
        self.genome_config = genome_config
        self.generations_unchanged = generations_unchanged
        self.fitness = fitness_function  
        self.optimum = optimum      
        self.create_population()        

    def create_population(self):
        """
        initializing genetic algorithm population
        each individual information structure is initialized with 
        random values
        :return:
        """
        self.population = [ self.genome_class(self.genome_config) for _ in range(self.pop_size)]        
        self.best_genome = self.population[0]

    def resolve(self, strategy = 'crossover_strategy', plot_results = False, write_results = False):
        """_summary_

        Args:
            strategy (str, optional): _description_. Defaults to 'crossover_strategy'.

        Returns:
            _type_: _description_
        """
        
        # choosing solver strategy
        strategies = {'crossover_strategy': self._resolve_crossover_strategy}    
        resolve_one_generation = self._resolve_crossover_strategy
        if strategy in strategies:
            resolve_one_generation = strategies[strategy]            
        
                
        best_fitnesses, mean_fitnesses = [], []        
        count_generations_unchanged = 0
        generation = 0
        
        while (self.best_genome.fitness != self.optimum and count_generations_unchanged < self.generations_unchanged):        
            
            generation += 1
                        
            self.evaluate_fitnesses()
            
            resolve_one_generation() # updating population                        
            
            fitnesses = [genome.fitness for genome in self.population]
            
            if len(best_fitnesses)!= 0 and best_fitnesses[-1] == self.best_genome.fitness:
                count_generations_unchanged += 1
            else:
                count_generations_unchanged = 0
                
            best_fitnesses.append(self.best_genome.fitness)
            mean = sum(fitnesses) / len(fitnesses)
            mean_fitnesses.append(mean)
            
            if write_results:
                print(f"Generation {generation}, best fitness: {self.best_genome.fitness}, mean fitness: {mean} , best genome: {self.best_genome.chromosome}") 
            
        if plot_results:
            self.plot_results(best_fitnesses, mean_fitnesses)
            print(self.best_genome.chromosome)
        return self.best_genome.fitness


    def _resolve_crossover_strategy(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_genome = self.population[0]
        shuffle(self.population) 
        
        new_population = []       
        
        for index in range(0, self.pop_size, 2):            
            parent1, parent2 = self.population[index], self.population[index + 1]            
            child1, child2 = parent1.crossover(parent2)
            
            # evaluating fitness
            child1.fitness = self.fitness(child1.chromosome)            
            child2.fitness = self.fitness(child2.chromosome)
                        
            family = [parent1, parent2, child1, child2]
            
            # taking the first two best in the parent/child competition and appending to next generation
            family.sort(key=lambda x: x.fitness, reverse=True)
            new_population.append(family[0])
            new_population.append(family[1])
        
        self.population = copy.copy(new_population)
                

    def plot_results(self, best_fitnesses = [], mean_fitnesses = []):
        """_summary_

        Args:
            best_fitnesses (list, optional): _description_. Defaults to [].
            mean_fitnesses (list, optional): _description_. Defaults to [].
        """
        plt.title('fitness evolution')
        plt.plot(best_fitnesses)
        plt.plot(mean_fitnesses)
        plt.legend(['best', 'avg'])
        plt.show()

    def evaluate_fitnesses(self):
        """
        calculating fitness of individuals
        we maximize fitness function
        :return:
        """        
        for _, genome in enumerate(self.population):
            genome.fitness = self.fitness(genome.chromosome)