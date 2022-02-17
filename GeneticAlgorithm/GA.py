import copy
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm.BitArrayGenome import BitArrayGenome

class GeneticAlgorithm:            

    def __init__(self,genome_config = {'type': BitArrayGenome, 'size': 40, 'crossover': '2X'}, 
                 pop_size = 100,
                 generations_unchanged = 5, 
                 optimum = 40,
                 fitness_function = lambda x: np.sum(x)):
        
        """        
        :param pop_size: population size                                
        :param genome_config: dictionary with configurations for genome including type, size and crossover
        :param generations_unchanged: used for stoping criteria, if best genome fitness does not improve for a number of generations then stop genetic algorithm
        :param optimum: optimum of the problem, if best genome fitness reaches optimum then stop genetic algorithm
        :param fitness_function: fitness function to be maximized
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
        Initializing genetic algorithm population each individual information structure is initialized with random values        
        & setting the best genome
        """
        self.population = [ self.genome_class(self.genome_config) for _ in range(self.pop_size)]        
        self.best_genome = self.population[0]

    def resolve(self, strategy = 'crossover_strategy', plot_results = False, write_results = False):
        """ Solves the genetic algorithm

        Args:
            strategy (str, optional): Strategy for solving GA. Defaults to 'crossover_strategy'.
            plot_results (bool, optional): Defaults to False.
            write_results (bool, optional): Defaults to False.

        Returns:
            best fitness: best genome fitness value
        """
        
        # choosing solver strategy
        strategies = {'crossover_strategy': self._resolve_crossover_strategy}    
        resolve_one_generation = self._resolve_crossover_strategy
        if strategy in strategies:
            resolve_one_generation = strategies[strategy]            
        
                
        best_fitnesses, mean_fitnesses = [], []        
        count_generations_unchanged = 0
        generation = 0
        
        # generate new population until the stoping criteria is not meet
        while (self.best_genome.fitness != self.optimum and count_generations_unchanged < self.generations_unchanged):        
            
            generation += 1
                        
            self.evaluate_fitnesses()
            
            resolve_one_generation() # updating population with the new generated population                      
            
            fitnesses = [genome.fitness for genome in self.population]
            
            # counting if best fitness not changes
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
        """
         Generating a new population according to crossover only strategy        
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
        """Ploting the fitness evolution other generations

        Args:
            best_fitnesses (list, optional): Defaults to [].
            mean_fitnesses (list, optional): Defaults to [].
        """
        plt.title('fitness evolution')
        plt.plot(best_fitnesses)
        plt.plot(mean_fitnesses)
        plt.legend(['best', 'avg'])
        plt.show()

    def evaluate_fitnesses(self):
        """
        Calculating fitness of each genome in the population        
        """        
        for _, genome in enumerate(self.population):
            genome.fitness = self.fitness(genome.chromosome)