import copy
import datetime
from pydoc import resolve
from random import shuffle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from GeneticAlgorithm.Genome import IGenome
from GeneticAlgorithm.BitArrayGenome import BitArrayGenome



class GeneticAlgorithm:            

    def __init__(self,genome_config, pop_size, mutation_rate=0.3, max_iter=10000, elitism_rate=0.2, fitness_function = lambda x: np.sum(x)):
        """
        :param pop_size: population size                                
        :param mutation_rate: mutation rate
        :param max_iter: maximum number of iterations
        :param elitism_rate: elitism rate
        """
        self.probability_wheel = []
        self.pop_size = pop_size        
        self.population = []
        self.mutation_rate = mutation_rate        
        self.max_iter = max_iter
        self.elitism_rate = elitism_rate
        self.genome_class = genome_config['type']
        self.genome_config = genome_config
        self.create_population()
        self.fitness = fitness_function        

    def create_population(self):
        """
        initializing genetic algorithm population
        each individual information structure is initialized with 
        random values
        :return:
        """
        self.population = [ self.genome_class(self.genome_config) for _ in range(self.pop_size)]        

    def resolve(self, strategy = 'crossover_strategy', plot_results = False):
        """_summary_

        Args:
            strategy (str, optional): _description_. Defaults to 'crossover_strategy'.

        Returns:
            _type_: _description_
        """
        strategies = {'elitism': self._resolve_elitism_strategy, 'crossover_strategy': self._resolve_crossover_strategy}    
        resolve_one_generation = self._resolve_crossover_strategy
        if strategy in strategies:
            resolve_one_generation = strategies[strategy]
            
                
        best_fitnesses, mean_fitnesses = [], []
        
        for iter in tqdm(range(self.max_iter)):
            
            self.evaluate_fitnesses()
            
            resolve_one_generation() # updating population                        
            
            fitnesses = [genome.fitness for genome in self.population]
            best_fitnesses.append(self.best_genome.fitness)
            mean = sum(fitnesses) / len(fitnesses)
            mean_fitnesses.append(mean)
            
            tqdm.write(f"Iteration {iter}, best fitness: {self.best_genome.fitness}, mean fitness: {mean} , best genome: {self.best_genome.chromosome}") #, best genome: {self.best_genome.chromosome}
            
        if plot_results:
            self.plot_results(best_fitnesses, mean_fitnesses)
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
            child1, child2 = self.reproduce(parent1, parent2, crossover=self.two_point_crossover)
            
            # evaluating fitness
            child1.fitness = self.fitness(child1.chromosome)            
            child2.fitness = self.fitness(child2.chromosome)
                        
            family = [parent1, parent2, child1, child2]
            
            # taking the first two best in the parent/child competition and appending to next generation
            family.sort(key=lambda x: x.fitness, reverse=True)
            new_population.append(family[0])
            new_population.append(family[1])
        
        self.population = copy.deepcopy(new_population)
                
        

    def _resolve_elitism_strategy(self):
        """
        The genetic algorithm itself ...
        Tries to maximize the fitness function by generating individuals
        :return: best individual found
        """              
        self.evaluate_probability_wheel()

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_genome = self.population[0]
        new_population = self.population[:int(self.elitism_rate * self.pop_size)]
        shuffle(self.population)

        for _ in range(int((1 - self.elitism_rate) * self.pop_size)):
            parent1 = self.select_probability_wheel()
            parent2 = self.select_probability_wheel()
            child1, child2 = self.reproduce(parent1, parent2, crossover=self.two_point_crossover)

            r1, r2 = np.random.random(), np.random.random()
            if (r1 < self.mutation_rate):
                    child1.mutate()
            if (r2 < self.mutation_rate):
                    child2.mutate()
                    
            new_population.append(child1)
            new_population.append(child2)
            
        self.population = copy.deepcopy(new_population)


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

    def reproduce(self, parent1: IGenome, parent2: IGenome, crossover):
        """
        Produces a child genome from 2 parents
        :param parent1:
        :param parent2:
        :return: child1, child2
        """        
        assert(len(parent1.chromosome) == len(parent2.chromosome)) #checking if parents length are equal
        
        offspring1, offspring2 = crossover(parent1, parent2) # appling crossover operator
        
        assert(len(offspring1) == len(offspring2)) #checking if children length are equal
        
        child1, child2 = self.genome_class(self.genome_config), self.genome_class(self.genome_config)
        child1.chromosome = offspring1
        child2.chromosome = offspring2
        
        return child1, child2


    def one_point_crossover(self, parent1: IGenome, parent2: IGenome):        
        """_summary_

        Args:
            parent1 (IGenome): _description_
            parent2 (IGenome): _description_

        Returns:
            _type_: _description_
        """
        point = np.random.randint(len(parent1.chromosome))
        return np.hstack((parent1.chromosome[:point], parent2.chromosome[point:])), np.hstack((parent1.chromosome[:point], parent2.chromosome[point:]))   
    
    def two_point_crossover(self, parent1: IGenome, parent2: IGenome):        
        """_summary_

        Args:
            parent1 (IGenome): _description_
            parent2 (IGenome): _description_

        Returns:
            _type_: _description_
        """
        point1, point2 = np.random.randint(0, len(parent1.chromosome), 2)        
        if point1 > point2:
            point1, point2 = point2,  point1
        offspring1 = np.hstack((parent1.chromosome[:point1], parent2.chromosome[point1:point2], parent1.chromosome[point2:]))
        offspring2 = np.hstack((parent2.chromosome[:point1], parent1.chromosome[point1:point2], parent2.chromosome[point2:]))
        return offspring1, offspring2

    def evaluate_fitnesses(self):
        """
        calculating fitness of individuals
        we maximize fitness function
        :return:
        """        
        for _, genome in enumerate(self.population):
            genome.fitness = self.fitness(genome.chromosome)


    def select_probability_wheel(self):
        """
        :return:
        """
        if len(self.probability_wheel) == 0:
            return self.population[np.random.randint(0, self.pop_size - 1)]
        return self.probability_wheel[np.random.randint(0, len(self.probability_wheel) - 1)]

    def evaluate_probability_wheel(self):
        """
        generating probability wheel for "roulette selection"
        :return:
        """
        total_fitness = sum(genome.fitness for genome in self.population)

        for genome in self.population:
            if genome.fitness != 0 and total_fitness != 0:
                genome.percent = int(1 + (genome.fitness / total_fitness) * 100)
            else:
                genome.percent = 1

        self.probability_wheel = []
        for genome in self.population:
            for _ in range(genome.percent):
                self.probability_wheel.append(genome)
        shuffle(self.probability_wheel)