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
        self.genome_class = genome_config['type']
        self.genome_config = genome_config
        self.generations_unchanged = generations_unchanged
        self.fitness = fitness_function  
        self.optimum = optimum      
        self._create_population()        

    def _create_population(self):
        """
        Initializing genetic algorithm population each individual information structure is initialized with random values        
        & setting the best genome
        """
        self.population = [ self.genome_class(self.genome_config) for _ in range(self.pop_size)]
        for genome in self.population:
            genome.random_initialize()                

    def resolve(self, strategy = 'crossover_strategy', plot_results = False, write_results = False, trace_measures = False):
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
        
        self.trace_measures = trace_measures
                
        best_fitnesses, mean_fitnesses = [], []      
        
        # measures of population, lists of integers
        schema_trace_nr_members = [] # tracing the schemata number in population  
        schema_trace_fitness = [] # tracing the schemata fitness in population  
        schema_trace_counterpart_fitness = [] # tracing the schemata counterpart fitness in population          
        bit_ones_proportion = [] # tracing bit ones proporiton
        self.errors = [] # tracing errors in selection decisions
        self.correct = [] # tracing correct selection decisions
        
        count_generations_unchanged = 0
        generation = 0
        
        # evaluating fitness
        self._evaluate_fitnesses()
        self.best_genome = max(self.population, key= lambda x: x.fitness)               
        
        # generate new population until the stoping criteria is not meet
        while (self.best_genome.fitness != self.optimum and count_generations_unchanged < self.generations_unchanged):        
            
            generation += 1                                    
            
            if trace_measures:
                n, m1, std1, m2, std2 = self._get_schema_measures()
                schema_trace_nr_members.append(n) # tracing the schemata members number in population
                schema_trace_fitness.append((m1,std1)) # tracing the schemata members fitness in population
                schema_trace_counterpart_fitness.append((m2,std2)) # tracing the schemata counterpart members fitness in population                
                bit_ones_proportion.append(self._get_prop_of_bit_ones()) # tracing the bit ones proportion in population                
                                    
            changes_in_population = resolve_one_generation() # updating population with the new generated population                                              
            
            # counting if best fitness not changes
            if not changes_in_population:
                count_generations_unchanged += 1
            else:
                count_generations_unchanged = 0

            if plot_results:
                fitnesses = [genome.fitness for genome in self.population]                
                best_fitnesses.append(self.best_genome.fitness)
                mean = sum(fitnesses) / len(fitnesses)
                mean_fitnesses.append(mean)
            
            if write_results:
                print(f"Generation {generation}, best fitness: {self.best_genome.fitness}, mean fitness: {mean} , best genome: {self.best_genome.chromosome}") 
            
        if plot_results:
            self.plot_results(best_fitnesses, mean_fitnesses)
            
            if trace_measures:
                self.plot_schemata_analysis(schema_trace_nr_members, schema_trace_fitness, schema_trace_counterpart_fitness)
                self.plot_measures(bit_ones_proportion)
            
            print(self.best_genome.chromosome)
            
        return max(self.population, key = lambda x: x.fitness).fitness, generation
    
    def _resolve_crossover_strategy(self):
        """
         Generating a new population according to crossover only strategy        
        """         
        shuffle(self.population) 
        
        new_population = []       
        correct, errors = 0, 0                
        changes_in_population = False
        
        for index in range(0, self.pop_size, 2):            
            parent1, parent2 = self.population[index], self.population[index + 1]            
            child1, child2 = parent1.crossover(parent2)            
            
            # evaluating fitness
            child1.fitness = self.fitness(child1.chromosome)            
            child2.fitness = self.fitness(child2.chromosome)
                        
            family = [child1, child2, parent1, parent2]            
            if parent1.fitness == child1.fitness or parent1.fitness == child2.fitness:
                family.remove(parent1)
            if parent2.fitness == child2.fitness or parent2.fitness == child1.fitness:
                family.remove(parent2)                
            
            if (child1.fitness > parent1.fitness and child1.fitness > parent2.fitness) or (child2.fitness > parent1.fitness and child2.fitness > parent2.fitness):
                changes_in_population = True
                
            # taking the first two best in the parent/child competition and appending to next generation
            family.sort(key=lambda x: x.fitness, reverse=True)
            
            winner1, winner2 = family[0], family[1]                                                 
            self.population[index] = winner1
            self.population[index + 1] = winner2
            
            if self.best_genome.fitness < winner1.fitness:
                self.best_genome = winner1
            
            if self.trace_measures:
                for i in range(len(parent1.chromosome)):
                    if parent1.chromosome[i] != parent2.chromosome[i]:
                        if winner1.chromosome[i] == 0 and winner2.chromosome[i] == 0:
                            errors += 1
                            break
                        if winner1.chromosome[i] == 1 and winner2.chromosome[i] == 1:
                            correct += 1
                            break
        
        # tracing the nr of correct selection decision and errors
        if self.trace_measures:
            self.correct.append(correct)
            self.errors.append(errors)
        
        return changes_in_population                

    def plot_results(self, best_fitnesses = [], mean_fitnesses = []):
        """Ploting the fitness evolution other generations

        Args:
            best_fitnesses (list, optional): Defaults to [].
            mean_fitnesses (list, optional): Defaults to [].
        """
        plt.title('Fitness evolution')
        plt.plot(best_fitnesses)
        plt.plot(mean_fitnesses)
        plt.legend(['best', 'avg'])
        plt.show()
        
    def plot_measures(self, bit_ones_proportion):
        
        plt.title('Bit ones proportion')      
        plt.plot(bit_ones_proportion, color='blue')
        plt.show()
        
        plt.title('Measures over population')        
        plt.plot(self.correct, color='green')
        plt.plot(self.errors, color = 'red')
        plt.legend(['Correct selection decisions', 'Errors in selection decisions'])
        plt.show()
        
    def _get_schema_measures(self, schemata = (0,0)):
        """Getting number of schema members in the current population

        Args:
            schemata (tuple, optional): (Schemata value, Schemata index). Defaults to (0,0).

        Returns:
            int, float, float, float, float : number of schemata members, fitness mean of schemata members, fitness standard deviation of schemata members,
                                                                          fitness mean of non schemata members, fitness standard deviation of non schemata members,             
        """        
        schema_members = list(filter(lambda x: x.chromosome[schemata[1]] == schemata[0], self.population))
        non_schema_members = list(filter(lambda x: x.chromosome[schemata[1]] != schemata[0], self.population))
        schema_members_fitness = [genome.fitness for genome in schema_members]
        non_schema_members_fitness = [genome.fitness for genome in non_schema_members]                        
        return len(schema_members), np.mean(schema_members_fitness), np.std(schema_members_fitness), np.mean(non_schema_members_fitness), np.std(non_schema_members_fitness)
    
    def _get_prop_of_bit_ones(self):
        """Getting the proportion of bit ones in the population

        Returns:
            float in [0,1]: The proportion of bit ones in population
        """
        nr_bit_ones = sum(genome.fitness for genome in self.population)   
        nr_bits = self.pop_size * self.genome_config['size']
        return nr_bit_ones / nr_bits
        
    def plot_schemata_analysis(self, first_schema_nr, first_schema_fitness, second_schema_fitness):
        """Ploting the schemata information over generations

        Args:
            first_schema_nr (int[]): the array of number of first schema members over generations,
            first_schema_fitness ((float, float)[]): the array of tuples (mean, std) of fitness of first schema members over generations,
            second_schema_fitness ((float, float)[]): the array of tuples (mean, std) of fitness of second schema members over generations,,
            
        """        
        plt.title('Schemata analysis nr of schema members')
        x = np.arange(0, len(first_schema_nr))
        second_schema_trace = [self.pop_size - n for n in first_schema_nr] 
        plt.stackplot(x, second_schema_trace, first_schema_nr, colors=['b', 'r'])    
        plt.plot(x,second_schema_trace, '*', color='black') 
        plt.xticks(range(0,len(first_schema_nr),2)) # setting x axis to show integers          
        plt.legend(['1****...*', '0****...*'])
        plt.show()

        plt.style.use('seaborn-whitegrid')                        
        plt.title('Schemata analysis fitness')
        y2 = [f[0] for f in second_schema_fitness]
        e2 = [f[1] for f in second_schema_fitness]
        plt.errorbar(x, y2, yerr = e2, color='b', fmt='-o', capsize=4)        
        y1 = [f[0] for f in first_schema_fitness]
        e1 = [f[1] for f in first_schema_fitness]
        plt.errorbar(x + 0.2, y1, yerr = e1, color='red', fmt='-o', capsize=4)        
        
        plt.legend(['1****...* fitness', '0****...* fitness'])
        plt.show()
        

    def _evaluate_fitnesses(self):
        """
        Calculating fitness of each genome in the population        
        """        
        for genome in self.population:
            genome.fitness = self.fitness(genome.chromosome)