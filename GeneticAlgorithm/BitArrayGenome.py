from copy import copy
from GeneticAlgorithm.IGenome import IGenome
import numpy as np

class BitArrayGenome(IGenome):
    def __init__(self, config = {'size': 40, 'crossover': '2X'}):
        """
        Args:
            config (dict, optional): configuration of genome including size and crossover type . Defaults to {'size': 40, 'crossover': '2X'}.
        """
        self.fitness = 0            
        self.config = config
        self.size: int = config['size']        
        crossover_options = {'1X': self._one_point_crossover, '2X': self._two_point_crossover, 'UX': self._uniform_crossover}
        self.crossover_operator = crossover_options[config['crossover']]

    def random_initialize(self):
        """
            _summary_ : Initialize a single genome with an array of bits of length size
        """
        self.chromosome = np.random.randint(0, 2, self.size)
        
    def mutate(self):
        """
            _summary_ : Bit flip at a random position        
        """
        point = np.random.randint(self.size)
        self.chromosome[point] = 1 - self.chromosome[point]        
        
    def crossover(self, genome):
        """_summary_: Produces 2 children genome from 2 parents        

        Args:
            genome (BitArrayGenome): the other parent
            
        Returns:
            BitArrayGenome, BitArrayGenome : children after crossover
        """                
        assert(len(self.chromosome) == len(genome.chromosome)) #checking if parents length are equal
        
        offspring1, offspring2 = self.crossover_operator(genome) # appling crossover operator
        
        assert(len(offspring1) == len(offspring2)) #checking if children length are equal
        
        child1, child2 = BitArrayGenome(config = self.config), BitArrayGenome(config = self.config)        
        child1.chromosome = offspring1
        child2.chromosome = offspring2
        
        return child1, child2
    
    
    def _two_point_crossover (self, genome):
        """ Two point crossover            
            choosing two random points and swap parents parts
        Args:
            genome (BitArrayGenome): the second parent for the crossover

        Returns:
            list({0,1}),list({0,1}): children offsprings
        """
        point1, point2 = np.random.randint(0, len(self.chromosome), 2)        
        if point1 > point2:
            point1, point2 = point2,  point1
        offspring1 = np.hstack((self.chromosome[:point1], genome.chromosome[point1:point2], self.chromosome[point2:]))
        offspring2 = np.hstack((genome.chromosome[:point1], self.chromosome[point1:point2], genome.chromosome[point2:]))
        return offspring1, offspring2
        
    def _uniform_crossover (self, genome):
        """ Uniform crossover            

        Args:
            genome (BitArrayGenome): the second parent for the crossover

        Returns:
            list({0,1}),list({0,1}): children offsprings
        """
        
        offspring1, offspring2 = np.copy(self.chromosome), np.copy(genome.chromosome)
        probabilities = np.random.rand(self.size) < 0.5
        offspring1[probabilities] = genome.chromosome[probabilities]
        offspring2[probabilities] = self.chromosome[probabilities]    
        return offspring1, offspring2
        
    
    def _one_point_crossover (self, genome):
        """ One point crossover
            choosing a random point and swap parents parts

        Args:
            genome (BitArrayGenome): the second parent for the crossover

        Returns:
            list({0,1}),list({0,1}): children offsprings
        """
        point = np.random.randint(len(self.chromosome))
        return np.hstack((self.chromosome[:point], genome.chromosome[point:])), np.hstack((self.chromosome[:point], genome.chromosome[point:]))   
    
    
    
    def __str__(self):
        """ Converting to string representation

        Returns:
            str : string representation of the genome
        """
        return super().__str__()