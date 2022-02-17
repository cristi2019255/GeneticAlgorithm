                    
from GeneticAlgorithm.Genome import IGenome
import numpy as np

class BitArrayGenome(IGenome):
    def __init__(self, config = {'size': 40, 'crossover': '2X'}):
        """
        TODO:add description
        """        
        self.fitness = 0            
        self.config = config
        self.size: int = config['size']
        self.crossover_operator: str = config['crossover']        
        self.random_initialize()        

    def random_initialize(self):
        """
        TODO:add description
        """                
        self.chromosome = np.random.randint(0,2, self.size)
        
    def mutate(self):
        """_summary_ : Bit flip at a random position

        Returns:
            _type_: _description_
        """
        point = np.random.randint(self.size)
        self.chromosome[point] = 1 - self.chromosome[point]        
        
    def crossover(self, genome):
        """
        Produces 2 children genome from 2 parents
        :param self:
        :param genome:
        :return: child1, child2
        """        
        crossover_options = {'1X': self._one_point_crossover, '2X': self._two_point_crossover, 'UX': self._uniform_crossover}
        
        assert(len(self.chromosome) == len(genome.chromosome)) #checking if parents length are equal
        
        offspring1, offspring2 = crossover_options[self.crossover_operator](genome) # appling crossover operator
        
        assert(len(offspring1) == len(offspring2)) #checking if children length are equal
        
        child1, child2 = BitArrayGenome(self.config), BitArrayGenome(self.config)
        child1.chromosome = offspring1
        child2.chromosome = offspring2
        
        return child1, child2
    
    
    def _two_point_crossover (self, genome):
        point1, point2 = np.random.randint(0, len(self.chromosome), 2)        
        if point1 > point2:
            point1, point2 = point2,  point1
        offspring1 = np.hstack((self.chromosome[:point1], genome.chromosome[point1:point2], self.chromosome[point2:]))
        offspring2 = np.hstack((genome.chromosome[:point1], self.chromosome[point1:point2], genome.chromosome[point2:]))
        return offspring1, offspring2
        
    def _uniform_crossover (self, genome):
        probabilities = np.random.rand(self.size)        
        r = 1 / self.size 
        offspring1, offspring2 = [0] * self.size, [0] * self.size
        for i in range(self.size):
            if (probabilities[i] < r):
                offspring1[i], offspring2[i] = genome.chromosome[i], self.chromosome[i]
            else:
                offspring1[i], offspring2[i] = self.chromosome[i], genome.chromosome[i]
        return offspring1, offspring2
        
    
    def _one_point_crossover (self, genome):
        point = np.random.randint(len(self.chromosome))
        return np.hstack((self.chromosome[:point], genome.chromosome[point:])), np.hstack((self.chromosome[:point], genome.chromosome[point:]))   
    
    
    
    def __str__(self):
        return super().__str__()