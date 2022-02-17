                    
from GeneticAlgorithm.Genome import IGenome
import numpy as np

class BitArrayGenome(IGenome):
    def __init__(self, config):
        """
        TODO:add description
        """        
        self.fitness = 0            
        self.size = config['size']
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
        
    def __str__(self):
        return super().__str__()