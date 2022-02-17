from abc import abstractclassmethod
from re import I
import numpy as np


class IGenome:    

    @abstractclassmethod
    def __init__(self):            
        self.chromosome = []
        self.fitness = 0
        raise NotImplementedError
    
    @abstractclassmethod
    def random_initialize(self):            
        raise NotImplementedError
    
    @abstractclassmethod
    def mutate(self):            
        raise NotImplementedError
    
    @abstractclassmethod
    def crossover(self, genome):
        raise NotImplementedError
        
    
    def __str__(self):
        return "Fitness: " + str(self.fitness) + "\n" + "Chromosome: " + str(self.chromosome) + "\n" 