from GeneticAlgorithm.BitArrayGenome import BitArrayGenome
from GeneticAlgorithm.Fitness import TF, CO
from GeneticAlgorithm.utils import find_optimal_population_size, test_ga

GENOME_CONFIG = {'type': BitArrayGenome, 'size': 40, 'crossover': 'UX'}

def main():   
   print(find_optimal_population_size(TF, genome_config= GENOME_CONFIG))       
   #test_ga(TF, genome_config= GENOME_CONFIG, population_size= 1280)
     
if __name__ == '__main__':
    main()
    