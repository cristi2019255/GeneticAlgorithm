from GeneticAlgorithm.BitArrayGenome import BitArrayGenome
from GeneticAlgorithm.Fitness import FITNESS_FUNCTIONS
from GeneticAlgorithm.utils import find_optimal_population_size, test_ga

GENOME_CONFIG = {'type': BitArrayGenome, 'size': 40, 'crossover': '2X'}

def experiments_one_type_crossover():
   for fitness_function in FITNESS_FUNCTIONS:
         population_size = find_optimal_population_size(fitness_function=fitness_function, genome_config= GENOME_CONFIG)       
         test_ga(fitness_function=fitness_function, genome_config=GENOME_CONFIG, population_size = population_size)

def experiments():
   
   """
   Experiments: 2X crossover
   """
   experiments_one_type_crossover()
   """
   Experiments: UX crossover
   """
   GENOME_CONFIG['crossover'] = 'UX'
   experiments_one_type_crossover()

def main():   
   experiments()      
   #test_ga(CO, genome_config= GENOME_CONFIG, population_size= 200, nr_of_runs=1, plot_results= True, trace_measures = True)
     
if __name__ == '__main__':
    main()
    