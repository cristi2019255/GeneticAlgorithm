from GeneticAlgorithm.BitArrayGenome import BitArrayGenome
from GeneticAlgorithm.Fitness import CO, TF_deceptive_linked, TF_deceptive_not_linked, TF_non_deceptive_linked, TF_non_deceptive_not_linked
from GeneticAlgorithm.utils import find_optimal_population_size, test_ga

GENOME_CONFIG = {'type': BitArrayGenome, 'size': 40, 'crossover': '2X'}

def main():   
   #print(find_optimal_population_size(TF_non_deceptive_linked, genome_config= GENOME_CONFIG))       
   test_ga(CO, genome_config= GENOME_CONFIG, population_size= 70)
   #test_ga(CO, genome_config= GENOME_CONFIG, population_size= 200, nr_of_runs=1, plot_results= True, trace_measures = True)
     
if __name__ == '__main__':
    main()
    