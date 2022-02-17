
import datetime
from random import shuffle
from GeneticAlgorithm.BitArrayGenome import BitArrayGenome
from GeneticAlgorithm.GA import GeneticAlgorithm
from GeneticAlgorithm.Fitness import TF, B, CO
import numpy as np

NUMBER_OF_RUNS = 20    
GENOME_CONFIG = {'type': BitArrayGenome, 'size': 40}
POPULATION_SIZE = 60 # found with find_optimal_size
MAX_ITERATION = 50  

RESULTS_FILE_NAME = './Output/GA/results.txt'

def write_file(file_name, content):
    """
    write in a file
    :param file_name:
    :param content: the message which is written
    :return:
    """
    file = open(file_name, 'a')
    file.write(content + "\n")
    file.close()
    
def find_optimal_population_size():
    OPTIMAL_SOLUTION = 40
    POPULATION_SIZE_UPPERBOUND = 1280
    
    population_size = 10
    old_population_size = 10
    is_population_size_optimal = False
    bisection = False
    
    while (not is_population_size_optimal and population_size <= POPULATION_SIZE_UPPERBOUND):
        opt_count = 0
        for i in range(NUMBER_OF_RUNS):
            print(f'Run {i}')
            ga = GeneticAlgorithm(pop_size = population_size, 
                                        max_iter=MAX_ITERATION,
                                        genome_config=GENOME_CONFIG,
                                        fitness_function= CO
                                        )
            fitness = ga.resolve(strategy='crossover_strategy')
            if fitness == OPTIMAL_SOLUTION:
                opt_count += 1
        
        print(f'Population size: {population_size}, old population size: {old_population_size}, nr of optimums {opt_count}/{NUMBER_OF_RUNS}')
            
        if opt_count >= NUMBER_OF_RUNS - 1:
            if population_size == old_population_size:
                return population_size
            else:
                bisection = True               
                population_size = int((old_population_size + population_size) / 2)                                
                                                
                if population_size % 10 != 0:
                    return old_population_size                
        else:   
            if bisection:
                return 2 * population_size - old_population_size
            else:         
                old_population_size = population_size
                population_size *= 2        
    
    
        
    return population_size
        
     
def main():   
          
    write_file(RESULTS_FILE_NAME,
                    'Params:\nPopulation_size=' + str(POPULATION_SIZE) +                      
                    '\nmax_iter=' + str(MAX_ITERATION)
                )

    fitnesses = []  
    begin_time = datetime.datetime.now()
    for i in range(NUMBER_OF_RUNS):
        print(f'Run {i}')
        ga = GeneticAlgorithm(pop_size = POPULATION_SIZE, 
                                    max_iter=MAX_ITERATION,
                                    genome_config=GENOME_CONFIG,
                                    fitness_function= TF
                                    )
        fitness = ga.resolve(strategy='crossover_strategy')
        fitnesses.append(fitness)
    end_time = datetime.datetime.now()

 
    best = fitnesses[np.argmax(fitnesses)]
    worst = fitnesses[np.argmin(fitnesses)]
    avg = sum(fitnesses)/len(fitnesses)
    write_file(RESULTS_FILE_NAME, 'Best=' + str(best) + '\nAvg=' + str(avg) + '\nWorst=' + str(worst))
    write_file(RESULTS_FILE_NAME, 'Run time for ' + str(NUMBER_OF_RUNS) + ' runs is ' + str(end_time - begin_time)) 
    
    
    print(end_time - begin_time)    
    

if __name__ == '__main__':
    optimal_pop_size = find_optimal_population_size()
    print(optimal_pop_size)
    