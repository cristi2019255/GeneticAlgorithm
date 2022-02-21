from tqdm import tqdm
from GeneticAlgorithm.GA import GeneticAlgorithm
import datetime
import numpy as np


NUMBER_OF_RUNS = 20    
OPTIMAL_SOLUTION = 40
POPULATION_SIZE_UPPERBOUND = 1280

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

def count_optimal(population_size, genome_config, fitness_function):
    opt_count = 0
                                
    for _ in tqdm(range(NUMBER_OF_RUNS)):            
        ga = GeneticAlgorithm(pop_size = population_size, 
                                        optimum= OPTIMAL_SOLUTION,
                                        genome_config=genome_config,
                                        fitness_function= fitness_function
                                        )
        fitness, _ = ga.resolve(strategy='crossover_strategy')
        if fitness == OPTIMAL_SOLUTION:
                opt_count += 1
                
    return opt_count
    
def find_optimal_population_size(fitness_function, genome_config):    
    print (    '\n\n\nSolving for: '
               + '\nFitness function: ' + str(fitness_function.__name__) 
               + '\nGenome crossover: ' + str(genome_config['crossover'])
               + '\n')
    population_size = 10
    old_population_size = 10
    is_population_size_optimal = False
    bisection = False
    
    begin_time = datetime.datetime.now()            
    while (population_size <= POPULATION_SIZE_UPPERBOUND):
        opt_count = count_optimal(population_size, genome_config, fitness_function)        
        print(f'Population size: {population_size}, old population size: {old_population_size}, nr of optimums {opt_count}/{NUMBER_OF_RUNS}')
                            
        if opt_count >= NUMBER_OF_RUNS - 1:            
            break                
        else: 
            old_population_size = population_size
            population_size *= 2        
    
    if population_size <= POPULATION_SIZE_UPPERBOUND:
        print('Bisection search')
        up = population_size
        low = old_population_size
        while(low < up and int((low + up) / 2) % 10 == 0):       
            population_size = int((low + up) / 2)  
            opt_count = count_optimal(population_size, genome_config, fitness_function)            
            print(f'Population size: {population_size}, nr of optimums {opt_count}/{NUMBER_OF_RUNS}')
                                        
            if opt_count >= NUMBER_OF_RUNS - 1:            
                up = population_size                
            else: 
                low = population_size                
                                                    
    end_time = datetime.datetime.now()
    run_time = end_time - begin_time
    print('Runtime: ' + str(run_time)) 
    pop_str = str(population_size) if population_size <= POPULATION_SIZE_UPPERBOUND else 'FAILED'
    write_file(RESULTS_FILE_NAME, 
               '\nPopulation_size=' + pop_str
               + '\nRuntime: ' + str(run_time) 
               + '\nFitness function: ' + str(fitness_function.__name__) 
               + '\nGenome crossover: ' + str(genome_config['crossover'])
               + '\n\n\n')      
    return population_size

def test_ga(fitness_function, genome_config, population_size, nr_of_runs = 20, plot_results = False, trace_measures = False):                
    fitnesses = []
    generations = []
    begin_time = datetime.datetime.now()            
    for _ in range(nr_of_runs):
        ga = GeneticAlgorithm(pop_size = population_size, 
                                    optimum= OPTIMAL_SOLUTION,
                                    genome_config=genome_config,
                                    fitness_function= fitness_function
                                    )
        fitness, nr_generations = ga.resolve(strategy='crossover_strategy', plot_results=plot_results, trace_measures= trace_measures)
        fitnesses.append(fitness)
        generations.append(nr_generations)
    end_time = datetime.datetime.now()
    run_time = end_time - begin_time
    
    write_file(RESULTS_FILE_NAME, 
               '\nPopulation_size=' + str(population_size) 
               + '\nNumber of runs: ' + str(nr_of_runs)
               + '\nRuntime: ' + str(run_time) 
               + '\nFitness function: ' + str(fitness_function.__name__) 
               + '\nFitness mean: ' + str(np.mean(fitness)) +  ', std: (' + str(np.std(fitnesses)) + ') '
               + '\nGenerations mean: ' + str(np.mean(generations))
               + '\n\n\n')                  

    print('Done ... look in the ' + RESULTS_FILE_NAME + ' for the results')