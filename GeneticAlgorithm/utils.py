import time
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
               '\\hline'
               + '\nPopulation size & \t Runtime (h:mm:ss.ms) & \t Fitness function & \t Genome crossover \t \\\\ \n' 
               + '\\hline \n'
               + pop_str + ' & \t ' + str(run_time) + ' & \t ' + str(fitness_function.__name__).replace('_', ' ') + ' & \t ' + str(genome_config['crossover']) + '\t \\\\ \n'
               + '\\hline \n'
               + '\n\n') 
    
    if population_size >= POPULATION_SIZE_UPPERBOUND: raise Exception('Error, population size optimum is bigger than ' + str(POPULATION_SIZE_UPPERBOUND))     
    return population_size

def test_ga(fitness_function, genome_config, population_size, nr_of_runs = 20, plot_results = False, trace_measures = False):                
    fitnesses = []
    generations = []    
    runtimes = []
    fitness_evals = []
    for _ in tqdm(range(nr_of_runs)):
        begin_time = time.time()
        ga = GeneticAlgorithm(pop_size = population_size, 
                                    optimum= OPTIMAL_SOLUTION,
                                    genome_config=genome_config,
                                    fitness_function= fitness_function
                                    )
        fitness, nr_generations = ga.resolve(strategy='crossover_strategy', plot_results=plot_results, trace_measures= trace_measures)
        end_time = time.time()
        run_time = end_time - begin_time
        runtimes.append(run_time)
        fitnesses.append(fitness)
        fitness_evals.append(nr_generations * population_size)
        generations.append(nr_generations)
                    
    
    write_file(RESULTS_FILE_NAME, 
               '\\hline'
               + '\nPopulation size & \t Number of runs & \t Fitness function & \t Runtime mean (seconds) (std)  & \t Fitness mean (std) & \t Fitness evaluation mean (std) & \t Generations mean (std) \t \\\\ \n'
               + '\\hline \n'
               + str(population_size) + ' & \t ' + str(nr_of_runs) + ' & \t ' + str(fitness_function.__name__).replace('_', ' ') 
               + ' & \t ' + str(round(np.mean(runtimes), 4)) +  ' (' + str(round(np.std(runtimes),4)) + ') ' + ' & \t '               
               + str(round(np.mean(fitness),4))              +  ' (' + str(round(np.std(fitnesses),4)) + ') ' + ' & \t '
               + str(round(np.mean(fitness_evals),4))        +  ' (' + str(round(np.std(fitness_evals),4)) + ') ' + ' & \t '
               + str(round(np.mean(generations),4))          +  ' (' + str(round(np.std(generations),4)) + ') ' + '\t \\\\ \n'
               + '\\hline \n'
               + '\n\n')                  

    print('Done ... look in the ' + RESULTS_FILE_NAME + ' for the results')