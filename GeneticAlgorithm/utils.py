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
        
def find_optimal_population_size(fitness_function, genome_config):    
    population_size = 10
    old_population_size = 10
    is_population_size_optimal = False
    bisection = False
    
    begin_time = datetime.datetime.now()            
    while (not is_population_size_optimal and population_size <= POPULATION_SIZE_UPPERBOUND):
        opt_count = 0
        for i in range(NUMBER_OF_RUNS):
            print(f'Run {i}')
            ga = GeneticAlgorithm(pop_size = population_size, 
                                        optimum= OPTIMAL_SOLUTION,
                                        genome_config=genome_config,
                                        fitness_function= fitness_function
                                        )
            fitness, _ = ga.resolve(strategy='crossover_strategy') # , write_results=True
            if fitness == OPTIMAL_SOLUTION:
                opt_count += 1
        
        print(f'Population size: {population_size}, old population size: {old_population_size}, nr of optimums {opt_count}/{NUMBER_OF_RUNS}')
            
        if opt_count >= NUMBER_OF_RUNS - 1:
            if population_size == old_population_size:
                is_population_size_optimal = True                
            else:
                bisection = True               
                population_size = int((old_population_size + population_size) / 2)                                
                                                
                if population_size % 10 != 0:
                    is_population_size_optimal = True
                    population_size = old_population_size                
        else:   
            if bisection:
                is_population_size_optimal = True
                population_size = 2 * population_size - old_population_size
            else:         
                old_population_size = population_size
                population_size *= 2        
    
    end_time = datetime.datetime.now()
    run_time = end_time - begin_time
    print(run_time) 
    write_file(RESULTS_FILE_NAME, 
               '\nPopulation_size=' + str(population_size) 
               + '\nRuntime: ' + str(run_time) 
               + '\nFitness function: ' + str(fitness_function.__name__) 
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