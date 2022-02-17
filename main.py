
import datetime
from random import shuffle
from GeneticAlgorithm.BitArrayGenome import BitArrayGenome
from GeneticAlgorithm.GA import GeneticAlgorithm
from GeneticAlgorithm.Fitness import TF, B, CO
import numpy as np


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
    
def main():  
    NUMBER_OF_RUNS = 20
    
    GENOME_CONFIG = {'type': BitArrayGenome, 'size': 40}
    POPULATION_SIZE = 50
    MAX_ITERATION = 50  

    RESULTS_FILE_NAME = './Output/GA/results.txt'
    
    
          
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
    main()
    