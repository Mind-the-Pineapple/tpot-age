# This script analysis some diagnostics from tpot
import os
import joblib
import matplotlib.pyplot as plt

def print_predecessor_per_generation(results, start_gen, stop_gen):
    '''
    This function prints all predecessors per generation.
    This is very hard to visualise
    inputs:
    -------
        results: Dictionary with the model definition per generation
        start_gen: first generation to take into account (default 2)
        stop_gen: last generation to take into account (default100)_
    '''
    if start_gen > stop_gen:
        ValueError('Passed start generation is bigger than stop generation')

    for generation in range(stop_gen, start_gen, -1):
        # ignore the first generation, because all models come from ROOT
        print(generation)
        for key in results['evaluated_individuals'][generation].keys():
            predecessor = results['evaluated_individuals'][generation][key]['predecessor']
            print(predecessor)
        print(' ')

def print_number_of_sucessful_models_per_generation(results, start_gen, stop_gen):
    '''
    This funciton prints then umber of sucessfully evaluated models per generation
    inputs:
    -------
        results: Dictionary with the model definition per generation
        start_gen: first generation to take into account (default 2)
        stop_gen: last generation to take into account (default100)_
    '''
    for generation in range(stop_gen, start_gen, -1):
        n_evaluated_models = len(results['evaluated_individuals'][generation])
        print('Generation %d: %d were evaluated' %(generation,
                                                   n_evaluated_models))

def print_predecessor_generation(results, curr_generation):
    '''
    Print the predecessor's generation for all the models in a specific generation
    '''
    list_predecessors = []
    list_generations = []
    list_mae = []
    for key in results['evaluated_individuals'][curr_generation].keys():
        predecessor = ''.join(results['evaluated_individuals'][curr_generation][key]['predecessor'])
        print(predecessor)
        list_predecessors.append(predecessor)
        for generation in range(curr_generation-1, -1, -1):
            if predecessor in results['evaluated_individuals'][generation].keys():
                list_generations.append(generation)
                list_mae.append(abs(results['evaluated_individuals'][generation][predecessor]['internal_cv_score']))
                print('Found predecessor in %d generation' %generation)
                print('')

    # Check if there is any predecessor that is repeated in this generation
    print('List of repeated models')
    repeated_predecessors = set([x for x in list_predecessors if list_predecessors.count(x)>1])
    print(repeated_predecessors)
    print('There are %d repeated predecessors' %len(repeated_predecessors))
    return list_predecessors, list_generations, list_mae

def plot_predecessor_generation(curr_gen, list_predecessors, list_generation,
                                list_mae, root_path):
    '''
    '''
    plt.figure()
    plt.scatter(range(len(list_mae)), list_mae, c=list_generation)
    plt.savefig(os.path.join(root_path, 'generation_%d.png' %curr_gen))


# Load sample tpot analysis
root_path = '/code/BayOptPy/tpot/Output/random_seed/100_generations/random_seed_020'
results_path = 'tpot_BANC_freesurf_vanilla_combi_100gen.dump'
results = joblib.load(os.path.join(root_path, results_path))

# Plot list of predecessor for all generations.
# Note: This is a bit hard to interprete
# print_predecessor_per_generation(results, 2, 100)

# # Take a model predecessor from the last generation
# model = 'Ridge(input_matrix, Ridge__alpha=1.0, Ridge__random_state=42)'
# # Find in which generation this model predecessor was defined
# for generation in range(100-1, -1, -1):
#     if model in results['evaluated_individuals'][generation].keys():
#         print('Found predecessor model in %d  generation' %generation)

print_predecessor_per_generation(results, 97, 99)

# Print number of sucessfully evaluated models per generation
print_number_of_sucessful_models_per_generation(results,0, 100)

# Take one generation into the analysis and look what are the predecessor's from
# all the models in the current passed generation
curr_gen = 50
list_predecessors, list_generations, list_mae = print_predecessor_generation(results, curr_gen)
plot_predecessor_generation(curr_gen, list_predecessors, list_generations,
                            list_mae, root_path)
