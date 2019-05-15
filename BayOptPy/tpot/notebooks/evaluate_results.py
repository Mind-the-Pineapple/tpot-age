'''
This scripts define functions that will be used by the different notebooks to
analyse the results
'''
import pandas as pd


def predecessor_generation(results, curr_generation, verbose):
    '''
    Print the predecessor's generation for all the models in a specific generation
    '''
    data_df = pd.DataFrame()
    for key in results['evaluated_individuals'][curr_generation].keys():
        predecessor = ''.join(results['evaluated_individuals'][curr_generation][key]['predecessor'])
        for generation in range(curr_generation-1, -1, -1):
            if predecessor in results['evaluated_individuals'][generation].keys():
                mae = abs(results['evaluated_individuals'][generation][predecessor]['internal_cv_score'])
                data_df = data_df.append({'model': key, 'predecessor': predecessor, 'generation': int(generation),
                                          'mae': mae},
                                 ignore_index=True)
    if verbose:
        print('Current Generation: %d' %curr_generation)
        print('List of repeated models')
        repeated_predecessors = set([x for x in list(data_df['predecessor']) if list(data_df['predecessor']).count(x)>1])
        print(repeated_predecessors)
        print('There are %d repeated predecessors' )
    return data_df

def prepare_df_for_plotly(data_df):
    '''
    Create new columns
    '''
    # Create a new column in the dataframe that contains only the list of models in the predecessor model.
    list_models_predecessor = []
    list_models_curr_model = []
    for idx, _ in data_df.iterrows():
        # Check if the current row contains an ensamble, if yes split them all speratatly
        predecessor = data_df['predecessor'][idx].split('(input_matrix')[0].split('(')
        list_models_predecessor.append(predecessor)
        # Do the same thing for the current model
        model = data_df['model'][idx].split('(input_matrix')[0].split('(')
        list_models_curr_model.append(model)
    data_df['list_models_predecessor'] = list_models_predecessor
    data_df['list_models_curr_model'] = list_models_curr_model
    # Create a column to visualise on plotly
    data_df['visualisation'] = 'Predecessor Generation: ' + data_df['generation'].astype(str) + '<br>' + \
          'Predecessor Models:' + data_df['list_models_predecessor'].astype(str) + '<br>' + \
          'Curr Model list: ' + data_df['list_models_curr_model'].astype(str)
    return data_df

