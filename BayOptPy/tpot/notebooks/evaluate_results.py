'''
This scripts define functions that will be used by the different notebooks to
analyse the results
'''
import pandas as pd
from plotly import tools
import plotly.graph_objs as go

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

def plot_plotly(results, n_plots, curr_gen_idx, ex_plot, subplots):
    # Create the same plot for other 5 generations and put the next to
    # each other
    data = []
    rows = 2
    columns = 5

    fig = tools.make_subplots(rows=rows, cols=columns,
    subplot_titles=tuple(['Gen %d'%x for x in
                          range(1 + n_plots*ex_plot,
                                n_plots + n_plots * ex_plot + 1)]))

    for i in range(rows):
        for j in range(columns):
            curr_gen = subplots[curr_gen_idx] # Load the data for the current generation
            data_df = predecessor_generation(results, curr_gen, verbose=False)
            # Add additional columns for visualisation
            data_df = prepare_df_for_plotly(data_df)
            sp = go.Scatter(y = data_df['mae'],
                          mode='markers',
                          text=data_df['visualisation'],
                          hoverinfo = 'text',
                          marker=dict(
                                size=16,
                                color=data_df['generation'], #set color equal to a variable
                         colorscale='Viridis',
                         cmin=0,
                         cmax = 100),
                         showlegend=False)
            fig.append_trace(sp, i+1, j+1)
            curr_gen_idx +=1


    fig['layout'].update(height=600, width=1000, title='Multiple Generations')
    return fig, curr_gen_idx
