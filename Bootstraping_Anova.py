# Importing required libraries
import pandas as pd
import numpy as np
import math
import numpy as np
import seaborn as sns
from scipy.stats import f
from scipy.stats import spearmanr
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import warnings

warnings.filterwarnings('ignore')

def sns_heatmap(table):
    """
    Prints heatmap
    """
    sns.heatmap(table, annot=True, cmap='Blues')
    plt.show()

def read_clean_data(filename):
    """
    Performs reading and cleaning of the dataset
        - Drop duplicates
        - Choose intervention
        - Datatype conversion
        - String formating

    Args:
        - filename: string

    Return:
        - df: dataframe
    """
    #Reading the data
    df = pd.read_csv(filename)

    #Removing the duplicates
    df.drop_duplicates(subset=['structureNumber'], keep='last', inplace=True)

    #Choosing the Intervention among subNumberIntervention, deckNumberIntervention and supNumberIntervention
    intervention = 'deckNumberIntervention'

    #Replacing the int values to strings
    df[intervention]=df[intervention].replace([0.0,1.0,2.0,3.0], [0,1,1,1])

    #Cleaning the data
    df['regions'] = df['regions'].replace(['Region - 2 '], 'Region - 2')
    df['regions'] = df['regions'].replace(['Region - 1 '], 'Region - 1')

    df = df[df.yearBuilt >= 500]
    df = df[df.yearBuilt != 1935]
    counties = [-1,
                15,
                29,
                57,
                61,
                63,
                71,
                73,
                83,
                85,
                87,
                91,
                99,
                103,
                105,
                115,
                117,
                119,
                121,
                123,
                137,
                139,
                149,
                161,
                169,
                171,
                177]

    for i in counties:
        df = df[df.countyCode != i]

    df = df[df.owner != 3 ]
    df = df[df.owner != 25]
    df = df[df.owner !=-1]

    # Filtering the column 'scourCriticalBridges'
    df = df[df.scourCriticalBridges != "N"]
    df = df[df.scourCriticalBridges != "U"]

    # Converting the averageDailyTraffic to integer
    df['averageDailyTraffic'] = df['averageDailyTraffic'].apply(np.int64)
    df['scourCriticalBridges'] = df['scourCriticalBridges'].apply(np.int64)
    df = df[df.averageDailyTraffic > 0]

    return df

def prepare_data(df, stratify_by, group, value):
    """
    Description:
        - sample from all class

    Args:
        - df: dataframe
        - stratify_by: stratify by class
        - group: intervention variable
        - value: feature values

    Return:
       - sampled_df: stratified sample data by group and value
    """

    # Define a function to sample 40 records from each region
    def sample_group(group, s_no=25):

        # Sampling repaired bridges
        repaired_rows = group[group['deckNumberIntervention'] == 0.0]
        repaired_sample = repaired_rows.sample(n=s_no)

        # Sampling non-repaird bridges 
        non_repaired_rows = group[group['deckNumberIntervention'] == 1.0]
        non_repaired_sample = non_repaired_rows.sample(n=s_no,replace=True)

        # Concatenate
        group = pd.concat([repaired_sample,non_repaired_sample], axis=0)
        return group

    # Apply the function to each group and combine the results
    sampled_df = df.groupby(stratify_by, group_keys=False).apply(sample_group)
    sampled_df = sampled_df[[value, group]]
    return sampled_df

def calc_f_statistic(data, grouping_var, dependent_var):
    # Calculate the group means and the overall mean
    group_means = data.groupby(grouping_var)[dependent_var].mean() # individual group mean
    overall_mean = data[dependent_var].mean() #

    # Calculate the sum of squares between groups
    ss_between = np.sum((group_means - overall_mean)**2) * (len(group_means) - 1)

    # Calculate the sum of squares within groups
    ss_within = np.sum((data[dependent_var] - data.groupby(grouping_var)[dependent_var].transform('mean'))**2)

    # Calculate the degrees of freedom
    df_between = len(group_means) - 1
    df_within = len(data) - len(group_means)

    # Calculate the F-statistic
    f_statistic = (ss_between / df_between) / (ss_within / df_within)

    # Calculate the p-value from the F-statistic and degrees of freedom
    p_value = f.sf(f_statistic, df_between, df_within)

    return f_statistic, p_value


def stratified_bootstrap_anova(data,
                               grouping_var,
                               dependent_var,
                               n_bootstrap):

    # Create a list of strata based on the grouping variable
    #strata = data.groupby(grouping_var).apply(lambda x: x.index.values).tolist() 

    # Define a function to calculate the ANOVA F-statistic
    def anova_func(data, group=grouping_var, value=dependent_var):

        # Calculate the group means and the overall mean
        group_means = data.groupby(group)[value].mean()
        overall_mean = data[value].mean()

        # Calculate the sum of squares between groups
        ssb = np.sum((group_means - overall_mean)**2) * (len(group_means) - 1)

        # Calculate the sum of squares within groups
        ssw = np.sum((data[value] - data.groupby(group)[value].transform('mean'))**2)

        # Calculate the degrees of freedom
        dfb = len(group_means) - 1
        dfw = len(data) - len(group_means)

        # Calculate the mean sum of squares between groups and within groups
        msb = ssb / dfb
        msw = ssw / dfw

        # Calculate the F-statistic and p-value
        f_statistic = msb / msw
        p_value = f.sf(f_statistic, dfb, dfw)

        # Calculate eta-squared
        eta_squared = ssb / (ssb + ssw)

        return f_statistic, eta_squared

    # Perform the bootstrap ANOVA with stratification
    f_statistics = []
    eta_squareds = []

    for i in range(n_bootstrap):
        #indices = [np.random.choice(stratum, size=320) for stratum in strata]
        #indices = np.concatenate(indices)
        #sampled_data = data.loc[indices]
        sampled_data = data
        f_stat, eta_sq = anova_func(sampled_data)
        f_statistics.append(f_stat)
        eta_squareds.append(eta_sq)
    observed_f_statistic, _ = calc_f_statistic(data, grouping_var, dependent_var)
    p_value = np.mean(f_statistics >= observed_f_statistic)
    eta_squared = np.mean(eta_squareds)

    return p_value, eta_squared

def compute_rank_correlation(feature_strata_results, stratas):
    """
    Calculate the Spearman rank correlation co-efficient

    Args:
        - feature_strata_results: A list of dataframes

    Return:
        - corr: Co-efficient
        - p-value
    """
    feature_1 = []
    feature_2 = []
    rank_coeffs = []
    p_values = []

    len_feat_strat_res = len(feature_strata_results)

    for index_i in range(len_feat_strat_res):
        for index_j in range(len_feat_strat_res):
            df1 = feature_strata_results[index_i]
            df2 = feature_strata_results[index_j]
            eta_sq_1 = df1.index
            eta_sq_2 = df2.index
            corr, pvalue =  spearmanr(eta_sq_1, eta_sq_2)

            #print(stratas[index_i], stratas[index_j])
            #print(corr, pvalue)

            feature_1.append(stratas[index_i])
            feature_2.append(stratas[index_j])

            rank_coeffs.append(round(corr, 2))
            p_values.append(round(pvalue, 2))


    # Create a sample DataFrame
    datatable = {'feature1': feature_1,
                 'feature2': feature_2,
                 'rank_coeffs': rank_coeffs,
                 'p_values': p_values}

    data = pd.DataFrame(datatable)

    # Create a crosstabulation table of stratas and rank_coeff
    rank_cross_tab = pd.pivot_table(data,
                         values='rank_coeffs',
                         index=data['feature1'],
                         columns=data['feature2'],
                         aggfunc='mean')

    # Create a crosstabulation table of stratas and p-value
    pvalue_cross_tab = pd.pivot_table(data,
                         values='p_values',
                         index=data['feature1'],
                         columns=data['feature2'],
                         aggfunc='mean')

    # Print the crosstabulation table
    return rank_cross_tab, pvalue_cross_tab


def convergence_checker(convergence_results_list, iterations):
    """
    Calculate the Spearman rank correlation co-efficient

    Args:
        - convergence_results_list: A list of dataframes

    Return:
        - corr: Co-efficient
        - p-value
    """
    iteration_1 = []
    iteration_2 = []

    #p_values = []
    len_conv_results = len(convergence_results_list)

    #for index_i in range(len_conv_results):
        #for index_j in range(len_conv_results):
    df1 = convergence_results_list[0]
    df2 = convergence_results_list[1]
    eta_sq_1 = df1.index
    eta_sq_2 = df2.index
    corr, pvalue =  spearmanr(eta_sq_1, eta_sq_2)
    rank_coeffs = round(corr, 2)

            #iteration_1.append(iterations[index_i])
            #iteration_2.append(iterations[index_j])

            #p_values.append(round(pvalue, 2))



    ## Create a sample DataFrame
    #datatable = {'iteration_1': iteration_1,
    #            'iteration_2': iteration_2,
    #            'rank_coeffs': rank_coeffs}#,
    #            #'p_values': p_values}
    #data = pd.DataFrame(datatable)
#
#
    ## Create a crosstabulation table of stratas and rank_coeff
    #rank_cross_tab = pd.pivot_table(data,
    #                    values='rank_coeffs',
    #                    index=data['iteration_1'],
    #                    columns=data['iteration_2'],
    #                    aggfunc='mean')

    # Create a crosstabulation table of stratas and p-value
    #value_cross_tab = pd.pivot_table(data,
    #                    values='p_values',
    #                    index=data['feature1'],
    #                    columns=data['feature2'],
    #                    aggfunc='mean')

    # Print the crosstabulation table
    return rank_coeffs

def run_process(stratas, values, group, df, n=1000):
    """
    Description:
        Runs through the bootstrapping process for all stratas, values, and group
    """
    for strata in stratas:
        features = []
        p_values = []
        effect_sizes = []

        for value in values:
            data = prepare_data(df,
                            stratify_by=strata,
                            group=group,
                            value=value)

            # Perform ANOVA 
            pvalue, eta = stratified_bootstrap_anova(data,
                                              grouping_var=group,
                                              dependent_var=value,
                                              n_bootstrap=n)

            features.append(value)
            p_values.append(pvalue)
            effect_sizes.append(eta)

        # Create a dataframe to save results per strata
        results_df = pd.DataFrame({'feature': features,
                                   'p_value': p_values,
                                   'effect_size': effect_sizes})

        return results_df

def compute_list_spearman_ranking(bootstrap_n_list, bootstrap_results):
    # Initialize the spearmanr coeff list
    rank_coeffs_list = []

    for i in range(len(bootstrap_results) - 1):

        # Get previous and current results
        prev = bootstrap_results[i].index
        curr = bootstrap_results[i+1].index

        # Compute spearman ranking correlation
        corr, pvalue =  spearmanr(prev, curr)

        # Save spearmanr correlation coefficients
        rank_coeffs = round(corr, 2)
        rank_coeffs_list.append(rank_coeffs)

    return bootstrap_n_list, rank_coeffs_list

def plot_line_chart(bootstrap_n_list, rank_coeffs):
    """
    Plot line chart to present X-axis: spearman r coefficients

    Args:
        X-axis:
        Y-axis:

    Returns:
        Line graph that summarizes the trend
        in spearman ranking coeffs with respect
        to bootstrap_n_list
    """
    # Example data
    df = pd.DataFrame({'N': bootstrap_n_list, 'Spearman rank coeffs': rank_coeffs})

    # Create a line trace
    trace = go.Scatter(x=df['N'], y=df['Spearman rank coeffs'], mode='lines')

    # Create a layout
    layout = go.Layout(title='Bootstrap N Vs. Spearman rank coefficients',
                       xaxis=dict(title='Bootstrap N'),
                       yaxis=dict(title='Spearman rank coeffs'))

    # Create a figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the figure
    pio.show(fig)

def main():
    """
    Driver function
    """
    filename = 'nebraska_deep_county_with_regions.csv'

    # Read and clean dataset
    df = read_clean_data(filename)

    # Define stratas, columns, and intervention
    #stratas = ['regions']
    stratas = ['countyCode']
    #stratas = ['regions', 'urbanization', 'countyCode']

    # Features
    values = [
              'yearBuilt',
              'averageDailyTraffic',
              'scourCriticalBridges',
              #'bridgeRoadwayWithCurbToCrub',
              'operatingRating',
              'longitude',
              'latitude',
              'lengthOfMaximumSpan',
              'structureLength',
              'lanesOnStructure',
              'avgDailyTruckTraffic'
    ]

    group = 'deckNumberIntervention'

    # Set start and end
    start_n = 1000
    end_n = 10000

    # Set interval
    interval = 1000

    # Set
    bootstrap_results = []

    # Prepare dataset for each strata
    print("Running for n:", 25)
    bootstrap_n_list = []
    for n in range(start_n, end_n, interval):
        results_df = run_process(stratas,
                             values,
                             group,
                             df,
                             n=n)
        print("printing for n:", n)
        bootstrap_n_list.append(n)
        bootstrap_results.append(results_df)

    bootstrap_n_list,  ranks_coeffs_list = compute_list_spearman_ranking(bootstrap_n_list, bootstrap_results)
    plot_line_chart(bootstrap_n_list[1:], ranks_coeffs_list)

#    ran=[]
#    itr=[]
#    while True:
#        print("Programming started")
#        convergence_results_list = []
#        iterations=[iteration_start,iteration_start+iteration_size]
#        for iteration in iterations:
#            feature_strata_results = []
#
#            # Prepare dataset for each strata
#            for strata in stratas:
#                features = []
#                p_values = []
#                effect_sizes = []
#
#                for value in values:
#                    data = prepare_data(df,
#                                    stratify_by=strata,
#                                    group=group,
#                                    value=value)
#
#                    # Perform ANOVA 
#                    pvalue, eta = stratified_bootstrap_anova(data,
#                                                      grouping_var=group,
#                                                      dependent_var=value,
#                                                      n_bootstrap=iteration)
#
#                    features.append(value)
#                    p_values.append(pvalue)
#                    effect_sizes.append(eta)
#
#                # Create a dataframe to save results per strata
#                results_df = pd.DataFrame({'feature': features,
#                                           'p_value': p_values,
#                                           'effect_size': effect_sizes})
#
#                print("printing results")
#                print(results_df)
#                # Sort by two columns: p_value and effect_size
#                sorted_df = results_df.sort_values(['p_value', 'effect_size'],
#                                           ascending=[True, False])
#
#                # Append all dataframes 
#                feature_strata_results.append(sorted_df)
#
#            # Compute rank correlation
#            #ranks, pvalues = compute_rank_correlation(feature_strata_results, stratas)
#            #sns_heatmap(ranks)
#            #sns_heatmap(pvalues)
#            convergence_results = pd.concat(feature_strata_results, axis=0)
#            convergence_results_list.append(convergence_results)
#            print('Iteration running :',iteration)
#
#        # Compute rank correlation
#        rank= convergence_checker(convergence_results_list, iterations)
#        print('The correlation is: ',rank)
#
#        #storing test values to create a Line chart
#        ran.append(rank)
#        itr.append(iteration_start)
#
#        # Checking convergence
#        if rank >= 0.95:
#            print("Convergence occured at :",iteration_start)
#            break;
#        else:
#            print("Convergence didn't occur at :",iteration_start)
#            print("\n")
#            iteration_start=iteration_start+2*iteration_size
#

if __name__=='__main__':
    main()
