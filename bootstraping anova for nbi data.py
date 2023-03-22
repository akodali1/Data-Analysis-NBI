#Importing required libraries
import pandas as pd
import pingouin as pg
import numpy as np
import warnings
import math
warnings.filterwarnings('ignore')

#Reading the data
df = pd.read_csv('nebraska_deep_county_with_regions.csv')

#Removing the duplicates
df.drop_duplicates(subset=['structureNumber'], keep='last', inplace=True)

#Choosing the Intervention among subNumberIntervention, deckNumberIntervention and supNumberIntervention
intervention = 'deckNumberIntervention'

#Replacing the int values to strings
df[intervention]=df[intervention].replace([0.0,1.0,2.0,3.0], [0,1,1,1])

#Cleaning the data
df['regions'] = df['regions'].replace(['Region - 2 '], 'Region - 2')
df['regions'] = df['regions'].replace(['Region - 1 '], 'Region - 1')

#Removing the outliers
df2 = df[df.yearBuilt >= 500]
df2 = df2[df2.yearBuilt != 1935] 
counties = [-1, 15, 29, 57, 61, 63, 71, 73, 83, 85, 87, 91, 99, 103, 105, 115, 117, 119, 121, 123, 137, 139, 149, 161, 169, 171, 177]
for i in counties:
    df2 = df2[df2.countyCode != i]
df2=df2[df2.owner != 3 ]
df2=df2[df2.owner != 25]
df2=df2[df2.owner!=-1]

#Filtering the column 'scourCriticalBridges'
df2=df2[df2.scourCriticalBridges != "N"]
df2=df2[df2.scourCriticalBridges != "U"]

#Converting the averageDailyTraffic to integer
df2['averageDailyTraffic']=df2['averageDailyTraffic'].apply(np.int64)
df2=df2[df2.averageDailyTraffic>0]
    
#Feature
feature='averageDailyTraffic'

# Automating for different types of tests
different_tests =['countyCode']
#different_tests =['countyCode','regions','urbanization','owner']


#Conducting Anova and finding mean and std.dev. for the data
anova_results_dict={}
mean_std_results_dict={}
for different_test in different_tests:
    anova_results=pd.DataFrame()
    mean_not_repaired = []
    mean_repaired = []
    std_not_repaired = []
    std_repaired = []
    mean_combined = []
    std_combined = []
    for i in range(0,1000):
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        df5 = pd.DataFrame()
        df6 = pd.DataFrame()
        for category in df2[different_test].unique():
            sample_size=math.floor(400/len(df2[different_test].unique()))
            for j in [0.0,1.0]:
                if len(df2[(df2[different_test] == category) & (df2[intervention] == j)]) >0: 
                    df3=df3.append(df2[(df2[different_test] == category) & (df2[intervention] == j)].sample(n = sample_size,replace=True,random_state=i))
            if len(df2[(df2[different_test] == category) & (df2[intervention] == j)]) >0: 
                df4=df4.append(df2[(df2[different_test] == category) & (df2[intervention] == 0.0)].sample(n = sample_size,replace=True,random_state=i))
                df5=df5.append(df2[(df2[different_test] == category) & (df2[intervention] == 1.0)].sample(n = sample_size,replace=True,random_state=i))
        df6=df4.append(df5)
        #Conducting mean and std. dev.
        mean_not_repaired.append(df4[feature].mean())
        std_not_repaired.append(df4[feature].std())
        mean_repaired.append(df5[feature].mean())
        std_repaired.append(df5[feature].std())
        mean_combined.append(df6[feature].mean())
        std_combined.append(df6[feature].std())
        
        #Performing Anova
        anova = pg.anova(data=df3, dv=intervention, between=feature)
        anova_results=anova_results.append(anova)
    mean_std_results = pd.DataFrame({'mean_not_repaired': mean_not_repaired, 'std_not_repaired': std_not_repaired,
                                     'mean_repaired': mean_repaired, 'std_repaired': std_repaired,
                                     'mean_combined': mean_combined, 'std_combined': std_combined})
    anova_results=anova_results.reset_index()
    del anova_results['index']
    mean_std_results_dict[f'result_{different_test}'] = mean_std_results
    anova_results_dict[f'result_{different_test}'] = anova_results

    #Accessing the Anova results for different tests
    print(anova_results_dict[f'result_{different_test}'])

    #Accessing the mean and std.dev results for different tests
    print(mean_std_results_dict[f'result_{different_test}'])