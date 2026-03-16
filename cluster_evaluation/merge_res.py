# avereging results of vsv file 

import os
import pandas as pd

#importing the csv files
df1 = pd.read_csv('master_evaluation_results_FINALl.csv')
df2 = pd.read_csv('master_evaluation_results_FINAL_young.csv')

#creaing a new dataframe which is the avg of the two dataframes using the model as the key 
df_avg = pd.DataFrame()
df_avg['Model'] = df1['Model']
df_avg['Flip (Ext)'] = (df1['Flip (Ext)'] + df2['Flip (Ext)']) / 2
df_avg['sFID'] = (df1['sFID'] + df2['sFID']) / 2
df_avg['L1'] = (df1['L1'] + df2['L1']) / 2
df_avg['L1.5'] = (df1['L1.5'] + df2['L1.5']) / 2
df_avg['L2'] = (df1['L2'] + df2['L2']) / 2

#saving the new dataframe to a csv file
df_avg.to_csv('master_evaluation_results_FINAL_avg.csv', index=False)
print("Averaged results saved to master_evaluation_results_FINAL_avg.csv")



# plotting athe re