
# coding: utf-8

# # In silico Voting Experiments
# 
# Implementation of cultures and voting rules
# 
# Code uploaded to the github.

# In[1]:


import pandas as pd
import sklearn
import math
import numpy as np
import random
random.seed(12)

simulation_runs = 1000

n = 5
K = 5

alpha_values = [0, 0.4, 0.7, 1]
beta_values  = [0, 0.3, 0.5, 0.7, 1]

df1 = pd.DataFrame(columns=['beta = '+str(beta) for beta in beta_values])

for alpha in alpha_values:
    line = []
    for beta in beta_values:
        line.append(alpha*beta/2)
    print(line)
    df1.loc[f'alpha = {alpha}'] = line

print('===============')
print(df1)


