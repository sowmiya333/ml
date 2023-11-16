import pandas as pd
import numpy as np
d = pd.read_csv("dataset.csv") 

print(d)
a = np.array(d)[:,:-1] 
print(" The attributes are: ",a)
t = np.array(d)[:,-1] 
print("The target is: ",t) 
def fun(c,t): 
 for i, val in enumerate(t): 
 if val == "Yes": 
 specific_hypothesis = c[i].copy() 
 break 
  
 for i, val in enumerate(c): 
 if t[i] == "Yes": 
 for x in range(len(specific_hypothesis)): 
 if val[x] != specific_hypothesis[x]: 
 specific_hypothesis[x] = '?' 
 else: 
 pass  
 return specific_hypothesis 
print(" The final hypothesis is:",train(a,t))
