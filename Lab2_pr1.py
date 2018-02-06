# In[8]:

## Akaash Chikarmane, Sean Tremblay
## Programming Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

df = pd.read_table('DF1', sep=',')#, header = None)
df = df.drop(df.columns[0], axis=1)
print(df)

#correlated_df = df.corr('pearson')

#print(correlated_df)

#This is part 2
covariance_df = np.cov(df, rowvar=False)
print(covariance_df)

print(" The numbers fit with the plot I got, because the closer the value is to being <= 1 the more correlated they are")

#Part1
pd.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
seaborn.pairplot(df)

plt.show()

mean = [0,0,0]
cov =[[1,.2,0],[.2,1,.5],[0, .5, 1]]
#mean = [0, 0]
#cov = [[1, 0], [0, 100]]

#gives correct covariance matrix
#cov_array = np.cov(np.random.multivariate_normal(mean, cov, size=1000), rowvar=False)

#1000 is too short
covariance_x = range(10,2000,10)
covariance_y_array = []

# Going to check it versus third row second column ,
# should be .5
for i in covariance_x:
    covariance_y = np.cov(np.random.multivariate_normal(mean, cov, i) , rowvar=False)[2][1]
    covariance_y_array.append(covariance_y)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(covariance_x, covariance_y_array)

plt.show()

print("Done")

