# In[10]:

## Programming Question 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin

## Questions
# Which one is more outlying?
# I would say the point at (-1, 1) is more of an outlier because it is farther away from the best fit line that the rest of the data seems to follow

# Propose a transformation:
# (sqrt(cov(input=Y)))^-1 <- transformation matrix
# Printed below

# Justify your choice of transformation
# Y = QZ <- we're after Z
# Y ~ N(mu, covariance matrix)
# var(Y) = var(QZ) = QZQ^T = covariance matrix
# If we say Z = identity matrix, we can just take the square root of the covariance matrix of Y to get Q
# (Q^-1)Y = Z

# Initial scatter plot
df = pd.read_table('DF2', sep=',')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.plot.scatter(x=0, y=1)
plt.xlabel("0th column")
plt.ylabel("1st column")
plt.title("Original data")
plt.show()

# Transformed scatter plot to show distance of each point from center. Shows that point at (-1, 1) is more of an outlier than (5.5, 5)
yT = df.transpose()
outliersT = np.array([[-1, 5.5], [1, 5]])
covariance = np.cov(m=yT, rowvar=True)
Q = splin.sqrtm(A=covariance)
Q_inverse = splin.inv(a=Q)
print("Transformation Q = ", Q_inverse)
outliers_transform = np.dot(Q_inverse, outliersT)
outliersData_transform = pd.DataFrame(outliers_transform.transpose())
zT = np.dot(Q_inverse, yT)
z = pd.DataFrame(zT.transpose())
z.plot.scatter(x=0, y=1)
plt.xlabel("Transformed 0th column")
plt.ylabel("Transformed 1st column")
plt.title("Transformed data")
plt.show()

print("Below are the two outlying points. The 0th row is formerly (-1, 1). The 1st row is formerly (5.5, 5)")
outliersData_transform.head()

