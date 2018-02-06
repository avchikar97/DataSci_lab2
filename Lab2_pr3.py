# In[11]:

## Programming Question 3

print("YEAH G")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("YEAH G")
from sklearn import linear_model
#from sklearn import datasets ## imports datasets from scikit-learn
#data = datasets.load_boston() ## loads Boston dataset from datasets library
import math
def standard_error_for_different_error(error_value):

    Beta_values = []
    # Akaash here's the website: https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

    lm = linear_model.LinearRegression()
    for i in range( 0 , 1000):


        Gaussian_Normal_x = np.random.randn(error_value)

        #print(Gaussian_Normal_x)
        Gaussian_Normal_e = np.random.randn(error_value)

        #print(Gaussian_Normal_e)
        # Don't need x since it s just 0's
        y_values = -3 + Gaussian_Normal_e

        # Get a column vector of x, y
        #model = lm.fit(X,y)

        lm.fit(Gaussian_Normal_x.reshape(error_value, 1), y_values)
        # SANITY CHECK:
        #lm.coef_
        #will give an output like:

        #array([ -1.07170557e-01,   4.63952195e-02,   2.08602])
        Beta_values.append(lm.coef_[0])

    standard_dev_d = np.std(Beta_values)

    return standard_dev_d

try:
    beta_hat = None

    error_range = range(5,506, 100)
    error_range.append(150)
    error_range.sort()

    empirical_standard_dev_of_error = []
    #empirical_standard_dev_of_error.append(standard_error_for_different_error(150))

    for error_value in error_range:

        empirical_standard_dev_of_error.append(standard_error_for_different_error(error_value))

    python_plot = plt.plot(error_range, empirical_standard_dev_of_error)
    plt.show()

    print("Standard dev at n=150 is .084596, a value of .15 is within 2 standard deviations of this value. If the value \\n needs to fall somewhere above or below 5-95% respectively then this s insignificant, so 1-2 standard deviatios is ok, this is fine  ")

   # plot 1 / sqrt(n)

    j = 0
    sqrt_error_range = []
    #empirical_standard_dev_of_error = []
    for value in error_range:
        sqrt_error_range.append(1/(np.sqrt(value)))
    #for error_value in sqrt_error_range:

    np_list_standard_dev = np.array(empirical_standard_dev_of_error, dtype=np.float)
    np_list_error_range = np.array(error_range, dtype=np.float)

    np_sqrt_list = np_list_standard_dev / np_list_error_range

    plt.plot(error_range, np_sqrt_list, color='green')
    #python_plot = plt.plot(sqrt_error_range, empirical_standard_dev_of_error)
    plt.show()

    print("Fit is good, the scaled graph follows the same pattern as the non-scaled one")
except Exception as e:
    print("YEAH G" + str(e))


