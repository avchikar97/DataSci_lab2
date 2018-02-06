# In[12]:

## Programming Question 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def retrieve_file(XXXX):
    X_string = str(XXXX)
    path = ".\\Names\\" + "yob" + X_string + ".txt"
    result = pd.read_table(path, sep=',', header=None)
    return result

# input k and XXXX, returns the top k names from year XXXX
def TopKNames(k, XXXX):
    df = retrieve_file(XXXX)
    df = df.groupby(0).agg(sum) #groups by name then aggregates/sums them
    sorted = df.sort_values(by=2, ascending=False) #sorts by frequency
    print(sorted[:k])
    return sorted.nlargest(n=k, columns=2)

# input Name returns the frequency for men and women of the name Name
def NameFreq(Name, XXXX):
    df = retrieve_file(XXXX)
    rows, columns = df.shape
    result = pd.DataFrame()
    for row in range(rows):
        if(df[0][row] == Name):
            print(Name + "(" + str(df[1][row]) + ")" + " has frequency " + str(df[2][row]))
            result = result.append(df.iloc[row], ignore_index = True)
    return result

# input Name = name to search for, XXXX = year, bPrint = boolean print (true = print, false = don't print) returns the relative frequency for men and women of the name Name
def NameRelFreq(Name, XXXX, bPrint):
    df = retrieve_file(XXXX)
    rows, columns = df.shape
    result = pd.DataFrame()
    for row in range(rows):
        if(df[0][row] == Name):
            if(bPrint):
                print(Name + "(" + str(df[1][row]) + ")" + " has relative frequency " + str(np.divide(df[2][row], rows)))
            result = result.append(df.iloc[row], ignore_index = True)
    result_rows, result_cols = result.shape
    if(not result.empty):
        extra = np.array([np.divide(result[2][0], rows)], dtype=float)
    if(result_rows > 1):
        extra = np.append(extra, [np.divide(result[2][1], rows)])
    if(not result.empty):
        result[3] = extra
        result.drop(2, axis=1, inplace=True)
        result.columns=[0, 1, 2]
        result = result.sort_values(2, ascending=False).reset_index(drop=True)
    if(bPrint):
        print(result)
    return result

# Outputs names that became more popular for the other gender in XXXX vs. YYYY. Does not check YYYY vs. XXXX. XXXX and YYYY can be any year between and including 1880 and 2015
def PopularityShift(XXXX=1880, YYYY=2015):
    result = []
    result = np.array(result)
    df0 = retrieve_file(XXXX)
    df1 = retrieve_file(YYYY)
    dup_df0 = df0.duplicated(subset=0, keep='first')
    for row in range(dup_df0.size):
        if(dup_df0[row]):
            name = df0[0][row]
            if(not (NameRelFreq(name, YYYY, False).empty)): # if the name exists in YYYY
                test0 = NameRelFreq(name, XXXX, False)
                #print(test0)
                test1 = NameRelFreq(name, YYYY, False)
                #print(test1)
                if(test0[1][0] != test1[1][0]): # if the higher relative frequency of name has a different gender in year YYYY than in year XXXX
                    result = np.append(result, [name])
    print("Names that shifted gender popularity between " + str(XXXX) + " and " + str(YYYY) + ": " , result)
    return result

TopKNames(10, 1980)
NameFreq("Michael", 1980)
NameRelFreq("Michael", 1980, True)
PopularityShift(1881, 2014)


