# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 13:02:55 2023

@author: sahana muralidaran (21076516)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import itertools as itr
from scipy.optimize import curve_fit

def tablefunc(filename):
    """
    This function takes the file name as the input and returns two tables
    1. years as columns (country_table)
    2. countries as columns (year_table)
    """
    data= pd.read_csv(filename,skiprows=4)
    data=data.dropna(how='all', axis='columns')
    #getting a list of column names that are numbers
    year = [val for val in list(data.columns.values) if val.isdigit()]
    #setting the index as country names
    data= data.set_index([pd.Index(data['Country Name'])])
    country_table= data[year]
    year_table = country_table.transpose()
    return(country_table,year_table)

def create_table(filenames):
    """
    This function creates a table by combining data from different datasets,
    where each dataset has the values of one attribute(as mentioned in filenames)
    input- filename (list of dataframe names which have different attributes)
    output- merged table with attribute values for the year 2019
    """
    for ind,filename in enumerate(filenames):
        filename_table=pd.read_csv(filename+'.csv',skiprows=4)
        filename_table= filename_table[['Country Name','Country Code','2019']]
        filename_table = filename_table.rename(columns={'2019': filename})
        if ind==0:
            merged=filename_table
        else:
            merged= pd.merge(merged,filename_table)
    classes= pd.read_csv('classes.csv')
    classes=classes[['Country Code','IncomeGroup']]
    merged= pd.merge(merged,classes)
    merged=merged.dropna(axis='rows')
    return merged
    
def heatmap(table):
    """
    This function generates the heatmap for the input table
    """
    corr= table.corr(method='pearson')
    plt.figure(figsize=(14,12))
    #plotting the heatmap
    sns.heatmap(corr, annot = True);
    plt.savefig('heatmap')
    plt.show()


def scatter_matrix(table):
    """
    This function generates the scatter matrix plot for the input table
    """
    sns.set(style="ticks")
    sns.pairplot(table, hue="IncomeGroup")
    plt.savefig('scatter_matrix')
    plt.show()

def subplots(filenames):
    for filename in filenames:
        filename_table=pd.read_csv(filename+'.csv',skiprows=4)
        filename_table=filename_table.dropna(how='all', axis='columns')
        countries= ['Sudan','Madagascar','India','Bangladesh','Brazil','Mexico',
                    'United Arab Emirates','Switzerland']
        years=['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
               '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
        year= years[-1::-2]
        year_2= year[-1::-1]
        filename_table= filename_table.set_index([pd.Index(filename_table['Country Name'])])
        country_table= filename_table[year_2]
        year_table = country_table.transpose()
        fig, axs = plt.subplots(2, 4, figsize=(18, 10))
        for ind,name in enumerate(countries): #loop for plotting the 8 subplots
            plt.subplot(2,4,ind+1)
            plt.plot(year_table[name],label=name)
            plt.legend()
            plt.xticks(rotation=45.0)
            plt.title(filename)
        plt.savefig('subplots8'+ filename)
        plt.show()

def norm(array):
    """ 
    Returns array normalised to [0,1]. 
    Array can be a numpy array
    or a column of a dataframe
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df

def clustering(x,y):
    # extract columns for fitting
    df_fit = merged[[x,y]].copy()
    # normalise dataframe and inspect result
    # normalisation is done only on the extract columns.
    #.copy() prevents changes in df_fit to affect merged.
    #This make the plots with the original measurements
    df_fit = norm_df(df_fit)
    print(df_fit.describe())
    print()
    for ic in range(2, 7):
    # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df_fit)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(df_fit, labels))
    # Plot for four clusters
    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    print(cen)
    plt.figure(figsize=(6.0, 6.0))
    # Individual colours can be assigned to symbols. 
    # The label l is used to select the l-th number from the colour table.
    plt.scatter(df_fit[x], df_fit[y], c=labels, cmap="Accent")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    for ic in range(4):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("cluster")
    plt.savefig('cluster'+ x + y)
    plt.show()

def func(x,a,b,c):
    """function to calculate the error limits"""
    return a * np.exp(-(x-b)**2 / c)

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))  
    pmix = list(itr.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y) 
    return lower, upper

def expoFunc(x,a,b):
    """exponential function"""
    return a**(x+b)

def curvefit(filename):
    df= pd.read_csv(filename,skiprows=4)
    columns=['Country Name','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
             '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    df=df[columns]
    years= [val for val in columns if val.isdigit()]
    df=df.loc[df['Country Name'] == 'Bangladesh']
    plt.scatter(years,df[years].values[0])
    plt.title('Scatter plot before curve fitting')
    plt.ylabel('gdp')
    plt.xlabel('years')
    plt.xticks(rotation=45.0)
    plt.show()
    x_data=[]
    for i in years:
        x_data.append(int(i))
    y_data = df[years].values[0]
    popt, pcov = curve_fit(expoFunc,x_data,y_data,p0=[1,0])
    a_opt, b_opt = popt
    y_mod = expoFunc(x_data,a_opt,b_opt)
    '''plot for scattering after fitting the curve'''
    plt.scatter(years,y_data)
    plt.plot(years,y_mod,color = 'r')
    plt.title('Bangladesh-GDP curve fitting')
    plt.ylabel('gdp')
    plt.xlabel('years')
    plt.xticks(rotation=45.0)
    plt.savefig("curvefit.png")
    plt.show()

filenames= ['co2_per_capita','gdp','renew_energy_percent',
            'under5_mortality','unemployment','urban_population',
            'access_electricity']
merged= create_table(filenames)
print(merged)
heatmap(merged)
scatter_matrix(merged)
subplots(filenames)
clustering('gdp','under5_mortality')
clustering('gdp','renew_energy_percent')
clustering('renew_energy_percent','under5_mortality')
curvefit('gdp.csv')
