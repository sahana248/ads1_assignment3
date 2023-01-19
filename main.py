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
from scipy.optimize import curve_fit

#------------------------------------------------------------------------------
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
    """
    This function creates subplots for 8 different countries.
    The input for this function contains a list of dataset names, each 
    representing different attributes.
    In this program this function runs 7 times since filenames contains a 
    list of seven different datasets. Also each time the function runs it
    produces a graph containing 8 subplots for the countries Sudan, Madagascar,
    India, Bangladesh, Brazil, Mexico, United Arab Emirates and Switzerland.
    """
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
    """
    This function plots 4 clusters and their respective centroids
    """
    # extract columns for fitting
    df_fit = merged[[x,y]].copy()
    # normalise dataframe and inspect result
    # normalisation is done only on the extract columns.
    #.copy() prevents changes in df_fit to affect merged.
    #This make the plots with the original measurements
    df_fit = norm_df(df_fit)
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

def expoFunc(x,a,b):
    """exponential function"""
    return a**(x+b)

def curvefit(filename):
    """
    This function takes the input as a filename (gdp.csv) and returns the 
    scatter plot before and after curve fitting. 
    """
    df= pd.read_csv(filename,skiprows=4)
    columns=['Country Name','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
             '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    df=df[columns]
    years= [val for val in columns if val.isdigit()]
    #taking only the data for Bangladesh
    df=df.loc[df['Country Name'] == 'Bangladesh']
    plt.figure(figsize=(8, 6))
    plt.scatter(years,df[years].values[0])
    plt.title('Scatter plot before curve fitting')
    plt.ylabel('gdp')
    plt.xlabel('years')
    plt.xticks(rotation=45.0)
    plt.show()
    #converting the years into integer to facilitate calculations
    x_data=[]
    for i in years:
        x_data.append(int(i))
    y_data = df[years].values[0]
    popt, pcov = curve_fit(expoFunc,x_data,y_data,p0=[1,0])
    a_opt, b_opt = popt
    y_mod = expoFunc(x_data,a_opt,b_opt)
    #plot for scattering after fitting the curve
    plt.figure(figsize=(8, 6))
    plt.scatter(years,y_data)
    plt.plot(years,y_mod,color = 'r')
    plt.title('Bangladesh-GDP curve fitting')
    plt.ylabel('gdp')
    plt.xlabel('years')
    plt.xticks(rotation=45.0)
    plt.savefig("curvefit.png")
    plt.show()

    
def error_range(filename):
    """
    This function will give a scatter plot with curve fitting and
    confidence range.
    Also this function produces and prints a new dataframe which consists 
    of predicted GDP values for future years, along with their respective
    lower and upper limit.
    """
    df= pd.read_csv(filename,skiprows=4)
    columns=['Country Name','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
                 '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    df=df[columns]
    years= [val for val in columns if val.isdigit()]
    #taking only the data for Bangladesh
    df=df.loc[df['Country Name'] == 'Bangladesh']
    #converting the years into integer to facilitate calculations
    x_data=[]
    for i in years:
        x_data.append(int(i))
    y_data = df[years].values[0]
    popt, pcov = curve_fit(expoFunc,x_data,y_data,p0=[1,0])
    a_opt, b_opt = popt
    da, db = np.sqrt(np.diag(pcov))
    y_mod = expoFunc(x_data,a_opt,b_opt)
    lower=expoFunc(x_data, a_opt-da, b_opt-db)
    upper=expoFunc(x_data, a_opt+da, b_opt+db)
    #plotting the curve fit along with confidence range
    plt.figure(figsize=(8, 6))
    plt.scatter(years,y_data)
    plt.plot(years, y_mod, 'r', label='Best fit')
    plt.fill_between(years, lower, upper, color='gray', alpha=0.5, label='Confidence Interval')
    plt.title('Curve fitting (GDP-Bangladesh) with confidence range')
    plt.ylabel('gdp')
    plt.xlabel('years')
    plt.xticks(rotation=45.0)
    plt.legend()
    plt.savefig("confidence.png")
    plt.show()
    #creating new dataframe new_df containing future year values, predicted
    #GDP values, lower error limit and upper error limit.
    new_df = pd.DataFrame()
    future_x = [2025,2030,2035,2040]
    future_y = expoFunc(future_x,a_opt,b_opt)
    future_lower=expoFunc(future_x, a_opt-da, b_opt-db)
    future_upper=expoFunc(future_x, a_opt+da, b_opt+db)
    new_df['Year']= future_x
    new_df['predicted GDP']= future_y
    new_df['Lower error limit']=future_lower
    new_df['upper error limit']=future_upper
    print(new_df)

#------------------------------------------------------------------------------

filenames= ['co2_per_capita','gdp','renew_energy_percent',
            'under5_mortality','unemployment','urban_population',
            'access_electricity']
merged= create_table(filenames)
heatmap(merged)
scatter_matrix(merged)
subplots(filenames)
clustering('gdp','under5_mortality')
clustering('gdp','renew_energy_percent')
clustering('renew_energy_percent','under5_mortality')
curvefit('gdp.csv')
error_range('gdp.csv')
