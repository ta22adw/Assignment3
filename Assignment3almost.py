#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:08:03 2023

@author: timothyatoyebi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
import errors as err
# Load the dataset into a Pandas dataframe
# Importing data in csv file

def read_data(q):
    """
    Reads files from comma separated values and imports them into a Python DataFrame.

    Arguments:
    a: string, That is the name of the CSV file to be read
    b: integer, shows how many rows in the csv file should be 
    skipped
    
    Returns:
    data: A pandas dataframe with all values from the excel file
    data_t: The transposed pandas dataframe
    """
    data = pd.read_csv(q, skiprows=4)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    df_data = data.set_index(data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    df_data = df_data.set_index('Year').dropna(axis=1)
    df_data = df_data.drop(['Country Name'])
    return data, df_data


# read files using function and extract the desired columns
# for Labor Force
lab_data = read_data("Labor_force.csv")
lab_data = lab_data[0]
print(lab_data.head())
# listing the column needed for Labor Force
lab_data = lab_data.iloc[:, [0,52]]

# for Unemployment Rate
unemp_data = read_data("Unemployment_rate.csv")
unemp_data = unemp_data[0]
print(unemp_data.head())
# listing the column needed for Unemployment Rate
unemp_data = unemp_data.iloc[:, [0,52]]
unemp_data = unemp_data.round(2)

# joining the dataframes to form a single one
dataf_s = pd.concat([lab_data, unemp_data["2010"]], axis=1)
print(dataf_s)

# removing of countries/regions not needed
regions = ['Arab World',
           'Caribbean small states',
           'Central Europe and the Baltics',
           'Early-demographic dividend',
           'East Asia & Pacific (excluding high income)',
           'Euro area',
           'Europe & Central Asia (excluding high income)',
           'European Union',
           'Fragile and conflict affected situations',
           'Heavily indebted poor countries (HIPC)',
           'High income',
           'Latin America & Caribbean (excluding high income)',
           'Latin America & the Caribbean (IDA & IBRD countries)',
           'Least developed countries: UN classification',
           'Low & middle income',
           'Low income',
           'Lower middle income',
           'Middle East & North Africa (excluding high income)',
           'Middle income',
           'North America',
           'OECD members',
           'Other small states',
           'Pacific island small states',
           'Small states',
           'South Asia (IDA & IBRD)',
           'Sub-Saharan Africa (excluding high income)',
           'Sub-Saharan Africa (IDA & IBRD countries)',
           'Upper middle income',
           'World']

# Remove the countries from the DataFrame
dataf_s = dataf_s[~dataf_s.index.isin(regions)]
print(dataf_s)

def clean_data(dataf_s, w, e):
    """
    This function takes a dataframe as input and performs several 
    cleaning tasks on it:
    1. Sets the index to 'Country Name'
    2. Renames the columns to 'Labour Force' and 'Unemployment'
    3. Prints information about the dataframe
    4. Drops rows with missing values
    
    Parameters:
    dataf_s (pandas dataframe): the dataframe to be cleaned
    
    Returns:
    pandas dataframe: the cleaned dataframe
    """
    # set index to 'Country Name'
    dataf_s = dataf_s.set_index('Country Name')
    
    # rename columns
    dataf_s.columns.values[0] = w
    dataf_s.columns.values[1] = e
    
    # print information about the dataframe
    #print(data.info())
    
    # drop rows with missing values
    dataf_s = dataf_s.dropna(axis=0)
    
    # return cleaned data
    return dataf_s

df = clean_data(dataf_s, "Labor Force", "Unemployment Rate")
print(df)

#Define function to plot a scatter plot
def scatter_plot(data, x_col, y_col, title, xlabel, ylabel):
    """
    This function takes a dataframe and plots a scatter plot with the 
    specified columns for x and y values, as well as a title, x-axis label, 
    and y-axis label.
    
    Parameters:
    data (pandas dataframe): the dataframe to be plotted
    x_col (int or string):  index of x-axis column. 
    y_col (int or string):  index of y-axis column. 
    title (string): the title of the plot
    xlabel (string): the label for the x-axis
    ylabel (string): the label for the y-axis
    """
    # extract x and y values from the dataframe
    x = data.iloc[:, x_col]
    y = data.iloc[:, y_col]
    
    # create scatter plot
    plt.scatter(x, y)
    
    # add title, legend, x-axis label, and y-axis label
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    #save plot figure
    plt.savefig('General Scatter Plot', dpi = 300)
    
    # show plot
    plt.tight_layout()
    plt.show()
    
#Activate scatter plot function
scatter_plot(df, 0, 1, "Scatter Plot of Labor Force vs Unemployment Rate", 
             "Labor Force", "Unemployment Rate")

# using the describe
print(df["Labor Force"].describe())
print(df["Unemployment Rate"].describe())

dfa = df[df.iloc[:,1] <= 8.047] #countries below world average
print(dfa['Unemployment Rate'].mean())
dfb = df[(df.iloc[:,1] > 8.047) & (df.iloc[:,1] < 10.01)] #
print(dfb)
dfc = df[df.iloc[:,1] >= 30]
print(dfc)

#Plot a scatter plot of countries with a gdp per capita 
#below the world avergae, i.e < 8.047
#Activate scatter plot function
scatter_plot(dfa, 0, 1, 
             "Countries with Unemployment Rate Below 8.047(World Average)", 
             "Unemployment Rate", "labour Force")

#Normalise the data
scaler = preprocessing.MinMaxScaler()
dfa_norm = scaler.fit_transform(dfa)
print(dfa_norm)

#Define a function to determine the number of effective clusters for KMeans
def optimise_k_num(data, max_k):
    """
    A function to determine the optimal number of clusters for k-means 
    clustering on a given dataset. The function plots the relationship 
    between the number of clusters and the inertia, and displays the plot.
    
    Parameters:
    - data (array-like): the dataset to be used for clustering
    - max_k (int): the maximum number of clusters to test for
    
    Returns: None
    """
        
    means = []
    inertias = []
    
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        max_iter=1000)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
        
    # generating the elbow_plot
    fig = plt.subplots(figsize=(8,5))
    plt.plot(means, inertias, "b-")
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method Showing Optimal Number of K", fontsize=16, 
              fontweight='bold')
    plt.grid(True)
    plt.savefig("K-Means Elbow Plot.png", dpi = 300)
    plt.show()
    
    return fig

#Activate the optimum k function to get the number of effective clusters
optimise_k_num(dfa_norm, 10)

#Create a function to run the KMeans model on the dataset
def kmeans_model(data, n_clusters):
    """
    Applies K-Means clustering on the data and returns the cluster labels.

    Parameters:
        data (numpy array or pandas dataframe) : The data to be clustered
        n_clusters (int) : The number of clusters to form.

    Returns:
        numpy array : The cluster labels for each data point
        numpy array : The cluster centers
        float : The inertia of the model
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, 
                    random_state=0)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    return clusters, centroids, inertia

#Activate KMeans clustering function
clusters, centroids, inertia = kmeans_model(dfa, 3)
print("Clusters: ", clusters)
print("Centroids: ", centroids)
print("Inertia: ", inertia)

#Calculate the silhouette score for the number of clusters
sil_0 = silhouette_score(dfa_norm, clusters)
print(sil_0)

dfa['Clusters'] = clusters
print(dfa)


#Define a function that plots the clusters
def plot_clusters(df, cluster, centroids):
    """
    Plot the clusters formed from a clustering algorithm.
    
    Parameters:
    df: DataFrame containing the data that was clustered.
    cluster: Array or Series containing the cluster labels for each 
    point in the data.
    centroids: Array or DataFrame containing the coordinates of the 
    cluster centroids.
    """
    
    df.iloc[:,1]
    df.iloc[:,0]
    cent1 = centroids[:,1]
    cent2 = centroids[:,0]
    plt.scatter(df.iloc[cluster == 0, 1], df.iloc[cluster == 0, 0], s=50,
               c='blue', label='Cluster 0')
    plt.scatter(df.iloc[cluster == 1, 1], df.iloc[cluster == 1, 0], s=50,
               c='orange', label='Cluster 1')
    plt.scatter(df.iloc[cluster == 2, 1], df.iloc[cluster == 2, 0], s=50,
               c='green', label='Cluster 2')
    #Centroid plot
    plt.scatter(cent1, cent2, c='red', s=100, label='Centroid')
    plt.title('Cluster of Countries with Unemployment Below 8.047',
              fontweight='bold')
    plt.ylabel('Labour Force', fontsize=12)
    plt.xlabel('Unemployment', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Clusters.png", dpi = 300)
    plt.show()

#Activate the function to plot the clusters    
plot_clusters(dfa, clusters, centroids)

#Carry out cluster analysis by plotting a bar chart showing the country 
#distribution in each cluster
sns.countplot(x='Clusters', data=dfa)
plt.savefig('Cluster distribution.png')
plt.title('Cluster Distribution of Countries with Unemployment Below.047'
          , fontweight='bold')
plt.show()

#Define a polynomial function to plot a curve fit curve
def fit_polynomial(x, a, b, c, d):
    """
    Fit a polynomial of degree 3 to a given set of data points.
    
    Parameters: 
    x: x-coordinates of the data points.
    a,b,c,d: function coefficients.
    
    Returns: Optimal values for the coefficients of the polynomial.
    """
    #popt, pcov = curve_fit(fit_polynomial, x, y)
    return  a*x*3 + b*x*2 + c*x + d

from numpy.polynomial import Polynomial
#Initialise variables
x_axis = dfa.values[:,1]
y_axis = dfa.values[:,0]

#Instantiate the curvefit function
popt, pcov = opt.curve_fit(fit_polynomial, x_axis, y_axis)
a, b, c, d = popt
print('y = %.5f * x^3 + %.5f * x^2 + %.5f * x + %.5f' % (a, b, c, d))
#print(pcov)

#Generate the curvefit line variables
d_arr = dfa.values[:,1] #convert data to an array
p = Polynomial.fit(x_axis, y_axis, 3)
x_fit = np.linspace(x_axis.min(), x_axis.max())
y_fit = p(x_fit)

x_line = np.arange(min(d_arr), max(d_arr)+1, 1) #a random range of points
y_line = fit_polynomial(x_line, a, b, c, d) #generate y-axis variables 
plt.scatter(x_axis, y_axis, label="Countries") #scatterplot
#plot the curvefit line
plt.plot(x_fit, y_fit, 'r-', color='black', linewidth=3, label="Curvefit")
plt.title('Cluster of Countries showing Prediction Line (Curvefit)',
              fontweight='bold')
plt.ylabel('Labour Force', fontsize=12)
plt.xlabel('Unemployment', fontsize=12)
plt.legend(loc='lower right')
plt.annotate('y = 0.00671x + 58.308', (3000, 55), fontweight='bold')
plt.savefig("Scatterplot Prediction Line Curvefit.png", dpi = 300)
plt.show()

#Generate the confidence interval and error range
sigma = np.sqrt(np.diag(pcov))
low, up = err.err_ranges(d_arr, fit_polynomial, popt, sigma)
#print(low, up)
#print(pcov) 

ci = 1.95 * np.std(y_axis)/np.sqrt(len(x_axis))
lower = y_line - ci
upper = y_line + ci
print(f'Confidence Interval, ci = {ci}')

#plot showing best fitting function and the error range
plt.scatter(x_axis, y_axis, label="Countries")
plt.plot(x_fit, y_fit, '--', color='black', linewidth=3, 
         label="Curvefit")
plt.fill_between(x_line, lower, upper, alpha=0.3, color='green', 
                 label="Error range")
plt.title('Cluster Showing Prediction Line (Curvefit) & Error Range',
              fontweight='bold')
plt.ylabel('Labour Force', fontsize=12)
plt.xlabel('Unemployment', fontsize=12)
plt.annotate(f'C.I = {ci.round(3)}', (7800, 60), fontweight='bold')
plt.legend(loc='lower right')
plt.savefig("Scatterplot Prediction Line error.png", dpi = 300)
plt.show()