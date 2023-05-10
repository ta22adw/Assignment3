#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 7 10:56:52 2023

@author: timothyatoyebi 22027370
"""
# Assignment 3:  Clustering and fitting
# Data source:https://data.worldbank.org/topic/climate-change
#GITHUB : https://github.com/ta22adw/Assignment3
# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from datetime import datetime

# Load the dataset into a Pandas dataframe
# Importing data in csv file

def read_data(q, w):
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
    data = pd.read_csv(q, skiprows=w)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    df_data = data.set_index(data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    df_data = df_data.set_index('Year').dropna(axis=1)
    df_data = df_data.drop(['Country Name'])
    return data, df_data

q = 'WorldbankData.csv'
w = 4

data, df_data = read_data(q, w)
print(df_data)
print(df_data.head)

# Sorting the dataframe to obtain information for the targeted indicators

def indicator_s(e,r):
    """
    Reads and selects precise indicators from world bank dataframe, to a python DataFrame

    Arguments:
    e: 'First Indicator'
    r: 'Second Indicator'

    
    Returns:
    ind: A pandas dataframe with all values from 
    """
    ind = data[data['Indicator Name'].isin([e,r])]
    
    return ind

e = 'Urban population'
r = 'Renewable electricity output (% of total electricity output)'



i = indicator_s(e,r)

print(i.head)

# Slicing the dataframe to get data for the countries of interest

def country_s(countries):
    """
    Reads and selects country of interest from world bank dataframe, to a python DataFrame

    Arguments:
    countries: sets of countries selected by me 
    
    Returns:
    sc: A pandas dataframe with specific countries selected
    """
    
    sc = i[i['Country Name'].isin(countries)]
    sc = sc.dropna(axis=1)
    sc = sc.reset_index(drop=True)
    
    return sc

# Selecting countries needed

countries = ['United States', 'India', 'China', 'United Kingdom', 'Nigeria', 'South Africa']


sc = country_s(countries)
print(sc.head)

# STATISTICS OF THE DATA
stats_desc = sc.groupby(["Country Name", "Indicator Name"])
print(stats_desc.describe())

def gc_in(indicator):
    """
    Add countries to a Python DataFrame after selecting and grouping them according to the specified metrics.

    Arguments:
    indicator: 
    
    Returns:
    gc_in_con: A pandas dataframe with specific countries selected
    """
    gc_in_con = sc[sc["Indicator Name"] == indicator]
    gc_in_con = gc_in_con.set_index('Country Name', drop=True)
    gc_in_con = gc_in_con.transpose().drop('Indicator Name')
    gc_in_con[countries] = gc_in_con[countries].apply(pd.to_numeric, errors='coerce', axis=1)
    return gc_in_con

# Inputting the indicator
Population = gc_in("Urban population")
print(Population.head())

Electricity = gc_in("Renewable electricity output (% of total electricity output)")
print(Electricity.head())

# Normalize the data for clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(Population, Electricity)

# Perform KMeans clustering on the data
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(normalized_data)

# Plot the clusters
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', color='red')
plt.xlabel('Population')
plt.ylabel('Electricity')
plt.title('Clustering of Countries based on Population and CElectricity')
plt.show()

# Calculate cluster centers
#cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

