# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 06:51:29 2025

@author: dream
"""

# Name: Mackenzie Jackson
# Module: 5 & 6
# "Project: Cleaning 2 Datasets
# Citations:
# Source 1: Min/Max Normalization Sample from module
# URL: https://canvas.oregonstate.edu/courses/2026331/pages/exploration-data-normalization-transformation-and-feature-engineering?module_item_id=25805929
# Date: 11/04/2025
# 
import pandas as pd
import numpy as np
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

climatefile = "climate_data.csv"
climatedf = pd.read_csv(climatefile)
arrivalsfile = "CountryArrivals.csv"
arrivalsdf = pd.read_csv(arrivalsfile)
trailsfile = "AllTrails_Cleaned.csv"
alltrailsdf = pd.read_csv(trailsfile)

###################################################
###
###Country Arrivals###
###
###################################################
print("\nCleaning Country Arrivals Dataframe...\n")
print("\Before Cleaning Country Arrivals:\n")
print(arrivalsdf)

#drop unwanted columns
arrivalsdf.drop(columns=["Country Code", "Indicator Name", "Indicator Code"], inplace=True)

print("\nDatatypes for Country Arrivals:\n")
print(arrivalsdf.dtypes) 

print("\nMissing Values:\n")
print(arrivalsdf.isnull().sum())

##Cleaning Countries Variable##

# Check for missing countries (before)
missing_countries = arrivalsdf['Country Name'].isnull().sum()
print(f"Missing country values': {missing_countries}")

#Countplot for Country Records (before)
#plt.figure(figsize=(12,6))
#sns.countplot(y='Country Name', data=arrivalsdf, order=arrivalsdf['Country Name'].value_counts().index)
#plt.title("Count of Records by Country (Before Cleaning)")
#plt.xlabel("Number of Records")
#plt.ylabel("Country Name")
#plt.show()

#drop unwanted countries that are not on my alltrails data
trails_countries = set(alltrailsdf['Country'].unique())
arrivalsdf = arrivalsdf[arrivalsdf['Country Name'].isin(trails_countries)]

# Check for missing countries (after)
missing_countries = arrivalsdf['Country Name'].isnull().sum()
print(f"Missing countries in arrivals: {missing_countries}")

#Countplot for Country Records (after)
#plt.figure(figsize=(12,6))
#sns.countplot(y='Country Name', data=arrivalsdf, order=arrivalsdf['Country Name'].value_counts().index)
#plt.title("Count of Records by Country (After Cleaning)")
#plt.xlabel("Number of Records")
#plt.ylabel("Country Name")
#plt.show()


##Cleaning Years Variables##

# Check for missing year counts (before)
print("\nMissing Year Values Before:\n")
print(arrivalsdf.isnull().sum())

#heatmap for countries before
#plt.figure(figsize=(12,6))
#sns.heatmap(arrivalsdf[year_cols].isnull(), cbar=False)
#plt.title("Missing Data by Country and Year (Before Cleaning)")
#plt.xlabel("Year")
#plt.ylabel("Country Index")
#plt.show()

#drop unwanted years that are not 1995-2020
keep_years = [str(year) for year in range(1995, 2021)]
keep_cols = ['Country Name'] + keep_years
arrivalsdf = arrivalsdf[keep_cols]

#interpolate missing data
arrivalsdf[keep_years] = arrivalsdf[keep_years].interpolate(
    axis=1, method='linear', limit_direction='both')

#check for missing year counts (after)
print("\nMissing Year Values After:\n")
print(arrivalsdf.isnull().sum())

#check for negative numbers
negative_years = (arrivalsdf[keep_years] < 0)
count_negatives = negative_years.sum().sum()
print(f"Negative values in years: {count_negatives}")

#boxplot to find outliers in years
#plt.figure(figsize=(12,6))
#sns.boxplot(data=arrivalsdf[keep_years], orient='h')
#plt.title("Tourist Arrivals per Year")
#plt.xlabel("Number of Arrivals (mill)")
#plt.ylabel("Year")
#plt.show()

#barchart for top 10 country arrivals (mean)
#arrivalsdf['Average_Arrivals'] = arrivalsdf[keep_years].mean(axis=1)
#top_avg = arrivalsdf[['Country Name', 'Average_Arrivals']].sort_values(by='Average_Arrivals', ascending=False).head(10)
#plt.figure(figsize=(12,6))
#sns.barplot(x='Average_Arrivals', y='Country Name', data=top_avg)
#plt.title("Top 10 Countries by Average Tourist Arrivals")
#plt.xlabel("Average Number of Arrivals (mill)")
#plt.ylabel("Country")
#plt.show()

print("\nAfter Cleaning Country Arrivals:\n")
print(arrivalsdf)

###################################################
###
###Normalized Arrivals Dataframe (min/max)###
###
###################################################

print("\nNormalizing Country Arrivals Dataframe...\n")
print("\nBefore Normalizing: \n", arrivalsdf.head(20))

#Only use year columns (only numerics)
year_cols = [str(year) for year in range(1995, 2021)]

#copy dataframe to scale
arrivals_scaled = arrivalsdf.copy()

scaler = MinMaxScaler()
arrivals_scaled[year_cols] = scaler.fit_transform(arrivals_scaled[year_cols])

#After normalizing
print("\nAfter normalization:\n", arrivals_scaled.head(20))

###################################################
###
###KMeans Arrivals Dataset###
###
###################################################
print("\nPrepping Arrivals Dataframe for Kmeans...\n")
print("\nBefore Prepping for Kmeans: \n", arrivals_scaled.head(20))

#copy over to new dataframe
arrivals_kmeans = arrivals_scaled[year_cols].copy()

#After prepping
print("\nAfter Prepping for Kmeans:\n", arrivals_kmeans.head(20))

#Save to new csv file
arrivals_kmeans.to_csv('Arrivals_kmeans.csv', index=False)

###################################################
###
###Climate Data###
###
###################################################
print("\nCleaning Climate Dataframe...\n")
print("\nBefore Cleaning Climate Data:\n")
print(climatedf)


print("\nDatatypes for Climate Data:\n")
print(climatedf.dtypes) 

##Cleaning Country Variables##

#check for missing countries
missing_countries = climatedf['country'].isnull().sum()
print(f"\nMissing countries in climate data: {missing_countries}\n")

#histogram plot to show number of records per country
#helps to see if there are any duplicates as they should all be the same
#plt.figure(figsize=(12,6))
#top_counts = climatedf['country'].value_counts().head(15)
#sns.barplot(x=top_counts.values, y=top_counts.index)
#plt.title("Top Countries")
#plt.xlabel("Number of Country Records")
#plt.ylabel("Country")
#plt.show()

#drop unwanted countries that are not on my alltrails data
climatedf = climatedf[climatedf['country'].isin(trails_countries)]

print("\nAfter dropping countries:\n")
print(climatedf)

#visualization of countries after cleaning
#plt.figure(figsize=(12,6))
#sns.countplot(y='country', data=climatedf, order=climatedf['country'].value_counts().index)
#plt.title("Count of Records by Country (After Cleaning)")
#plt.xlabel("Number of Records")
#plt.ylabel("Country Name")
#plt.show()

##Cleaning Years Variable##

#barplot shows records per year to determine if any missing
#year_counts = climatedf['year'].value_counts().sort_index()
#plt.figure(figsize=(12,6))
#sns.barplot(x=year_counts.index, y=year_counts.values)
#plt.title("Number of Records per Year (Before Cleaning)")
#plt.xlabel("Year")
#plt.ylabel("Record Count")
#plt.show()

##Cleaning avg_temp_mean variable##

# Visualization before cleaning
# shows distribution of the avg temp mean 
#plt.figure(figsize=(12,6))
#sns.histplot(climatedf['avg_temp_mean_f'], bins=20)
#plt.title("Spread of Average Mean Temperature (Before Cleaning)")
#plt.xlabel("Average Temperature (F)")
#plt.ylabel("Times Occured")
#plt.show()

#check for missing entries
print(f"\nMissing entries in avg_temp_mean_f: {climatedf['avg_temp_mean_f'].isnull().sum()}\n")

print(climatedf['avg_temp_mean_f'].describe())


##Cleaning avg_temp_max variable##

# Visualization before cleaning
# shows distribution of the avg temp max 
#plt.figure(figsize=(12,6))
#sns.histplot(climatedf['avg_temp_max_f'], bins=20)
#plt.title("Spread of Average Max Temperature (Before Cleaning)")
#plt.xlabel("Average Temperature (F)")
#plt.ylabel("Times Occured")
#plt.show()

#check for missing entries
print(f"\nMissing entries in avg_temp_max_f: {climatedf['avg_temp_max_f'].isnull().sum()}\n")

print(climatedf['avg_temp_max_f'].describe())

##Cleaning avg_temp_min variable##

# Visualization before cleaning
# shows distribution of the avg temp min 
#plt.figure(figsize=(12,6))
#sns.histplot(climatedf['avg_temp_min_f'], bins=20)
#plt.title("Spread of Average Min Temperature (Before Cleaning)")
#plt.xlabel("Average Temperature (F)")
#plt.ylabel("Times Occured")
#plt.show()

#check for missing entries
print(f"\nMissing entries in avg_temp_min_f: {climatedf['avg_temp_min_f'].isnull().sum()}\n")

print(climatedf['avg_temp_min_f'].describe())

##Cleaning total_precip_mm##

# Visualization before cleaning
# shows distribution of the total precipitation annually 
#plt.figure(figsize=(12,6))
#sns.histplot(climatedf['total_precip_mm'], bins=20)
#plt.title("Spread of Total Precipitation (Before Cleaning)")
#plt.xlabel("Total Annual Precipitation (mm)")
#plt.ylabel("Times Occured")
#plt.show()

#check for missing entries
print(f"\nMissing entires in total_precip_mm: {climatedf['total_precip_mm'].isnull().sum()}\n")

print(climatedf['total_precip_mm'].describe())

#showing wettest countries
top_wet = climatedf.groupby('country')['total_precip_mm'].mean().sort_values(ascending=False).head(10)
print("\nTop Wettest Countries:\n")
print(top_wet)

#showing driest countries
top_dry = climatedf.groupby('country')['total_precip_mm'].mean().sort_values(ascending=True).head(10)
print("\nTop Driest Countries:\n")
print(top_dry)

##Cleaning avg_sunshine_hrs variable##

# Visualization before cleaning
# shows distribution of the the average sunshine hours annually
plt.figure(figsize=(12,6))
sns.histplot(climatedf['avg_sunshine_hrs'], bins=20)
plt.title("Spread of Average Annual Sunshine (Before Cleaning)")
plt.xlabel("Sunshine (hours)")
plt.ylabel("Times Occured")
plt.show()

#check for missing entries
print(f"\nMissing entires in avg_sunshine_hrs: {climatedf['avg_sunshine_hrs'].isnull().sum()}\n")

print(climatedf['avg_sunshine_hrs'].describe())

##Cleaning total_radiation_mj variable##

# Visualization before cleaning
# shows distribution of the the total solar radiation annually
#plt.figure(figsize=(12,6))
#sns.histplot(climatedf['total_radiation_mj'], bins=20)
#plt.title("Spread of Total Annual Solar Radiation (Before Cleaning)")
#plt.xlabel("Solar Radiation (MJ/m^2)")
#plt.ylabel("Times Occured")
#plt.show()

#check for missing entries
print(f"\nMissing entires in total_radiation_mj: {climatedf['total_radiation_mj'].isnull().sum()}\n")

print(climatedf['total_radiation_mj'].describe())

#showing highest radiation countries
top_rad = climatedf.groupby('country')['total_radiation_mj'].mean().sort_values(ascending=False).head(10)
print("\nTop Countries with High Radiation:\n")
print(top_rad)

#After Cleaning Climate Data
print("\nAfter Cleaning Climate Data:\n")
print(climatedf)

#save cleaned datasets
climatedf.to_csv('Climate_Annuals_Cleaned.csv', index=False)
arrivalsdf.to_csv('Country_Arrivals_Cleaned.csv', index=False)

###################################################
###
###Normalized Climate Dataframe (min/max)###
###
###################################################
print("\nNormalizing Climate Dataframe...\n")
print("\nBefore Normalizing:\n", climatedf)

#Only scale numeric columns
normalize_cols = [
    'avg_temp_mean_f',
    'avg_temp_max_f',
    'avg_temp_min_f',
    'total_precip_mm',
    'avg_sunshine_hrs',
    'total_radiation_mj'
    ]
#copy data frame to scale
climate_scaled = climatedf.copy()

#establish scaler
scaler = MinMaxScaler()
climate_scaled[normalize_cols] = scaler.fit_transform(climate_scaled[normalize_cols])

#After normalizing
print("\nAfter normalization:\n", climate_scaled)

###################################################
###
###KMeans Climate Dataset###
###
###################################################
print("\nPrepping Climate Dataframe for Kmeans...\n")
print("\nBefore Prepping for Kmeans:\n", climate_scaled)

#copy over to new dataframe
climate_kmeans = climate_scaled[normalize_cols].copy()

#After prepping
print("\nAfter Prepping for Kmeans:\n", climate_kmeans)

#Save to new csv file
climate_kmeans.to_csv('Climate_kmeans.csv', index=False)