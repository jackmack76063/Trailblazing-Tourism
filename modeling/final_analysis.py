
# Name: Mackenzie Jackson
# Module: 10
# "Project: Final"
# Description: One final clean and merge to compare results of three datasets. 
# Date: 11/30/2025
# 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

climatedf = pd.read_csv('Climate_Annuals_Cleaned.csv')
trails = pd.read_csv('AllTrails_Cleaned.csv')
tourism = pd.read_csv('Country_Arrivals_Cleaned.csv')

#Focus on years 2010-2019
filter_climate = climatedf[(climatedf['year'] >= 2010) & (climatedf['year'] <= 2019)]

#compute averages by country
climate_summary = filter_climate.groupby('Country').agg({
    'avg_temp_mean_f':'mean',
    'total_precip_mm':'mean',
    'avg_sunshine_hrs':'mean'
    }).reset_index()

# Rename columns
climate_summary.columns = ['Country', 'AvgTemp_F', 'AvgPrecip_mm', 'AvgSunshine_hrs']

# Look at most/least visited countries to focus in on
top10 = tourism.sort_values('2019', ascending=False).head(10)
bottom10 = tourism.sort_values('2019', ascending=True).head(10)
print("Top 10 Tourism Countries:\n", top10[['Country','2019']])
print("\nBottom 10 Tourism Countries:\n", bottom10[['Country','2019']])

#shifting tourism to long format (due to many year columns)
tourism_long = pd.melt(
    tourism,
    id_vars=['Country'],
    var_name='year',
    value_name='Arrivals'
)
#confirm year column is an int
tourism_long['year'] = tourism_long['year'].astype(int)

#merge tourism and climate data by country/year
merged_yearly = pd.merge(
    tourism_long, 
    climatedf, 
    on=['Country', 'year'], 
    how='inner'
    )
print("\nMerged dataset preview:\n", merged_yearly.head())

#Filter dataset to top/bottom countries:
focus_countries = list(top10['Country']) + list(bottom10['Country'])
focus = merged_yearly[ merged_yearly['Country'].isin(focus_countries)]
print("Rows in focus dataset: ", len(focus))
print(focus.head())

#Merge all trails dataset. 
focus = pd.merge(focus, trails, on='Country', how='left')
print(focus.head())

#Normalize columns for arrivals, helps analyze easier
focus['Arrivals_Millions'] = focus['Arrivals'] / 1_000_000

#Hiking trails a country has per million tourists
focus['Trails_per_MillionTourists'] = focus['TrailCount'] / (focus['Arrivals'] / 1_000_000)
focus.head()

####Visualizatuions###

#show tourism over time (top vs bottom countries)
sns.lineplot(data=focus, x='year', y='Arrivals_Millions', hue='Country')
plt.title("Tourism Trends: Top vs Bottom Countries")
plt.ylabel("Arrivals(Millions)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')   # << moves legend outside
plt.tight_layout()
plt.show()

#Temperature vs Tourism
sns.scatterplot(data=focus, x='avg_temp_mean_f', y='Arrivals_Millions', hue='Country')
plt.title("Tourism Numbers vs Avg Temps")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Rainfall vs Tourism
sns.scatterplot(data=focus, x='total_precip_mm', y='Arrivals_Millions', hue='Country')
plt.title("Tourism vs Precipitation")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Countries that are more hiking destions
plt.figure(figsize=(12,6))
sns.barplot(data=focus.drop_duplicates('Country'), x='Country', y='Trails_per_MillionTourists')
plt.xticks(rotation=90)
plt.title("Hiking Relationship: Trails Per Million Tourists")
plt.ylabel("Number of Trails per Million Visitors")
plt.show()
