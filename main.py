"""
Name: Luciano Zavala
Date: 02/05/22
Assignment: Module 5: Project Wrangling Data
Due Date: 02/06/22
About this project: python script that computes data wrangling on top of the data sets chosen.
Assumptions:NA
All work below was performed by LZZ
"""

import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# housing dataset
df = pd.read_csv("housing.csv")

# California wild fires dataset
df_wildfires = pd.read_csv("California_Fire_Incidents.csv")

print("\nGeneral information housing.csv:")
print("--------------------------------------------------")
df.info()

print("\nGeneral information California_Fire_Incidents.csv:")
print("--------------------------------------------------")
df_wildfires.info()

print("\nOcean proximity value counts:")
print("--------------------------------")
print(df["ocean_proximity"].value_counts())

# Data Wrangling  (30 points)

# Reshaping the table
df_wildfires.drop(columns=['AirTankers',
                           'ControlStatement',
                           'FuelType',
                           'StructuresDestroyed',
                           'StructuresEvacuated',
                           'StructuresThreatened'
                           ], axis=1, inplace=True)

print("\nReshaped values:")
print("-------------------------------------------------")
df_wildfires.info()

# replacing null values
df_wildfires['CrewsInvolved'].fillna((df_wildfires['CrewsInvolved'].mean()), inplace=True)
df_wildfires['Dozers'].fillna((df_wildfires['Dozers'].mean()), inplace=True)
df_wildfires['Engines'].fillna((df_wildfires['Engines'].mean()), inplace=True)
df_wildfires['Helicopters'].fillna((df_wildfires['Helicopters'].mean()), inplace=True)
df_wildfires['Injuries'].fillna((df_wildfires['Injuries'].mean()), inplace=True)
df_wildfires['PersonnelInvolved'].fillna((df_wildfires['PersonnelInvolved'].mean()), inplace=True)
df_wildfires['StructuresDamaged'].fillna((df_wildfires['StructuresDamaged'].mean()), inplace=True)
df_wildfires['WaterTenders'].fillna((df_wildfires['WaterTenders'].mean()), inplace=True)

print("\nReplaced values:")
print("----------------------------------------------------------------")
print(df_wildfires.head())

# Text Attributes (30 points)

# 20 most common text
print("\n20 most common texts:")
print("\nSearch keywords: ", collections.Counter(df_wildfires['SearchKeywords']).most_common(20))
print("Search Description: ", collections.Counter(df_wildfires['SearchDescription']).most_common(20), "\n")

# Parse 20 common text
df_forest = df_wildfires[df_wildfires['SearchDescription'].str.contains('Fires', regex=False, na=False)]
print("Number search descriptions fires: ", len(df_forest))
print("Search description: ", collections.Counter(df_wildfires['SearchDescription']).most_common(20))

# Density distribution by population in California
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=df["population"]/100, label="population", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# Identify and Mitigate Outliers:  (40 points)

# Outliers analysis
df.hist(bins=50, figsize=(20, 15))
plt.show()

df_wildfires.hist(bins=50, figsize=(20, 15))
plt.show()

# median house value
df['median_house_value'].hist(bins=50, figsize=(20, 15))
plt.show()

df_wildfires['ArchiveYear'].hist(bins=50, figsize=(20, 15))
plt.show()

# Box Plot
sns.boxplot(x=df['median_house_value'], data=df)
plt.show()

# Scatter plot
fig, ax = plt.subplots(figsize=(18, 10))
ax.scatter(df['median_house_value'], df['median_income'])

# x-axis label
ax.set_xlabel('median_house_value')

# y-axis label
ax.set_ylabel('median_income')
plt.show()

# dropping outliers
drop_values = np.where(df['median_house_value'] > 400000)
df.drop(drop_values[0], inplace=True)

# Box Plot
sns.boxplot(x=df['median_house_value'], data=df)
plt.show()

# scatter plot
fig, ax = plt.subplots(figsize=(18, 10))
ax.scatter(df['median_house_value'], df['median_income'])

# x-axis label
ax.set_xlabel('median_house_value')

# y-axis label
ax.set_ylabel('median_income')
plt.show()
