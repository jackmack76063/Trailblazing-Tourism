# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 06:18:53 2025

@author: dream
"""

# Name: Mackenzie Jackson
# Module: 8-9
# "Project: Unsupervised Analysis Methods and Decision Trees "
# Date: 11/07/2025
# Citations:
# Source 1: Module 9 Exploration- Decision Tree Modeling
# Description: Used code examples provided
# URL: https://canvas.oregonstate.edu/courses/2026331/pages/exploration-decision-tree-modeling-using-sklearn-slash-python?module_item_id=25805948
#

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay

filename = "AllTrails_Labeled.csv"
trailsDF = pd.read_csv(filename)

#Drop Country column
trailsDF.drop(columns=['Country'], inplace=True)

#Set up train/test/split
trailsDF_Train, trailsDF_Test=train_test_split(trailsDF)
print("The Training Data is\n", trailsDF_Train)
print("The Testing Data is \n", trailsDF_Test)

#Seperate labels
TrainLabel = trailsDF_Train["TrailLevel"]
print("Train Label: ", TrainLabel)
trailsDF_Train = trailsDF_Train.drop(["TrailLevel"], axis=1)
print("\nTraining Data after Label Removal: ", trailsDF_Train)

TestLabel = trailsDF_Test["TrailLevel"]
print("\nTest Label: ", TestLabel)
trailsDF_Test = trailsDF_Test.drop(["TrailLevel"], axis=1)
print("\nTesting Data after Label Removal: ", trailsDF_Test) 
 
###Data Visualizations###

#boxplot#
#How trail difficulty relates to hard percents
#plt.figure(figsize=(12,6))
#sns.boxplot(x=TrainLabel, y=trailsDF_Train['HardPercent'])
#plt.title("Distribution of Hard Percent by Trail Level")
#plt.xlabel("Trail Level")
#plt.ylabel("Hard Percent")
#plt.show()

#histogram#
#distrubution of trail length
#plt.figure(figsize=(12,6))
#sns.histplot(trailsDF_Train['AvgLen_mi'], kde=True, bins=20)
#plt.title("Distribution of Average Trail Length")
#plt.xlabel("Trail Length (mi)")
#plt.ylabel("Count")
#plt.show()

#scatterplot
#Hard Percent vs Avg Length
#plt.figure(figsize=(10,6))
#sns.scatterplot(
#    x=trailsDF_Train['HardPercent'],
#    y=trailsDF_Train['AvgElevGain_ft'],
#    hue=TrainLabel,
#)
#plt.title(" Trail Elevation and Difficulty Relationship")
#plt.xlabel("Hard Percent")
#plt.ylabel("Avg Trail Elevation (ft)")
#plt.show()

###########################


#instantiate/fit model
MyDT_Classifier = DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(trailsDF_Train, TrainLabel)
print("\nDecision Tree Classes: ", MyDT_Classifier.classes_)
print("\nMy Training Columns: ", trailsDF_Train.columns)

#Predict the Testing Dataset
Prediction=MyDT_Classifier.predict(trailsDF_Test)
print("\nMy Prediction:", Prediction)

label_names= MyDT_Classifier.classes_

Actual_Labels=TestLabel
Predicted_Labels=Prediction

#Create Confusion Matrix
con_matrix = confusion_matrix(Actual_Labels, Predicted_Labels)
print("\nConfusion Matrix Data: ",con_matrix)

sns.heatmap(
    con_matrix, 
    annot=True,
    cmap='Greens',
    xticklabels=label_names, 
    yticklabels=label_names, 
    cbar=False
    )
plt.title("Confusion Matrix for Trail Count Density")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#Create Tree Visualization
New_Classifier = DecisionTreeClassifier(max_depth=4)
New_Classifier = New_Classifier.fit(trailsDF_Train, TrainLabel)

MyPlot=tree.plot_tree(New_Classifier,
                   feature_names=trailsDF_Train.columns, 
                   class_names=New_Classifier.classes_,
                   filled=True
                   )

plt.title("All Trails Decision Tree")
plt.savefig("AlltrailsTree.jpg")
plt.close()