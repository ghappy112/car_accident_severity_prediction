# Severity Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv(r"US_Accidents_June20.csv", usecols=["Severity", "Temperature(F)", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset"])

# view dataset
print(df.head())
print(df.tail())
print(df.describe())

# recode sunrise sunset as booleans
df["Sunrise"] = df.Sunrise_Sunset == 'Day'
df["Sunset"] = df.Sunrise_Sunset == 'Night'
df["Sunrise"] = df["Sunrise"]*1
df["Sunset"] = df["Sunset"]*1

# clean dataset
df = df.dropna()
df = df.reset_index()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[df==np.inf]=np.nan
df = df.dropna()
df = df.reset_index()

df['Severity'] = df['Severity'].astype(str)

##########################################################################################################################
# Predictive Power Score Matrix for feature selection
import ppscore as pps
print(pps.matrix(df))
dfpps = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(dfpps, annot=True)
plt.show()

###############################################################################################
# prepare data for training and testing

X = df[["Temperature(F)", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Sunrise", "Sunset"]]
y = df["Severity"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42) #training and testing

###############################################################
# gradient boosted tree
from sklearn.ensemble import GradientBoostingClassifier
tree = GradientBoostingClassifier()

# train & test tree
tree.fit(X_train, y_train)
print("gradient boosted tree score:", tree.score(X_test, y_test))
from collections import Counter
print("Predictions:", Counter(tree.predict(X_test)))


# final tree
tree.fit(X, y)
print("gradient boosted tree score:", tree.score(X, y))
print("Predictions:", Counter(tree.predict(X)))

# save the tree
# joblib.dump(tree, r"TheTreeOfSeverity.pkl", compress = 1)
