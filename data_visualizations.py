# Copyright 2021, Gregory Happ, All rights reserved.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv(r"C:\Users\Greg Happ\Desktop\DATA INCUBATOR\capstone_car_accident_forecasting\US_Accidents_June20.csv", usecols=["Severity", "Temperature(F)", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset"])

#########################################################################################
# set theme
sns.set_theme(style="ticks")

# scatter plot matrix
sns.pairplot(df.sample(frac=0.001, random_state=42), hue="Severity")
plt.show()
#########################################################################################

#########################################################################################
# reset theme
sns.set_theme(style="darkgrid")

# bar chart
sns.countplot(x="Severity", data=df)
plt.show()
##########################################################################################
