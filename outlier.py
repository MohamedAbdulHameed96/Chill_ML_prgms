  
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

  
#sample data with an outlier
data = np.array([250,270,280,300,320,350,380,1000])

  
#Calculate z-scores for each datapoints (positive value finding)
z_scores = np.abs(stats.zscore(data))

  
#define a threshold for outliers based on z_score (Commonly used threshold of 3) threshold=standard deviation
threshold =3

  
#Find and print the outliers based on the z_score #print first index value
outliers = np.where(z_scores > threshold)[0]
print("outlier (Indices):",outliers)

  
#create an array to specify the color of each point
colors ='red'

  
#Visualize the data with outliers highlighted
plt.figure(figsize=(8,6))
plt.scatter(np.arange(len(data)),data, c=colors)
plt.xlabel("Data Point Index")
plt.ylabel("sale price")
plt.title("Outliers Detection using z_score")
plt.show()

  



