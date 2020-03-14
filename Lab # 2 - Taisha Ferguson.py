import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1 - Write a python function called “ correlation_coefficent_cal(x,y)”
# that implement the correlation coefficient.
# The formula for correlation coefficientis given below. The function should be written
# in a general form than can work for any dataset x and dataset y.
# The return value for this function is r.

def correlation_coefficent_cal(x,y):
    n= len(x)
    x_bar = np.sum(x) / len(x)
    y_bar = np.sum(y) / len(y)
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    i = 0
    while i < n:
        numerator = numerator + ((x[i] - x_bar) * (y[i] - y_bar))
        denominator1 = denominator1 + (x[i] - x_bar)*(x[i] - x_bar)
        denominator2 = denominator2 + (y[i] - y_bar)*(y[i] - y_bar)
        i=i+1
    r = numerator /(math.sqrt(denominator1*denominator2))
    return r


# 2 - Test the “ correlation_coefficent_cal(x,y)”functionwith the following simple dataset.
# The x and y here are dummy variable and should be replaced by any other dataset.

X = [1,2,3,4,5]
Y = [1,2,3,4,5]
Z = [-1,-2,-3,-4,-5]
G = [1,1,0,-1,-1,0,1]
H = [0,1,1,1,-1,-1,-1]

# a - Plot the scatter plot between X and Y
r_xy = correlation_coefficent_cal(X, Y)
plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot of X and Y with r={}".format(r_xy))
plt.show()

# b - Plot the scatter plot between X and Z

r_xz = correlation_coefficent_cal(X, Z)
plt.scatter(X,Z)
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Scatter plot of X and Z with r={}".format(r_xz))
plt.show()


# c - Plot the scatter between G and H

r_gh = correlation_coefficent_cal(G, H)
plt.scatter(G,H)
plt.xlabel("G")
plt.ylabel("H")
plt.title("Scatter plot of G and H with r={}".format(r_gh))
plt.show()

# e - Calculate r_xy , r_xz  and r_gh using the written function “ correlation_coefficent_cal(x,y)”.

print('The correlation coefficient between X and Y is ', r_xy)
print('The correlation coefficient between X and Z is ', r_xz)
print('The correlation coefficient between G and H is ', r_gh)


# 3 - Load the time series data tute1.

df= pd.read_csv('tute1.csv', index_col="Date")
print(df.head())
print(df.info())
df=df[['Sales', 'AdBudget', 'GDP']]
print(df.head())
print(df.tail())


# 6 - Plot Sales, AdBudget and GDP versus time steps
ax = plt.gca()
df["Sales"].plot(y='Sales',ax=ax, legend=True, title= "Sales, AdBUdget, and GDP (March 1, 1981- June 8, 1981)")
df["AdBudget"].plot(y='AdBudget',ax=ax, legend=True)
df['GDP'].plot(y='GDP', ax=ax, legend=True)
plt.show()


# 7 - Graph the scatter plot for Sales and GDP. (y-axis plot Sales and x-axis plot GDP).
#Add the appropriate x-label and y-label. Don't add any title i this step. This needs to be updated in step 11.
plt.scatter(df['Sales'],df['GDP'])
plt.xlabel("Sales")
plt.ylabel("GDP")
plt.show()

# 8 - Graph the scatter plot for Sales and AdBudget. (y-axis plot Sales and x-axis plot AdBudget).
#Add the appropriate x-label and y-label. Don't add any title i this step. This needs to be updated in step 11.
plt.scatter(df['Sales'],df['AdBudget'])
plt.xlabel("Sales")
plt.ylabel("AdBudget")
plt.show()

# 9 - Call the function correlation_coefficient_cal(x,y) with the y as the Sales data and the x as the GDP data.
#Save the correlation coefficient between these two variables as r_xy.

r_xy = correlation_coefficent_cal(df['GDP'], df['Sales'])
print("The correlation coefficient between Sales value and GDP is ",r_xy)



# 10 - Call the function correlation_coefficient_cal(y,z) with the y as the Sales data and the z as the AdBudget data.
#Save the correlation coefficient between these two variables as r_yz.

r_yz = correlation_coefficent_cal(df['AdBudget'], df['Sales'])
print("The correlation coefficient between Sales value and AdBuget is ",r_yz)


# 11 - Include the r_xy and r_yz in the title of the graphs developed in step 5 and 6.
# Write your code in a way that anytime r_xy and r_yz value changes it automatically updated on the figure title.

plt.scatter(df['Sales'],df['GDP'])
plt.xlabel("Sales")
plt.ylabel("GDP")
plt.title("Scatter plot of GDP and Sales with r={}".format(r_xy))
plt.show()


plt.scatter(df['Sales'],df['AdBudget'])
plt.xlabel("Sales")
plt.ylabel("AdBudget")
plt.title("Scatter plot of AdBudget and Sales with r={}".format(r_yz))
plt.show()

# 13 - Performthe ADF-test and plot the histogram plot on the raw Salesdata,
# first order difference Sales dataand the logarithmictransformationof the Sales data.
# Which Sales dataset is stationary and which Sales dataset is non-stationary?Justify your answer
# according tothe ADF-Statistics and the histogramplot.

#Raw Sales Data
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print('\t%s: %.3f' %(key, value))

df['Sales'].hist()
plt.title("Histrogram of Sales")
plt.show()
ADF_Cal(df['Sales'])

#Sales First Difference
firstorderdifference = []
for i in range(len(df['Sales'])):
    difference=df['Sales'][i]-df['Sales'][i-1]
    firstorderdifference.append(difference)
df['FirstOrderDifference'] = firstorderdifference
df1 = df.drop(index='3/1/1981')
print(df1.head())
df1['FirstOrderDifference'].hist()
plt.title("Histogram of Sales with First Difference")
plt.show()
ADF_Cal(df1['FirstOrderDifference'])

#LogTransform
df["LogSales"] = np.log(df['Sales'])
print(df.head())
df['LogSales'].hist()
plt.title("Histrogram of Sales with Log Transform")
plt.show()
ADF_Cal(df['LogSales'])


