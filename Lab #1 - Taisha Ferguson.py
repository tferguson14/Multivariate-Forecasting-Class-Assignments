# Lab #1 - Taisha Ferguson

# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

# 1 - Load the time series data called tute1
df= pd.read_csv('tute1.csv', index_col="Date")
print(df.head())
print(df.info())
df=df[['Sales', 'AdBudget', 'GDP']]
print(df.head())
print(df.tail())

# 4 - Plot Sales, AdBudget and GPD versus timestep.
#Sales

ax = df["Sales"].plot(legend=True, title= "Daily Sales March 1, 1981- June 8, 1981")
ax.set_ylabel("Sales")
plt.show()
#AdBudget

ax=df['AdBudget'].plot(legend=True, title="Daily AdBudget March 1, 1981- June 8, 1981")
ax.set_ylabel("AdBudget")
plt.show()
#GDP

ax=df['GDP'].plot(legend=True, title="Daily GDP March 1, 1981- June 8, 1981")
ax.set_ylabel('GDP')
plt.show()

# 5 & 6 - Find the time series statistics (average, variance and standard deviation) of Sales, AdBudget and GDP
print("The Sales mean is:",df['Sales'].mean(), "and the variance is:",df['Sales'].var(), "with standard deviation:",
df['Sales'].std())
print("The AdBudget mean is:",df['AdBudget'].mean(), "and the variance is:",df['AdBudget'].var(), "with standard deviation:",
df['AdBudget'].std())
print("The GDP mean is:",df['GDP'].mean(), "and the variance is:",df['GDP'].var(), "with standard deviation:",
df['GDP'].std())


# 7 - Prove that the Sales, AdBudgetand GDP in the this time series datasetis stationary
salesmean=[]
salesvariance=[]
admean=[]
advariance=[]
gdpmean=[]
gdpvariance=[]
salesmean=[]
for i in range(len(df.Sales)):
    #Sales
    smean=np.mean(df.Sales[:i])
    salesmean.append(smean)
    svar = np.var(df.Sales[:i])
    salesvariance.append(svar)
    #AdBudget
    amean = np.mean(df.AdBudget[:i])
    admean.append(amean)
    avar = np.var(df.AdBudget[:i])
    advariance.append(avar)
    #GDP
    gmean = np.mean(df.GDP[:i])
    gdpmean.append(gmean)
    gvar = np.var(df.GDP[:i])
    gdpvariance.append(gvar)
df["SalesMean"] = salesmean
df['SalesVariance'] = salesvariance
df["AdBudgetMean"] = admean
df['AdBudgetVariance'] = advariance
df["GDPMean"] = gdpmean
df['GDPVariance'] = gdpvariance
print(df.head())

# 8 - Plot all the means and variances.
#Means
ax = plt.gca()
df["SalesMean"].plot(y='Sales',ax=ax, legend=True, title= "Sales Mean, AdBudget Mean, GDP Mean (March 1, 1981- June 8, 1981)")
df["AdBudgetMean"].plot(y='AdBudget',ax=ax, legend=True)
df['GDPMean'].plot(y='GDP', ax=ax, legend=True)
plt.show()
#Variances
ax = plt.gca()
df["SalesVariance"].plot(y='Sales',ax=ax, legend=True, title= "Sales Variance, AdBudget Variance, GDP Variance (March 1, 1981- June 8, 1981)")
df["AdBudgetVariance"].plot(y='AdBudget',ax=ax, legend=True)
df['GDPVariance'].plot(y='GDP', ax=ax, legend=True)
plt.show()

# 9 - Perform an ADF-test to check if the Sales, AdBudget and GDP stationary or not.
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print('\t%s: %.3f' %(key, value))
ADF_Cal(df['Sales'])
ADF_Cal(df['AdBudget'])
ADF_Cal(df['GDP'])

