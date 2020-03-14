# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 3 - Create white noise with zero mean and standard deviation of 1 and 10000 samples
X = 1*np.random.randn(10000) + 0

# 4 - Write Python Code to estimate Autocorrelation Function
def autocorrelation(y):
    y_bar = np.sum(y) / len(y)
    acf_list = []
    numerator = 0
    denomintator = 0
    n = len(y)
    for i in range(len(y)):
        denomintator += np.square(y[i] - y_bar)
    p = 0
    while p < n:
        r = n - p
        for j, z in zip(range(p, n), range(0, r)):
            numerator += ((y[j] - y_bar) * (y[z] - y_bar))
        quotient = numerator / denomintator
        acf_list.append(quotient)
        numerator = 0
        p = p + 1
    return acf_list


# 4a - Plot the ACF for the generated data in step 3.
#The ACF needs to be plotted using “stem” command.
AcfX= autocorrelation(X)
plt.stem(AcfX)
plt.title("Autocorrelation Function of Random Variable X")
plt.xlabel("Lags")
plt.ylabel("Autocorrelation")
plt.show()


# 4b - Plot both the generated WN in step 3 versus time andplot the histogram.

plt.plot(X)
plt.title("White Noise Versus Time")
plt.show()
plt.hist(X)
plt.title("White Noise Histogram")
plt.show()

# # 5 - Load the time series dataset tute1.csv (fromLAB#1)

df=pd.read_csv("tute1.csv", index_col="Date", parse_dates=True)
df=df[['Sales', 'AdBudget', 'GDP']]
print(df.head())

# 5a - Using python code written in the previous step,
# plot the ACF for the “Sales” and “Sales” versus time next to each other.
# You can use subplot command.

AcfSales = autocorrelation(df["Sales"])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfSales)
ax2.plot(df['Sales'])
ax1.set_title("Autocorrelation of Sales")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("Sales versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
fig.autofmt_xdate()
plt.show()


# 5b - Using python code written in the previous step,
# plot the ACF forthe “AdBudget” and “AdBudegt” versus time next to each other. You can use subplot command.
AcfAdBudget = autocorrelation(df["AdBudget"])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfAdBudget)
ax2.plot(df['AdBudget'])
ax1.set_title("Autocorrelation of AdBudget")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("AdBudget versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('AdBudget')
fig.autofmt_xdate()
plt.show()

# # 5c - Using python code written in the previous step,
# # plot the ACF for the “GDP” and “GDP” versus time next to each other. You can use subplot command.
AcfGDP = autocorrelation(df["GDP"])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfGDP)
ax2.plot(df['GDP'])
ax1.set_title("Autocorrelation of GDP")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("GDP versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('GDP')
fig.autofmt_xdate()
plt.show()

# # 5d - Run the ADF-testfor part a , b, and c and display them next to ACF and time plot in each section.
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    a= print("ADF Statistic: %f" %result[0])
    b= print("p-value: %f" % result[1])
    for key, value in result[4].items():
        print('\t%s: %.3f' %(key, value))
    return result

result=ADF_Cal(df['Sales'])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfSales[:21])
ax2.plot(df['Sales'])
ax1.set_title("Autocorrelation of Sales")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("Sales versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
fig.autofmt_xdate()
text1 = "ADF Statistic: " + str(result[0])
text2 = "p-value: " + str(result[1])
ax1.text(.1,1, text1, fontsize=14)
ax1.text(.1,.9, text2, fontsize=14)
plt.show()

result=ADF_Cal(df['AdBudget'])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfAdBudget[:21])
ax2.plot(df['AdBudget'])
ax1.set_title("Autocorrelation of AdBudget")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("AdBudget versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
fig.autofmt_xdate()
text1 = "ADF Statistic: " + str(result[0])
text2 = "p-value: " + str(result[1])
ax1.text(.1,1, text1, fontsize=14)
ax1.text(.1,.9, text2, fontsize=14)
plt.show()


result=ADF_Cal(df['GDP'])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.stem(AcfGDP[:21])
ax2.plot(df['GDP'])
ax1.set_title("Autocorrelation of GDP")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.set_title("GDP versus Time")
ax2.set_xlabel('Time')
ax2.set_ylabel('GDP')
fig.autofmt_xdate()
text1 = "ADF Statistic: " + str(result[0])
text2 = "p-value: " + str(result[1])
ax1.text(.1,1, text1, fontsize=14)
ax1.text(.1,.9, text2, fontsize=14)
plt.show()

# # 5f - The number lags used for this question is 20.