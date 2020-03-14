# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- AVERAGE METHOD --------------------------------------------------------
# 2 - Write a python code that perform the task in step 1.
# Plot the True values versus Predicted values in one graph with different marker.
# You need to plot 3 graphs (because 3 models were developed in step 1) for this section.
# Add an appropriate title, legend, x-label, y-label to each graph
yt = np.array([1.5, 2.1,3.9, 4.4, 5.2])
y_hat = []
sum=0
for i in range(0,4):
    sum += yt[i]
    average = sum/(i+1)
    y_hat.append(average)
y_hat = np.array(y_hat[1:])
print("average predictions:", y_hat)
error = yt[2:] - y_hat
#print(error)
MSE1 = np.mean(np.square(error))
print("average MSE: ", MSE1)
df= pd.DataFrame(index=[1,2,3,4,5], columns= ['yt', 'y_hat', 'error'])
df['yt'] = np.array([1.5, 2.1,3.9, 4.4, 5.2])
df.iloc[2:,1] = y_hat
df.iloc[2:,2] = error
fig, ax = plt.subplots()
ax.plot(df['yt'], label="True Values", marker='.')
ax.scatter(x=3, y=df['y_hat'][3], label="Predicted t=3")
ax.scatter(x=4, y=df['y_hat'][4], label="Predicted t=4")
ax.scatter(x=5, y=df['y_hat'][5], label="Predicted t=5")
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel("Time(t)")
ax.set_ylabel("Value at Time t")
ax.set_title("Predictions for Average Method (One-Step Ahead Predictions) ")
ax.legend()
plt.show()


# 3 - Using python, calculate the forecast errors(3 errors for 3 models)
# which is the difference between predicted values and true values and display it as.
print(df.head())

# 4 - Using python, calculate the mean and variance of prediction errors.
mean1 = np.mean(df['error'])
var1 = np.var(df['error'])
print("The mean of forecast errors for the Average Method is = ", np.mean(df['error']))
print("The variance of forecast for the Average Method is = ", np.var(df['error']))

# 5 - Calculate the sum square of the prediction errors (SSE) and display the following message on the console.
print("The sum square error for the Average Method is: ", np.sum(np.square(df['error'])))

# 6 - Calculate the Q value for this estimateand display the following message(h=3 to be consistent)
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
errorlist = list(df['error'][2:])
print(errorlist)
ACFerrors1 = autocorrelation(error)
print(ACFerrors1)
Q1 = 4*np.sum(np.square(ACFerrors1[1:]))
print("The Q value for the Averaging Method is = ", Q1)

# ---------------------- NAIVE METHOD --------------------------------------------------------
# 7 - Repeat step 1 through 6 with the Naive Method
# Calculate one step Ahead Prediction
yt = np.array([1.5, 2.1,3.9, 4.4, 5.2])
y_hat_naive = yt[1:4]
error = yt[2:] - y_hat_naive
print("Naive predictions: ", y_hat_naive)
#print(error)
MSE2 = np.mean(np.square(error))
print("Naive MSE:", MSE2)

# Plot True Values versus Predicted Values
df= pd.DataFrame(index=[1,2,3,4,5], columns= ['yt', 'y_hat', 'error'])
df['yt'] = np.array([1.5, 2.1,3.9, 4.4, 5.2])
df.iloc[2:,1] = y_hat_naive
df.iloc[2:,2] = error
fig, ax = plt.subplots()
ax.plot(df['yt'], label="True Values", marker='.')
ax.scatter(x=3, y=df['y_hat'][3], label="Predicted t=3")
ax.scatter(x=4, y=df['y_hat'][4], label="Predicted t=4")
ax.scatter(x=5, y=df['y_hat'][5], label="Predicted t=5")
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel("Time(t)")
ax.set_ylabel("Value at Time t")
ax.set_title("Predictions for Naive Method (One-Step Ahead Predictions) ")
ax.legend()
plt.show()

# 3 - Using python, calculate the forecast errors(3 errors for 3 models)
# which is the difference between predicted values and true values and display it as.
print(df.head())

# 4 - Using python, calculate the mean and variance of prediction errors.
mean2 = np.mean(df['error'])
var2 = np.var(df['error'])
print("The mean of forecast errors for the Naive Method is = ", np.mean(df['error']))
print("The variance of forecast for the Naive Method is = ", np.var(df['error']))

# 5 - Calculate the sum square of the prediction errors (SSE) and display the following message on the console.
print("The sum square error for the Naive Method is: ", np.sum(np.square(df['error'])))

# 6 - Calculate the Q value for this estimateand display the following message(h=3 to be consistent)
errorlist = list(df['error'][2:])
print(errorlist)
ACFerrors2 = autocorrelation(error)
print(ACFerrors2)
Q2 = 4*np.sum(np.square(ACFerrors2[1:]))
print("The Q value for the Naive Method is = ", Q2)


#---------------------- DRIFT METHOD --------------------------------------------------------
# 8 Repeat step 1 through 6 with the drift method
yt = np.array([1.5, 2.1,3.9, 4.4, 5.2])
y_hat_drift = []
for i in range(1,4):
        x = yt[i] + ((yt[i]-yt[0])/i)
        y_hat_drift.append(x)
y_hat_drift = np.array(y_hat_drift)
print(y_hat_drift)
error = yt[2:] - y_hat_drift
print("Drift predictions: ", y_hat_drift)
#print(error)
MSE3 = np.mean(np.square(error))
print("Drift MSE: ",MSE3)
df= pd.DataFrame(index=[1,2,3,4,5], columns= ['yt', 'y_hat', 'error'])
df['yt'] = np.array([1.5, 2.1,3.9, 4.4, 5.2])
df.iloc[2:,1] = y_hat_drift
df.iloc[2:,2] = error
fig, ax = plt.subplots()
ax.plot(df['yt'], label="True Values", marker='.')
ax.scatter(x=3, y=df['y_hat'][3], label="Predicted t=3")
ax.scatter(x=4, y=df['y_hat'][4], label="Predicted t=4")
ax.scatter(x=5, y=df['y_hat'][5], label="Predicted t=5")
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel("Time(t)")
ax.set_ylabel("Value at Time t")
ax.set_title("Predictions for Drift Method (One-Step Ahead Predictions) ")
ax.legend()
plt.show()


# 3 - Using python, calculate the forecast errors(3 errors for 3 models)
# which is the difference between predicted values and true values and display it as.
print(df.head())

# 4 - Using python, calculate the mean and variance of prediction errors.
mean3 = np.mean(df['error'])
var3 = np.var(df['error'])
print("The mean of forecast errors for the Drift Method is = ", np.mean(df['error']))
print("The variance of forecast for the Drift Method is = ", np.var(df['error']))

# 5 - Calculate the sum square of the prediction errors (SSE) and display the following message on the console.
print("The sum square error for the Drift Method is: ", np.sum(np.square(df['error'])))

# 6 - Calculate the Q value for this estimateand display the following message(h=3 to be consistent)
errorlist = list(df['error'][2:])
print(errorlist)
ACFerrors3 = autocorrelation(error)
print(ACFerrors3)
Q3 = 4*np.sum(np.square(ACFerrors3[1:]))
print("The Q value for the Drift Method is = ", Q3)


#---------------------- Simple Exponential Smoothing (SES) METHOD --------------------------------------------------------
yt = np.array([1.5, 2.1,3.9, 4.4, 5.2])
y_hat_SES = [0,0,0]

for i in range(1,4):
    if i == 1:
        x = 0.5*yt[i]
        y_hat_SES[i-1] = x

    else:
        x = 0.5 * yt[i] + 0.5*y_hat_SES[i-2]
        y_hat_SES[i-1] =x


print(y_hat_SES)

y_hat_SES = np.array(y_hat_SES)
print(y_hat_SES)
error = yt[2:] - y_hat_SES
#print(error)
print("SES predictions: ", y_hat_SES)
MSE4 = np.mean(np.square(error))
print("SES MSE: ", MSE4)
# Plot the True Values versus the Predicted Values

df= pd.DataFrame(index=[1,2,3,4,5], columns= ['yt', 'yh_SES'])
df['yt'] = np.array([1.5, 2.1,3.9, 4.4, 5.2])
df.iloc[2:,1] = y_hat_SES
fig, ax = plt.subplots()
ax.plot(df['yt'], label="True Values", marker='.')
ax.plot(df['yh_SES'], label="Predicted SES", marker = ".")
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel("Time(t)")
ax.set_ylabel("Value at Time t")
ax.set_title("Predictions for SES Method")
ax.legend()
plt.show()

# Calculate Forecast errors

df['SES_errors'] = df['yt']-df['yh_SES']
print(df.head())

# Calculate Mean and Variance of Prediction Errors
mean4 = np.mean(df['SES_errors'])
var4 = np.var(df['SES_errors'])
print("The mean of forecast errors for the SES Method is  = ", np.mean(df['SES_errors']))
print("The variance of forecast errors for the SES Method is= ", np.var(df['SES_errors']))

# Calculate the SSE
print("The sum square error for the SES is: ", np.sum(np.square(df['SES_errors'])))


# Calculate the Q value

T2errors = list(df['SES_errors'][2:])
print(T2errors)
ACFerrors4 = autocorrelation(T2errors)
print(ACFerrors4)

Q4 = 4*np.sum(np.square(ACFerrors4[1:]))
print("The Q value for the SES method is = ", Q4)

# 10 - Create a table to compare the 4 forecast methods above by displaying Q values(h=3), MSE, Mean of prediction errors,
# variance of prediction errors


df= pd.DataFrame(columns= ['Forecasting Methods', 'Q Values', 'MSE', 'Prediction Error Mean','Prediction Error Variance'])
df['Forecasting Methods'] = ["Average", "Naive", "Drift", "SES"]
df['Q Values'] = [Q1,Q2,Q3,Q4]
df["MSE"] = [MSE1, MSE2, MSE3, MSE4]
df['Prediction Error Mean'] = [mean1,mean2, mean3, mean4]
df['Prediction Error Variance'] = [var1, var2, var3, var4]

print(df.head())
df.to_csv("modeltable1.csv")

# 11 - Using the Python program developed in the previous LAB, plot the ACF of predicted errors

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.stem(ACFerrors1)
ax1.set_title("ACF Average Prediction Errors")
ax1.set_ylabel('Autocorrelation')
ax1.set_xlabel('Lags')
ax2.stem(ACFerrors2)
ax2.set_title("ACF Naive Prediction Errors")
ax2.set_ylabel('Autocorrelation')
ax2.set_xlabel('Lags')
ax3.stem(ACFerrors3)
ax3.set_title("ACF Drift Prediction Errors")
ax3.set_ylabel('Autocorrelation')
ax3.set_xlabel('Lags')
ax4.stem(ACFerrors4)
ax4.set_title("ACF SES Prediction Errors")
ax4.set_ylabel('Autocorrelation')
ax4.set_xlabel('Lags')
plt.show()
