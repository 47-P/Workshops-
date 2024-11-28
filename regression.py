import pandas as pd  # importing Pandas library to read in the csv dataset 

import numpy as np # importing Numpy to test the functions we write 

from sklearn.linear_model import LinearRegression

const = 1 # This will be used to help normalize the 'Year' variable

dataset = pd.read_csv('dataset.csv')  

dataset = dataset.drop('Birth Number',  axis = 1)

dataset = dataset.drop('Crude Birth Rate',  axis = 1)

x = dataset['Year'].values
y = dataset['General Fertility Rate'].values


for i in range(110):
    x[i] = const
    const+=1

def mean(values):
    return sum(values) / len(values)


# Calculating the optimal value for b1

numerator = sum((x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x)))
denomenator = sum((x[i] - mean(x)) ** 2 for i in range(len(x)))

b_1 = numerator / denomenator

b_0 = mean(y) - (b_1 * mean(x))

# while(True):
#     input_value = int(input("Give me a year that you would want to predict the Fertility Rate for: "))
#     input_value_normalized = input_value - 1909
#     output_value = b_0 + (b_1 * input_value_normalized)
#     print(f"For year {input_value} predicted fertility rate is: {output_value}")

output_values = []
for i in range(110):
    output_value = b_0 + b_1 * x[i]
    output_values.append(output_value)

rounded_array = [format(value, ".2f") for value in output_values]

print(rounded_array)

x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)


output_values1 = model.predict(x)

rounded_array1 = [format(value, ".2f") for value in output_values1]

print(rounded_array1)