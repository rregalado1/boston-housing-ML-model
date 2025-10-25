from re import T
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame = True)
df = data.frame.copy()

df = df.rename(columns={"MedHouseVal" : "target"})

#Analize data
print("Shape: ", df.shape)
print("\nHead:")
print(df.head(10))

print("\nInfo: ")
print(df.info())

print("\nUseful information: ")
print(df.describe().T)


X = data.data
y = data.target