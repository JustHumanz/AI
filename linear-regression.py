import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./DP_LIVE_21062019100730690.csv")
dfx=df[:14] #Filter BEL
lg = LinearRegression()
x = df[["TIME"]]
y = df["Value"]
lg.fit(x,y)
predict_x = lg.predict([[2019]])
print(predict_x)
