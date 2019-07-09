import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("./DP_LIVE_21062019100730690.csv")

# print(df)

# print(df.head)

dfx = df[:14]  # Filter BEL
lg = LinearRegression()
x = dfx[["TIME"]]
y = dfx["Value"]
lg.fit(x, y)


accuracy = lg.score(x, y)
print("Akurasi Hasil: ",accuracy*100,'%')
# y_pred = lg.predict(x)
# print("Akurasi", accuracy_score(y, y_pred))
year = 2019
index = 1
for i in range(3):
    print("year " + str(year) + "=")
    predict_x = lg.predict([[year]])
    print(predict_x)
    year = year + index

# predict_x = lg.predict([[2019]])
# print(predict_x)


