from unittest import result
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data_classification.csv')
hours_slept = df['Hours_Slept'].tolist()
hours_studied = df['Hours_studied'].to_list()

fig = px.scatter(x=hours_slept, y=hours_studied)
#fig.show()

hours_slept = df['Hours_Slept'].tolist()
hours_studied = df['Hours_studied'].to_list()

results = df['results'].tolist()
colors = []

for data in results:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(data = go.Scatter(
    x = hours_studied, 
    y = hours_slept,
    mode = 'markers',
    marker=dict(color=colors)
))

#fig.show()

hours = df[['Hours_studied', 'Hours_Slept']]
results = df['results']


hours_train, hours_test, result_train, result_test = train_test_split(hours, results, test_size=0.25, random_state=0)
print(hours_train)

classifier = LogisticRegression(random_state = 0)
classifier.fit(hours_train, result_train)

results_predict = classifier.predict(hours_test)

from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(result_test, results_predict))

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
hours_train = sc_x.fit_transform(hours_train)  

user_hours_studied = int(input("Enter hours studied -> "))
user_hours_slept = int(input("Enter hours slept -> "))

user_test = sc_x.transform([[user_hours_studied, user_hours_slept]])

user_result_pred = classifier.predict(user_test)

if user_result_pred[0] == 1:
  print("This user may pass!")
else:
  print("This user may not pass!")