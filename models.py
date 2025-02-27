import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

st.title('Задание 6')

# объединение таблиц
red_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-white.csv", sep=";")
wine_data = pd.concat([red_wine, white_wine])
wine = wine_data.sample(6497).reset_index().drop(['index'], axis=1)


wine["quality_enc"] = wine["quality"].apply(lambda x: 0 if x in [3, 4, 5] else 1)


wine.drop(['quality'], axis=1, inplace=True)

X = wine.drop(['fixed acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
       'pH', 'sulphates','quality_enc'], axis=1)
y = wine['quality_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write('''- 🤖 Постройте модели классификации на основе двух наиболее коррелированных признаков. Параметры модели указаны в индивидуальном задании.''')

logistic = LogisticRegression(max_iter=565)
logistic.fit(X_train, y_train)
y_pred_logistic = logistic.predict(X_test)
log = '''logistic = LogisticRegression(max_iter=565)
logistic.fit(X_train, y_train)'''
st.code(log, language='python')

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn1 = '''knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)'''
st.code(knn1, language='python')

des_tree = DecisionTreeClassifier(max_depth=5)
des_tree.fit(X_train, y_train)
y_pred_des_tree = des_tree.predict(X_test)
des = '''des_tree = DecisionTreeClassifier(max_depth=5)
des_tree.fit(X_train, y_train)'''
st.code(des, language='python')


st.title("🍷 Wine Quality Prediction")

# Отображение исходных данных
with st.expander('Data'):
    st.write("X")
    st.dataframe(X)
    st.write("y")
    st.dataframe(y)


with st.sidebar:
    st.header("Введите признаки: ")
    alcohol = st.slider('Alcohol', 8.0, 14.9, 10.5)
    density = st.slider('Density', 0.99, 1.0, 1.1)
    volatile_acidity = st.slider('Volatile acidity', 0.08, 1.58, 0.34)


new = np.array([[alcohol, density, volatile_acidity]])
new_scaled = scaler.transform(new)  

pred_logistic = logistic.predict(new_scaled)[0]
pred_knn = knn.predict(new_scaled)[0]
pred_des_tree = des_tree.predict(new_scaled)[0]

def mapping(pred):
    return "Хорошее вино ✅" if pred == 1 else "Вино не очень... 😶"

st.subheader("🔮 Предсказания моделей:")
st.write(f"**Logistic Regression:** {mapping(pred_logistic)}")
st.write(f"**K-Nearest Neighbors:** {mapping(pred_knn)}")
st.write(f"**Decision Tree:** {mapping(pred_des_tree)}")
