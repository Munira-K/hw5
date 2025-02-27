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

st.title('Задания 2 , 3, 6')

red_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-white.csv", sep=";")
wine_data = pd.concat([red_wine, white_wine])
wine = wine_data.sample(6497).reset_index().drop(['index'], axis=1)

st.write('''
### 🔹 Обработка меток классов
- 🛠 Если есть пропущенные значения в метках классов, удалите соответствующие записи.
''')

if st.checkbox('Наличие пропущенных значений'):
    st.dataframe(wine.isnull().sum())

st.write('''
- 🔄 Если классов больше двух, объедините их так, чтобы получилась бинарная классификация с примерно равным количеством примеров.
- ⚖️ Если один класс преобладает, объедините все остальные в один. Всего должно остаться 2 класса в таргете. ''')

if st.checkbox('Распределение классов до объединения'):
    st.dataframe(wine.quality.value_counts().sort_index())

wine["quality_enc"] = wine["quality"].apply(lambda x: 0 if x in [3, 4, 5] else 1)

if st.checkbox('Распределение классов после объединения'):
    st.dataframe(wine.quality_enc.value_counts())








st.write('''---
### 🔹 Предобработка признаков
- 🔢 Преобразуйте числовые признаки, если они были неправильно распознаны.
- 🗑 Удалите категориальные признаки (текстовые значения).
- 📉 Заполните пропущенные значения средними значениями отдельно для положительного и отрицательного классов. ''')

wine.drop(['quality'], axis=1, inplace=True)
st.code("wine.drop(['quality'], axis=1, inplace=True)", language="python")
st.write('**✅ Сделано**')



st.write('''
---
###  📊 Разделите данные на обучающую (70%) и тестовую (30%) выборки.''')


X = wine.drop(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
       'pH', 'sulphates','quality_enc'], axis=1)
y = wine['quality_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.code('''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)''',language="python")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write('- ⚙️ Выполните стандартизацию признаков перед обучением моделей.')
st.code('''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)''',language="python" )