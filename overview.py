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

st.title('Задания 1, 4, 5')
st.write(''' ### 🔹 Загрузка данных:
         
Загрузите набор данных из репозитория UCI, включая столбец с метками классов, указанный в индивидуальном задании.
''')

# объединение таблиц
red_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-white.csv", sep=";")
wine_data = pd.concat([red_wine, white_wine])
wine = wine_data.sample(6497).reset_index().drop(['index'], axis=1)


if st.checkbox('Показать/скрыть данные таблицы : wine'):
    st.dataframe(wine)

data= wine.copy()
data["quality_enc"] = data["quality"].apply(lambda x: 0 if x in [3, 4, 5] else 1)

data.drop(['quality'], axis=1, inplace=True)



st.write(''' ---
### 🔹 Отбор признаков
- 🔍 Определите три наиболее значимых признака, содержащих более 10 уникальных значений по корреляции с таргетом.''')

if st.checkbox('Количество уникальных значений в столбцах: '):
    for i in data.columns:
        st.write(f" ✨Столбец **{i}**: {data[i].nunique()}")

st.dataframe(data.corrwith(data['quality_enc']).sort_values())
st.write(''' Три наиболее значимых признака по корреляции с таргетом:
- **alcohol** -  содержание алкоголя %
- **density** -  плотность %, cвязана с содержанием сахара и алкоголя
- **volatile acidity** -  летучая кислотность %, которая влияет на запах и вкус''')









st.write('''
---
### 🔹 Визуализация данных''')

fig2 = px.scatter(wine, 
                  x="free sulfur dioxide", 
                  y="total sulfur dioxide", 
                  size="quality", 
                  color="quality",
                  title="Распределение Свободный SO₂ с Общий SO₂",
                  hover_name="quality",
                  size_max=15)
st.plotly_chart(fig2)


fig3 = px.histogram(wine, 
                    x="quality", 
                    title="Распределение качества Вина",
                    nbins=6)
st.plotly_chart(fig3)

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data["fixed acidity"],
            data["volatile acidity"], 
            data["alcohol"],
            c= data['quality_enc'],
            )

ax.set_xlabel("Fixed Acidity")
ax.set_ylabel("Volatile Acidity")
ax.set_zlabel("Alcohol", fontsize=15)
ax.set_title(" Wine Quality Dataset")
ax.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in data['quality_enc'].unique()])
st.pyplot(fig)

