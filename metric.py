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

st.title('Задания 7, 8, 9')

# объединение таблиц
red_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\user\Desktop\ds_course\HW5\source\winequality-white.csv", sep=";")
wine_data = pd.concat([red_wine, white_wine])
wine = wine_data.sample(6497).reset_index().drop(['index'], axis=1)


wine["quality_enc"] = wine["quality"].apply(lambda x: 0 if x in [3, 4, 5] else 1)

wine.drop(['quality'], axis=1, inplace=True)

X = wine.drop(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
       'pH', 'sulphates','quality_enc'], axis=1)
y = wine['quality_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


logistic = LogisticRegression(max_iter=565)
logistic.fit(X_train, y_train)
y_pred_logistic = logistic.predict(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


des_tree = DecisionTreeClassifier(max_depth=5)
des_tree.fit(X_train, y_train)
y_pred_des_tree = des_tree.predict(X_test)



st.write(''' ---
### 🔹 Визуализация границ решений
- 📈 Отобразите границы решений для каждого классификатора с помощью plot_decision_regions
- 🏷 Подпишите оси и добавьте заголовок.''')
st.write('Нужно подождать пару минуточек...')

from mlxtend.plotting import plot_decision_regions
X = wine[['density', 'alcohol']].values
X = scaler.fit_transform(X)
y = wine['quality_enc'].values.astype(np.int64)
fig2, ax = plt.subplots(1, 3, figsize=(18, 6))

plot_decision_regions(X, y, clf=knn, legend=2, ax=ax[0])
ax[0].set_xlabel("density")
ax[0].set_ylabel("alcohol")
ax[0].set_title("KNN")

plot_decision_regions(X, y, clf=logistic, legend=2, ax=ax[1])
ax[1].set_xlabel("density")
ax[1].set_ylabel("alcohol")
ax[1].set_title("Logistic Regression")

plot_decision_regions(X, y, clf=des_tree, legend=2, ax=ax[2])
ax[2].set_xlabel("density")
ax[2].set_ylabel("alcohol")
ax[2].set_title("Decision Tree")

plt.tight_layout()
st.pyplot(fig2)












st.write('''---
## 🔹 ROC-кривые
- 📉 Постройте ROC-кривые для каждого классификатора на одном графике. Для тестовой выборки
- 🏷 Подпишите оси и добавьте заголовок.''')

y_proba_log = logistic.predict_proba(X_test)[:, 1]
y_proba_knn= knn.predict_proba(X_test)[:, 1]
y_proba_tree = des_tree.predict_proba(X_test)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)

from sklearn.metrics import roc_curve, auc
roc_auc_log = auc(fpr_log, tpr_log)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_tree = auc(fpr_tree, tpr_tree)

fig3 = plt.figure(figsize=(8, 6))

plt.plot(fpr_knn, tpr_knn, label="KNN Classifier")
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_tree, tpr_tree, label="Decision Tree Classifier")

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Бейзлайн')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые для разных классификаторов")
plt.legend()
st.pyplot(fig3)











st.write('''---
## 🔹 Оценка качества классификации
- 🏆 Определите лучший метод классификации по показателю площади под ROC-кривой (AUC) на трейне и тесте. Сделайте выводы о переобчении, недообучении моделей''')

y_proba_knn_train = knn.predict_proba(X_train)[:, 1]
y_proba_knn_test = knn.predict_proba(X_test)[:, 1]

y_proba_log_train = logistic.predict_proba(X_train)[:, 1]
y_proba_log_test = logistic.predict_proba(X_test)[:, 1]

y_proba_tree_train = des_tree.predict_proba(X_train)[:, 1]
y_proba_tree_test = des_tree.predict_proba(X_test)[:, 1]

roc_auc_test_knn = roc_auc_score(y_test, y_proba_knn_test)
roc_auc_train_knn = roc_auc_score(y_train, y_proba_knn_train)

roc_auc_test_log = roc_auc_score(y_test, y_proba_log_test)
roc_auc_train_log = roc_auc_score(y_train, y_proba_log_train)

roc_auc_test_tree = roc_auc_score(y_test, y_proba_tree_test)
roc_auc_train_tree = roc_auc_score(y_train, y_proba_tree_train)

auc_results = pd.DataFrame({
    'Model': ['KNN_test', 'KNN_train', 'Logistic_test', 'Logistic_train', 'Des_Tree train', 'Des_Tree test'],
    'AUC': [roc_auc_test_knn, roc_auc_train_knn, roc_auc_test_log, roc_auc_train_log, roc_auc_train_tree, roc_auc_test_tree]
})

st.dataframe(auc_results)
st.write('''
**KNN**  
- На трейне метрика Auc дает высокие показатели, но на тесте показатель упал. Это говорит о том что модель явно переобучилась, она слишком запомнила данные трейна, но плохо работает на тесте

**Логистическая регрессия**  
- На трейне и на тесте модель дает почти одинаковые показатели , значит модель стабильная, а значит недообучения или переобучения нет  

**Дерево решений**  
- Аналогично с логистической регрессией


Исходя из этого лучшим методом классификации по показателю AUC - **логистическая регрессия**''')
