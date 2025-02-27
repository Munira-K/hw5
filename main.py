import streamlit as st
import pandas as pd

pages = [
    st.Page('overview.py', title = 'Обзор'),
    st.Page('processing.py', title = 'Обработка'),
    st.Page('models.py', title = 'Модели'),
    st.Page('metric.py', title = 'Оценка')
]

pg_h = st.navigation(pages)
pg_h.run()