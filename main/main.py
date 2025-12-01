import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --- Настройка страницы ---
st.set_page_config(page_title="Diamond Price Predictor", layout="centered")
st.title("💎 Прогнозирование цены бриллианта")


# --- Загрузка моделей и скалеров (предполагается, что они уже обучены и сохранены) ---
@st.cache_resource
def load_models_and_scalers():
    # Загрузка модели LinearRegression
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)

    # Загрузка модели Keras Sequential
    keras_model = load_model('keras_model.h5')

    # Загрузка скалеров
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)

    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    return linear_model, keras_model, scaler_X, scaler_y


# Загружаем все объекты один раз
try:
    linear_model, keras_model, scaler_X, scaler_y = load_models_and_scalers()
    st.success("✅ Модели и скалеры успешно загружены.")
except Exception as e:
    st.error(f"❌ Ошибка загрузки моделей или скалеров: {e}")
    st.stop()

# --- Ввод данных пользователя ---

st.header("📊 Введите параметры бриллианта")

# Числовые переменные (слайдеры)
carat = st.slider("Carat (вес)", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
depth = st.slider("Depth (%)", min_value=50.0, max_value=70.0, value=60.0, step=0.1)
table = st.slider("Table (%)", min_value=50.0, max_value=80.0, value=55.0, step=0.1)
X = st.slider("X (длина, мм)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
Y = st.slider("Y (ширина, мм)", min_value=3.0, max_value=10.0, value=5.0, step=0.01)
Z = st.slider("Z (глубина, мм)", min_value=2.0, max_value=6.0, value=3.0, step=0.01)

# Категориальные переменные (выпадающие списки)
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
cut = st.selectbox("Cut (огранка)", cut_options)

color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
color = st.selectbox("Color (цвет)", color_options)

clarity_options = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
clarity = st.selectbox("Clarity (чистота)", clarity_options)

# --- Обработка данных для предсказания ---
if st.button("🔮 Предсказать цену"):
    # Создаем DataFrame с введенными данными
    input_data = {
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'x': [X],  # Важно: в sklearn обычно используется lowercase 'x'
        'y': [Y],
        'z': [Z],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity]
    }
    df_input = pd.DataFrame(input_data)

    # Кодируем категориальные признаки (как при обучении!)
    # Это важно — должен быть тот же порядок и те же категории!
    df_input_encoded = pd.get_dummies(df_input, columns=['cut', 'color', 'clarity'], drop_first=False)

    # Убедимся, что все нужные столбцы присутствуют (если при обучении были другие категории — добавьте их)
    # Например, если при обучении было 5 значений cut, а здесь только 3 — нужно добавить недостающие колонки
    expected_columns = scaler_X.feature_names_in_  # Если вы сохранили feature_names_in_ при обучении
    for col in expected_columns:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # Переставляем столбцы в том же порядке, что и при обучении
    df_input_encoded = df_input_encoded[expected_columns]

    # Масштабируем входные данные
    X_scaled = scaler_X.transform(df_input_encoded)

    # --- Предсказание по линейной модели ---
    y_pred_linear_scaled = linear_model.predict(X_scaled)
    y_pred_linear = scaler_y.inverse_transform(y_pred_linear_scaled.reshape(-1, 1)).flatten()[0]

    # --- Предсказание по нейронной сети ---
    y_pred_keras_scaled = keras_model.predict(X_scaled, verbose=0)
    y_pred_keras = scaler_y.inverse_transform(y_pred_keras_scaled).flatten()[0]

    # --- Вывод результатов ---
    st.markdown("---")
    st.header("📈 Результаты")

    # Линейная модель
    st.subheader("🔹 Линейная модель")
    st.write(f"**Вычисленный Y (цена):** ${y_pred_linear:,.2f}")
    # R² не вычисляется динамически без тестовых данных — можно оставить как константу из обучения
    # Или рассчитать, если есть test set — но это требует дополнительного кода.
    # Для демонстрации просто покажем "R²: N/A" или заглушку.
    st.write("**R²:** *Не рассчитывается в реальном времени*")

    # Нейронная сеть
    st.subheader("🔹 Нейронная сеть")
    st.write(f"**Вычисленный Y (цена):** ${y_pred_keras:,.2f}")
    st.write("**R²:** *Не рассчитывается в реальном времени*")

    # Сравнение
    st.markdown("---")
    st.info(f"💡 Разница между моделями: ${abs(y_pred_linear - y_pred_keras):,.2f}")

# --- Подсказка ---
st.markdown("---")
st.caption(
    "⚠️ Убедитесь, что файлы `linear_model.pkl`, `keras_model.h5`, `scaler_X.pkl`, `scaler_y.pkl` находятся в той же папке, что и этот скрипт.")