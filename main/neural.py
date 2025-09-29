import pandas as pd
import numpy  as np

import sklearn
from matplotlib.colors import ListedColormap
from pandas import DataFrame
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score

from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras import models


def generate_data():
    df = pd.read_csv(
        "../dataset/diamonds_nan_24.csv",
        sep='\t',
        decimal=',',
        header=0,
        na_values=['isntKnown']
    )
    print()
    print(df.columns)
    print()
    print(df.shape)
    print()
    print(df.isna().sum())
    print()

    # axis='index' - режим удаления строк, в которых есть пропуски.
    # axis='columns' - режим удаления столбцов, в которых есть пропуски.
    df = df.drop(
        labels=['ID'],  # список названий столбцов
        axis=1
    )
    df = df.dropna(axis='index', how='any')
    print(df.isna().sum())

    # print("Размер таблицы после исключения пропусков", df.shape)
    print("Итого: " + str(df.shape[0]) + " записи, " + str(df.shape[1]) + " столбец (признак).")
    df = df.replace(',', '.', regex=True)

    df_encoded = df.copy()
    # for col in df.columns:
    #     if df[col].dtype == 'object':  # только строковые столбцы
    #         le = LabelEncoder()
    #         df_encoded[col] = le.fit_transform(df[col])

    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # от худшего к лучшему
    df_encoded['color'] = df['color'].map({v: i + 1 for i, v in enumerate(color_order)})

    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    df_encoded['cut'] = df['cut'].map({v: i + 1 for i, v in enumerate(cut_order)})

    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    df_encoded['clarity'] = df['clarity'].map({v: i + 1 for i, v in enumerate(clarity_order)})
    df = df_encoded

    print(df[0:3])
    plotTableNA(df)

    obj_col_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']  # названия признаков в исходной таблицы в виде списка


    df_dummies = pd.get_dummies(
        data=df[obj_col_names],  # таблица с признаками для кодирования
        prefix=obj_col_names,  # сокращенные приставки к новым столбцам
        dtype=int,  # результат сравнения в виде целых чисел {0,1}
    )

    print(df_dummies[:3])


def plotTableNA(data_f, add_to_title =""):
    """  Функция для построения графика отображение отсутствующих значений """
    if   len(data_f.columns)> 20:
        k = 3
    elif len(data_f.columns)== 1:
        k = 1
    else:
        k = 1.7

    fig_size_h = 15        # размер полотна, высота
    fig_size_w =int(len(data_f.columns) / k) # размер полотна, ширина
    fig, ax = plt.subplots( figsize=(fig_size_w, fig_size_h)) # создать полотно для рисования, figsize-размер в дюймах

    # Отрисовать матрицу значений функцией imshow. Применяется цветовая палитра
    plt.imshow(data_f.isna(),
               cmap = ListedColormap([ '#3B5A92', 'white', ]),  # выбор цветовой шкалы, аналог cm.get_cmap('jet'),
               aspect='auto',  # ‘auto’ | ‘equal’ | scalar]    # режим соотношения сторон
               interpolation= 'none', )                         # отключить размытие
    ax.set_xticks(np.arange(len(data_f.columns)))
    ax.set_xticklabels(data_f.columns, rotation=-70, )
    plt.title("Графическое отображение отсутствующих значений (белые). " + add_to_title)
    plt.grid(False) # Сетка
    plt.show()
#-------------------------------------
# Вызов функции с указанием параметров


generate_data()

# model = models.Sequential()

def vectorize_sequences(df:DataFrame, dimension=10):
    result=np.zeros(len(df), dimension)
    for i, sequence in enumerate(df):
        result[i, sequence] = 1.
    return result


