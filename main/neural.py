import pandas as pd
import numpy  as np
import seaborn as sns

from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


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

    print(df.describe().T)

    df = df.drop(
        labels=['ID'],
        axis=1
    )
    plotTableNA(df)
    df = df.dropna(axis='index', how='any')
    print(df.isna().sum())

    print("Итого: " + str(df.shape[0]) + " записи, " + str(df.shape[1]) + " столбец (признак).")

    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    df['color'] = df['color'].map({v: i + 1 for i, v in enumerate(color_order)})

    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    df['cut'] = df['cut'].map({v: i + 1 for i, v in enumerate(cut_order)})

    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    df['clarity'] = df['clarity'].map({v: i + 1 for i, v in enumerate(clarity_order)})

    print(df[0:3])

    print(df.dtypes)

    corr_matrix = df.corr(method='pearson')
    print(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Матрица корреляции признаков")
    plt.show()


def plotTableNA(data_f, add_to_title =""):
    if   len(data_f.columns)> 20:
        k = 3
    elif len(data_f.columns)== 1:
        k = 1
    else:
        k = 1.7

    fig_size_h = 15
    fig_size_w =int(len(data_f.columns) / k) # размер полотна, ширина
    fig, ax = plt.subplots( figsize=(fig_size_w, fig_size_h)) # создать полотно для рисования, figsize-размер в дюймах

    plt.imshow(data_f.isna(),
               cmap = ListedColormap([ '#3B5A92', 'white', ]),  # выбор цветовой карты
               aspect='auto',  # ‘auto’ | ‘equal’ | scalar]    # режим соотношения сторон
               interpolation= 'none', )                         # отключить размытие
    ax.set_xticks(np.arange(len(data_f.columns)))
    ax.set_xticklabels(data_f.columns, rotation=-70, )
    plt.title("Графическое отображение отсутствующих значений (белые). " + add_to_title)
    plt.grid(False)
    plt.show()

generate_data()

# model = models.Sequential()


