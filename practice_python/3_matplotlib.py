"""
    PRÁCTICA 1. ENTORNO DE TRABAJO Y REPASO DE PYTHON.

    PARTE 5: MATPLOTLIB.

    Dos gráficas sobre el conjunto iris. Cada función dibuja una figura, la guarda en el
    directorio figures/ con plt.savefig y la muestra con plt.show. Toda gráfica lleva
    título, etiquetas en los ejes y, si hay varias series, leyenda.

    Ejecute el script DESDE ESTE DIRECTORIO:   python 3_matplotlib.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# un color fijo para cada especie, el mismo en todas las gráficas
COLORES = {'setosa': 'tab:blue', 'versicolor': 'tab:orange', 'virginica': 'tab:green'}
# el nombre de cada columna de X, en orden: la columna j es NOMBRES[j]
NOMBRES = ['largo del sépalo (cm)', 'ancho del sépalo (cm)',
           'largo del pétalo (cm)', 'ancho del pétalo (cm)']


def cargar_iris(fichero):
    """
    Lee el fichero csv de iris y devuelve (X, y): X, de forma (150, 4), con las cuatro
    medidas de cada flor (una flor por fila), e y, de forma (150,), con el nombre de su
    especie. Se proporciona programada.
    """
    X = np.loadtxt(fichero, delimiter=',', skiprows=1, usecols=range(4))
    y = np.loadtxt(fichero, delimiter=',', skiprows=1, usecols=4, dtype=str)
    return X, y


def ejemplo_lineas(fichero):
    """
    El ejemplo del guión: dos curvas en la misma figura, con título, ejes, leyenda y rejilla.
    Se proporciona programado.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    plt.figure(figsize=(7, 4))
    plt.plot(x, np.sin(x), label='seno')
    plt.plot(x, np.cos(x), label='coseno', linestyle='--')
    plt.title('Dos funciones en una misma figura')
    plt.xlabel('x (radianes)')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(fichero, dpi=150, bbox_inches='tight')
    plt.show()


def barras_por_especie(X, y, j, fichero):
    """
    Diagrama de barras con la media del atributo j (la columna j de X) en cada especie: una
    barra por especie (con su color de COLORES) y el valor, con dos decimales, escrito
    encima de cada barra.
    Pistas: X[y == especie, j] son los valores del atributo j en las flores de esa especie;
    plt.bar(nombres, valores, color=...); plt.text(x, y, texto, ha='center').
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


def dispersion_por_especie(X, y, jx, jy, fichero):
    """
    Nube de puntos del atributo jy frente al atributo jx, con un color por especie y
    leyenda: una llamada a plt.scatter por especie, seleccionando sus flores con la
    máscara y == especie.
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    X, y = cargar_iris('data/iris.csv')
    print('X:', X.shape, '  y:', y.shape, '   esperado: (150, 4) y (150,)')
    ejemplo_lineas('figures/ejemplo_lineas.png')
    barras_por_especie(X, y, 2, 'figures/iris_barras.png')           # 2: largo del pétalo
    dispersion_por_especie(X, y, 2, 3, 'figures/iris_scatter.png')   # 3: ancho del pétalo
    print('Figuras guardadas en el directorio figures/')
