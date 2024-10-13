import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Exponencial:
    def __init__(self, scale=1.0, size=1000):
        # Inicializa la distribución exponencial con la escala y tamaño dados
        self.scale = scale
        self.size = size
        # Genera datos aleatorios siguiendo una distribución exponencial
        self.data = np.random.exponential(self.scale, self.size)

    def save_to_csv(self, filename='distribucion_exponencial.csv'):
        # Guarda los datos generados en un archivo CSV
        df = pd.DataFrame(self.data, columns=['valores'])
        df.to_csv(filename, index=False)

    def plot(self):
        # Grafica la función de densidad de probabilidad (PDF) de la distribución exponencial
        exponencial = stats.expon(scale=self.scale)
        x = np.linspace(exponencial.ppf(0.01), exponencial.ppf(0.99), 100)
        fp = exponencial.pdf(x)
        plt.plot(x, fp)
        plt.title('Distribución Exponencial')
        plt.ylabel('Probabilidad')
        plt.xlabel('Valores')
        plt.show()

    def compare_with_theoretical(self):
        # Genera datos teóricos para la comparación
        theoretical_data = np.random.exponential(self.scale, self.size)
        # Crea un DataFrame para comparar los datos generados y teóricos
        df = pd.DataFrame({
            'Generated': self.data,
            'Theoretical': theoretical_data
        })
        # Imprime un resumen estadístico de ambos conjuntos de datos
        print(df.describe())

        # Grafica el histograma de los datos generados
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g', label='Generated')
        # Grafica la PDF teórica en la misma gráfica
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        exponencial = stats.expon(scale=self.scale)
        fp = exponencial.pdf(x)
        plt.plot(x, fp, 'k', linewidth=2, label='Theoretical')
        plt.legend()
        plt.title('Comparación de Distribuciones Exponenciales')
        plt.ylabel('Densidad de probabilidad')
        plt.xlabel('Valores')
        plt.show()

# Crea una instancia de la clase Exponencial
exponential_dist = Exponencial()
# Guarda los datos generados en un archivo CSV
exponential_dist.save_to_csv()
# Compara los datos generados con la distribución teórica
exponential_dist.compare_with_theoretical()