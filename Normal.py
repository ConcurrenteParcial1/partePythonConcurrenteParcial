import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Normal:
    def __init__(self, mu=0, sigma=0.2, size=1000):
        # Inicializa la distribución normal con la media (mu), desviación estándar (sigma) y tamaño dados
        self.mu = mu
        self.sigma = sigma
        self.size = size
        # Genera datos aleatorios siguiendo una distribución normal
        self.data = np.random.normal(self.mu, self.sigma, self.size)

    def save_to_csv(self, filename='distribucion_normal.csv'):
        # Guarda los datos generados en un archivo CSV
        df = pd.DataFrame(self.data, columns=['valores'])
        df.to_csv(filename, index=False)

    def plot(self):
        # Grafica el histograma de los datos generados
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g')
        # Define el rango del eje x
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        # Calcula la función de densidad de probabilidad (PDF) teórica de la distribución normal
        p = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        # Grafica la PDF teórica
        plt.plot(x, p, 'k', linewidth=2)
        plt.title('Distribución Normal')
        plt.ylabel('Densidad de probabilidad')
        plt.xlabel('Valores')
        plt.show()

    def compare_with_theoretical(self):
        # Genera datos teóricos para la comparación
        theoretical_data = np.random.normal(self.mu, self.sigma, self.size)
        # Crea un DataFrame para comparar los datos generados y teóricos
        df = pd.DataFrame({
            'Generated': self.data,
            'Theoretical': theoretical_data
        })
        # Imprime un resumen estadístico de ambos conjuntos de datos
        print(df.describe())

        # Grafica el histograma de los datos generados
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g', label='Generated')
        # Define el rango del eje x
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        # Calcula la PDF teórica de la distribución normal
        p = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        # Grafica la PDF teórica en la misma gráfica
        plt.plot(x, p, 'k', linewidth=2, label='Theoretical')
        plt.legend()
        plt.title('Comparación de Distribuciones Normales')
        plt.ylabel('Densidad de probabilidad')
        plt.xlabel('Valores')
        plt.show()

# Crea una instancia de la clase Normal
normal_dist = Normal()
# Guarda los datos generados en un archivo CSV
normal_dist.save_to_csv()
# Compara los datos generados con la distribución teórica
normal_dist.compare_with_theoretical()