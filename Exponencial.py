import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Exponencial:
    def __init__(self, scale=1.0, size=1000):
        self.scale = scale
        self.size = size
        self.data = np.random.exponential(self.scale, self.size)

    def save_to_csv(self, filename='distribucion_exponencial.csv'):
        df = pd.DataFrame(self.data, columns=['valores'])
        df.to_csv(filename, index=False)

    def plot(self):
        exponencial = stats.expon(scale=self.scale)
        x = np.linspace(exponencial.ppf(0.01), exponencial.ppf(0.99), 100)
        fp = exponencial.pdf(x)
        plt.plot(x, fp)
        plt.title('Distribución Exponencial')
        plt.ylabel('Probabilidad')
        plt.xlabel('Valores')
        plt.show()

    def compare_with_theoretical(self):
        theoretical_data = np.random.exponential(self.scale, self.size)
        df = pd.DataFrame({
            'Generated': self.data,
            'Theoretical': theoretical_data
        })
        print(df.describe())

        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g', label='Generated')
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

exponential_dist = Exponencial()
exponential_dist.save_to_csv()
exponential_dist.compare_with_theoretical()