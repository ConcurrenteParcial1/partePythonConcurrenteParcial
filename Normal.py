import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Normal:
    def __init__(self, mu=0, sigma=0.2, size=1000):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.data = np.random.normal(self.mu, self.sigma, self.size)

    def save_to_csv(self, filename='distribucion_normal.csv'):
        df = pd.DataFrame(self.data, columns=['valores'])
        df.to_csv(filename, index=False)

    def plot(self):
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        plt.plot(x, p, 'k', linewidth=2)
        plt.title('Distribución Normal')
        plt.ylabel('Densidad de probabilidad')
        plt.xlabel('Valores')
        plt.show()

    def compare_with_theoretical(self):
        theoretical_data = np.random.normal(self.mu, self.sigma, self.size)
        df = pd.DataFrame({
            'Generated': self.data,
            'Theoretical': theoretical_data
        })
        print(df.describe())

        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='g', label='Generated')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        plt.plot(x, p, 'k', linewidth=2, label='Theoretical')
        plt.legend()
        plt.title('Comparación de Distribuciones Normales')
        plt.ylabel('Densidad de probabilidad')
        plt.xlabel('Valores')
        plt.show()

normal_dist = Normal()
normal_dist.save_to_csv()
normal_dist.compare_with_theoretical()