import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros de la distribución normal
mu, sigma = 0, 0.2

# Generar 1000 datos con distribución normal
data = np.random.normal(mu, sigma, 1000)

# Crear un DataFrame de pandas
df = pd.DataFrame(data, columns=['valores'])

# Guardar el DataFrame en un archivo CSV
df.to_csv('distribucion_normal.csv', index=False)

# Representar gráficamente los datos generados
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Añadir la curva de la distribución normal teórica
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)

plt.title('Distribución Normal')
plt.ylabel('Densidad de probabilidad')
plt.xlabel('Valores')
plt.show()