# Estudo de Caso 4 - Prevendo a Produção de Soja ao Longo do Tempo com Modelagem de Séries Temporais

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Carregando o dataset
df = pd.read_csv('producao_soja.csv', parse_dates = ['Data'])

# Plot da série temporal
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], df['Producao_Soja'])
plt.title('Produção de Soja em Toneladas')
plt.xlabel('Data')
plt.ylabel('Produção')
plt.grid(True)
plt.show()
