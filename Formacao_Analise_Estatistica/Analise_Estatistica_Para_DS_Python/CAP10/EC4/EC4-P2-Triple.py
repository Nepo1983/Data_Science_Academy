# Estudo de Caso 4 - Prevendo a Produção de Soja ao Longo do Tempo com Triple Exponential Smoothing

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Carregando o dataset
df = pd.read_csv('producao_soja.csv', parse_dates=['Data'])

# Plot da série temporal
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], df['Producao_Soja'])
plt.title('Produção de Soja em Toneladas')
plt.xlabel('Data')
plt.ylabel('Produção')
plt.grid(True)
plt.show()

# Aplicando o Triple Exponential Smoothing
# O parâmetro 'seasonal' pode ser 'add' para aditivo ou 'mul' para multiplicativo, dependendo da natureza da sazonalidade.
# O parâmetro 'seasonal_periods' é o número de períodos em uma estação (por exemplo, 12 para dados mensais).
modelo = ExponentialSmoothing(df['Producao_Soja'], seasonal = 'add', seasonal_periods = 12).fit()

# Fazendo previsões para os próximos 12 meses
forecast = modelo.forecast(steps=12)

# Preparando o período de data
forecast_dates = pd.date_range(start = df['Data'].iloc[-1] + pd.DateOffset(months = 1), periods = 12, freq = 'M')

# Plotando os resultados
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Producao_Soja'], label='Dados Históricos')
plt.plot(forecast_dates, forecast, color='red', label='Previsão')
plt.title('Produção de Soja / Previsões com Triple Exponential Smoothing')
plt.xlabel('Data')
plt.ylabel('Produção (Em Toneladas)')
plt.grid(True)
plt.legend()
plt.show()

print("Previsões para os próximos 12 meses:")
for date, value in zip(forecast_dates, forecast):
    print(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")