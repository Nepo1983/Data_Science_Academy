# Estudo de Caso 4 - Prevendo a Produção de Soja ao Longo do Tempo com Simple Exponential Smoothing

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Carregando o dataset
df = pd.read_csv('producao_soja.csv', parse_dates= ['Data'])

# Plot da série temporal
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], df['Producao_Soja'])
plt.title('Produção de Soja em Toneladas')
plt.xlabel('Data')
plt.ylabel('Produção')
plt.grid(True)
plt.show()

# Aplicando a Suavização Exponencial
modelo = SimpleExpSmoothing(df['Producao_Soja']).fit(smoothing_level = 0.2, optimized = False)

# Suavização Exponencial Simples (SES) é um modelo que baseia suas previsões na média ponderada de observações passadas com pesos decrescentes 
# exponencialmente à medida que as observações envelhecem. No entanto, para previsões futuras além da última observação, 
# a SES prevê uma constante – a última saída prevista do modelo.

# Fazendo previsões para os próximos 12 meses
forecast = modelo.forecast(steps = 12)

# Preparando o período de data
forecast_dates = pd.date_range(start = df['Data'].iloc[-1] + pd.DateOffset(months = 1), periods = 12, freq = 'M')

# Plotando os resultados
plt.figure(figsize = (12, 6))
plt.plot(df['Data'], df['Producao_Soja'], label = 'Dados Históricos')
plt.plot(forecast_dates, forecast, color = 'red', label = "Previsão")
plt.title('Produção de Soja / Previsões')
plt.xlabel('Data')
plt.ylabel('Produção (Em Toneladas)')
plt.grid(True)
plt.legend()
plt.show()

print("Previsões para os próximos 12 meses:")
for date, value in zip(forecast_dates, forecast):
    print(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")



    
