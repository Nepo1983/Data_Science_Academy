# Projeto Final - Machine Learning Para Portfólio de Projetos em Data Science

# Import
import numpy as np

# Classe para definição do algoritmo
class AlgoritmoDSA:
    
    # Método construtor
    def __init__(self, taxa_aprendizado = 0.01, num_iteracoes = 1000):
        
        # Inicializa os atributos da classe com os valores passados como argumento ou valor default
        self.taxa_aprendizado = taxa_aprendizado
        self.num_iteracoes = num_iteracoes
        
        # Inicializa os coeficientes com None
        self.pesos = None
        self.bias = None

    # Método da função sigmóide para gerar a previsão no formato de probabilidade (valores entre 0 e 1)
    def sigmoid(self, pred):
        
        # Retorna a função com o argumento pred (previsao) passado como argumento ao chamar a função
        return 1 / (1 + np.exp(-pred))

    # Método de treinamento
    def fit(self, X, y):
        
        # Extrai do shape o número de linhas e de colunas do conjunto de dados
        num_registros, num_atributos = X.shape

        # Inicializa a matriz de pesos com valores iguais a zero no mesmo shape do número de atributos
        self.pesos = np.zeros(num_atributos)

        # Inicializa o scalar bias com valor zero
        self.bias = 0

        # Otimização usando gradiente descendente
        for _ in range(self.num_iteracoes):
            
            # Faz a previsão usando o valor de X, pesos e bias
            previsao = np.dot(X, self.pesos) + self.bias
            
            # Converte a previsão no formato de probabilidade usando função sigmóide
            previsao_final = self.sigmoid(previsao)

            # Calcula os gradientes (derivadas da matriz de pesos e do bias)
            dw = (1 / num_registros) * np.dot(X.T, (previsao_final - y))
            db = (1 / num_registros) * np.sum(previsao_final - y)

            # Atualiza os pesos e bias usando o valor das derivadas e a taxa de aprendizado

            # Fórmula: pesos - taxa-aprendizado x derivada dos pesos
            self.pesos -= self.taxa_aprendizado * dw

            # Fórmula: bias - taxa-aprendizado x derivada do bias
            self.bias -= self.taxa_aprendizado * db

    # Método para as previsões
    def predict(self, X):
        
        # Faz as previsões com novos dados de entrada e com os valores aprendidos de pesos e bias
        previsao = np.dot(X, self.pesos) + self.bias
        
        # Converte as previsões no formato de probabilidade usando função sigmóide
        previsao_final = self.sigmoid(previsao)
        
        # Aplica o cut-off e converte probabilidades para classes binárias (0 ou 1)
        classe_prevista = [1 if i > 0.5 else 0 for i in previsao_final]
        
        return classe_prevista

# Treinamento do Modelo

# Dados de exemplo para treinar o modelo (você pode adaptar para seus próprios dados)
X = np.array([[1, 2], [2, 3], [3, 5], [1, 4], [5, 6], [6, 7]])
y = np.array([0, 0, 1, 0, 1, 1])

# Classe 0 = Não é Transação Suspeita
# Classe 1 = É Transação Suspeita

# Cria o modelo a partir da classe
modelo_dsa = AlgoritmoDSA(taxa_aprendizado = 0.01, num_iteracoes = 1000)

# Treina o modelo
modelo_dsa.fit(X, y)

# Realiza previsões com novos dados
novos_dados = np.array([[1, 2], [4, 5]])
resultado = modelo_dsa.predict(novos_dados)

# Bloco if para avaliar os resultados
for i, res in enumerate(resultado):
    entrada = novos_dados[i]
    if res == 0:
        print(f"\nPara os atributos de entrada {entrada} a previsão foi (0): Não é Transação Suspeita")
    elif res == 1:
        print(f"\nPara os atributos de entrada {entrada} a previsão foi (1): É Transação Suspeita")
    else:
        print(f"\nPara os atributos de entrada {entrada} a previsão foi ({res}): Outra Classe")

print("\n")

# Fim


