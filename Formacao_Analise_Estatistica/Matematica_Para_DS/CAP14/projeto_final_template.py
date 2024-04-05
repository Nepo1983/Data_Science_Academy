# Projeto Final - Machine Learning Para Portfólio de Projetos em Data Science

# Import
import numpy as np

# Classe para definição do algoritmo
class FraudDetector:
    
    # Método construtor
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        
        # Inicializa os atributos da classe com os valores passados como argumento ou valor default
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Inicializa os coeficientes com None
        self.bias = None
        self.weigths = None

    # Método da função sigmóide para gerar a previsão no formato de probabilidade (valores entre 0 e 1)
    def sigmoid(self, pred):
        return 1 / (1 + np.exp(-pred))
        
        # Retorna a função com o argumento pred (previsao) passado como argumento ao chamar a função


    # Método de treinamento
    def fit(self, X, y):
        pass
        # Extrai do shape o número de linhas e de colunas do conjunto de dados
        n_records, n_attributes = X.shape

        # Inicializa a matriz de pesos com valores iguais a zero no mesmo shape do número de atributos
        self.pesos = np.zeros(n_attributes)

        # Inicializa o scalar bias com valor zero
        self.bias = 0

        # Otimização usando gradiente descendente
        for _ in range(self.n_iterations):
            
            # Faz a previsão usando o valor de X, pesos e bias
            prediction = np.dot(X, self.weigths) + self.bias
            
            # Converte a previsão no formato de probabilidade usando função sigmóide
            final_prediction = self.sigmoid(prediction)

            # Calcula os gradientes (derivadas da matriz de pesos e do bias)
            dw = 
            db = 

            # Atualiza os pesos e bias usando o valor das derivadas e a taxa de aprendizado


            # Fórmula: pesos - taxa-aprendizado x derivada dos pesos


            # Fórmula: bias - taxa-aprendizado x derivada do bias
        
    # Método para as previsões
    def predict(self, X):
        pass
        
        # Faz as previsões com novos dados de entrada e com os valores aprendidos de pesos e bias

        
        # Converte as previsões no formato de probabilidade usando função sigmóide

        
        # Aplica o cut-off e converte probabilidades para classes binárias (0 ou 1)

        
        return classe_prevista

# Treinamento do Modelo


# Dados de exemplo para treinar o modelo (você pode adaptar para seus próprios dados)


# Classe 0 = Não é Transação Suspeita
# Classe 1 = É Transação Suspeita

# Cria o modelo a partir da classe


# Treina o modelo


# Realiza previsões com novos dados


# Bloco if para avaliar os resultados


# Fim


