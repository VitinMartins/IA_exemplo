import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Gerando dados fictícios para P=2, N=2000, C=2
np.random.seed(42)
X = np.random.randn(2000, 2)  # 2000 amostras, 2 características
y = np.random.randint(0, 2, 2000)  # Rótulos binários (0 ou 1)

# Visualizando os dados com gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Classe 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Classe 1')
plt.title('Gráfico de Dispersão dos Dados')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()

# Função para normalizar os dados
def normalize(X):
    # Normalização Min-Max
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Normalizando os dados de entrada
X_normalized = normalize(X)

# Função para dividir os dados em treinamento e teste
def train_test_split(X, y, test_size=0.2):
    n = X.shape[0]
    indices = np.random.permutation(n)
    test_set_size = int(n * test_size)
    
    # Índices para treino e teste
    train_indices = indices[test_set_size:]
    test_indices = indices[:test_set_size]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)

class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # Calculando a saída
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = 1 if linear_output >= 0 else 0
                
                # Atualizando pesos e bias
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

def monte_carlo_simulation(X, y, model, R=500, test_size=0.2):
    accuracies = []
    sensitivities = []
    specificities = []
    
    for _ in range(R):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Treinando o modelo
        model.fit(X_train, y_train)
        
        # Fazendo previsões
        y_pred = model.predict(X_test)
        
        # Calculando as métricas
        accuracy = np.mean(y_pred == y_test)
        sensitivity = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
        specificity = np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0)
        
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    return accuracies, sensitivities, specificities


# Construindo a matriz de confusão
def plot_confusion_matrix(y_true, y_pred):
    confusion = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        confusion[int(y_true[i]), int(y_pred[i])] += 1
    
    # Plotando a matriz de confusão
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matriz de Confusão')
    plt.show()

def calculate_metrics(accuracies, sensitivities, specificities):
    print("Acurácia - Média:", np.mean(accuracies), "Desvio padrão:", np.std(accuracies))
    print("Sensibilidade - Média:", np.mean(sensitivities), "Desvio padrão:", np.std(sensitivities))
    print("Especificidade - Média:", np.mean(specificities), "Desvio padrão:", np.std(specificities))
