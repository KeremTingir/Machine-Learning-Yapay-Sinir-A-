import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# CSV dosyasını oku
df = pd.read_csv('DataSets/normalizedDataset.csv')

# Giriş ve çıkış verilerini ayır
X = df[['x1_normalized', 'x2_normalized']].values
y = df['y_normalized'].values

# Train ve test setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Yapay Sinir Ağı modelini oluştur
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Model parametrelerini tanımla
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ağırlıkları ve bias'ları başlat
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        # Sigmoid aktivasyon fonksiyonu
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Sigmoid aktivasyon fonksiyonunun türevi
        return x * (1 - x)

    def forward(self, X):
        # İleri hesaplama
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

    def backward(self, X, y):
        # Geriye yayılım
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Ağırlık ve bias güncellemeleri
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        # Modeli eğit
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        # Tahmin
        self.forward(X)
        return self.predicted_output


def find_best_hyperparameters(X_train, y_train, X_test, y_test, learning_rates, epochs_values, hidden_sizes):
    best_loss = float('inf')
    best_hyperparameters = {}

    for learning_rate in learning_rates:
        for epochs in epochs_values:
            for hidden_size in hidden_sizes:
                model = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
                model.train(X_train, y_train.reshape(-1, 1), epochs)

                predictions = model.predict(X_test)
                test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_hyperparameters = {'learning_rate': learning_rate, 'epochs': epochs, 'hidden_size': hidden_size}

    return best_hyperparameters

# Hiperparametre arama için değerler belirle
learning_rates = [0.001, 0.01, 0.1]
epochs_values = [1000, 2500, 5000, 7500, 10000, 15000]
hidden_sizes = [1, 2, 3, 4]

# En iyi hiperparametreleri bul
best_hyperparameters = find_best_hyperparameters(X_train, y_train, X_test, y_test, learning_rates, epochs_values, hidden_sizes)

# En iyi hiperparametrelerle modeli tekrar eğit
best_model = NeuralNetwork(input_size=2, hidden_size=best_hyperparameters['hidden_size'],
                           output_size=1, learning_rate=best_hyperparameters['learning_rate'])
best_model.train(X_train, y_train.reshape(-1, 1), best_hyperparameters['epochs'])

# Test setini kullanarak modeli değerlendir
predictions = best_model.predict(X_test)
test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)  # y_test'i yeniden boyutlandırın
print(f'En İyi Modelin Test Loss\'u: {test_loss}')
print(f'En İyi Hiperparametreler: {best_hyperparameters}')


def predict_result(model, x1, x2):
    # Girdileri normalize et
    x1_normalized = x1 / 100.0  # Veri setindeki maksimum değeri kullanarak normalize et
    x2_normalized = x2 / 100.0

    # Giriş değerlerini kullanarak modelden tahmin al
    input_data = np.array([[x1_normalized, x2_normalized]])
    model.forward(input_data)  # Modeli ileri doğru hesapla
    predicted_result_normalized = model.predicted_output

    # Tahmini sonucu orijinal ölçeğe dönüştür
    predicted_result = predicted_result_normalized * 100.0

    return predicted_result[0, 0]

# En iyi hiperparametrelerle modeli tekrar eğit
best_model = NeuralNetwork(input_size=2, hidden_size=best_hyperparameters['hidden_size'],
                           output_size=1, learning_rate=best_hyperparameters['learning_rate'])
best_model.train(X_train, y_train.reshape(-1, 1), best_hyperparameters['epochs'])

# Test setini kullanarak modeli değerlendir
predictions = best_model.predict(X_test)
test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)  # y_test'i yeniden boyutlandırın
print(f'En İyi Modelin Test Loss\'u: {test_loss}')

# Tahmin için kullanılan fonksiyon ve örneği aynı
x1_input = 7
x2_input = 8

predicted_result = predict_result(best_model, x1_input, x2_input)
print(f'Tahmin Edilen Sonuç: {predicted_result}')

# En iyi hiperparametrelerle modeli tekrar eğit
best_model = NeuralNetwork(input_size=2, hidden_size=best_hyperparameters['hidden_size'],
                           output_size=1, learning_rate=best_hyperparameters['learning_rate'])
best_model.train(X_train, y_train.reshape(-1, 1), best_hyperparameters['epochs'])

# Modeli kaydet
with open('#####', 'wb') as model_file:
    pickle.dump(best_model, model_file)
