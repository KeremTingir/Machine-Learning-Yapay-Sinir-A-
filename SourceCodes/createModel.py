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
        #Giriş Katmanı ile Gizli Katman Arasındaki Ağırlıklar
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) * 0.01
        # Gizli Katman Bias'ı
        self.bias_hidden = np.zeros((1, self.hidden_size))
        # Gizli Katman ile Çıkış Katmanı Arasındaki Ağırlıklar
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size) * 0.01
        # Çıkış Katman Bias'ı
        self.bias_output = np.zeros((1, self.output_size))

    # Sigmoid aktivasyon fonksiyonu
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid aktivasyon fonksiyonunun türevi
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # İleri hesaplama
    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

    # Geri hesaplama
    def backward(self, X, y):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Ağırlık ve bias güncellemeleri
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate

    # Modeli eğit
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    # Tahmin
    def predict(self, X):
        self.forward(X)
        return self.predicted_output


# Modeli oluştur
new_model = NeuralNetwork(input_size=2, hidden_size=1, output_size=1, learning_rate=0.1)

# Belirlenen değerlere göre modeli eğit
new_model.train(X_train, y_train.reshape(-1, 1), epochs=10000)

# Modeli kaydet
with open('model.pkl', 'wb') as model_file:
    pickle.dump(new_model, model_file)
    
# Test setini kullanarak modeli değerlendir
#predictions = new_model.predict(X_test)
#test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)  # y_test'i yeniden boyutlandırın
#print(f'Yeni Modelin Test Loss\'u: {test_loss}')
