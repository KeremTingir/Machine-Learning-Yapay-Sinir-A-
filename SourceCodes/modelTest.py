import pickle
import numpy as np
from createModel import NeuralNetwork

# Modeli yükle
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Test etmek istediğiniz girdi değerlerini belirleyin
x1_input = 7
x2_input = 10

# Girdi değerlerini normalize edin (veri setindeki maksimum değeri kullanarak)
x1_normalized = x1_input / 100.0
x2_normalized = x2_input / 100.0

# Giriş değerlerini kullanarak modelden tahmin alın
input_data = np.array([[x1_normalized, x2_normalized]])
predicted_result_normalized = loaded_model.predict(input_data)

# Tahmini sonucu orijinal ölçeğe dönüştürün
predicted_result = predicted_result_normalized * 100.0

print(f'Tahmin Edilen Sonuç: {predicted_result[0, 0]}')

"""  Örnek Çiktilari:
    9 * 8 = 71.81
    6 * 8 = 45.54
    5 * 9 = 45.56
    2 * 2 = 4.82
    3 * 2 = 5.82
    6 * 2 = 11.13
    7 * 2 = 14.13
    7 * 10 = 71.85
    
    
"""

