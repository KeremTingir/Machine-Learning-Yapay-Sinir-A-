import pandas as pd

# CSV dosyasını oku
df = pd.read_csv('Datasets/normalizedDataset.csv')

# x2_normalized sütununun tek sayı olup olmadığını kontrol ederek train ve test verilerini ayır
train_data = df[df['x2_normalized'] * 100 % 2 == 1]
test_data = df[df['x2_normalized'] *100 % 2 == 0]

# Giriş ve çıkış verilerini ayır
X_train = train_data[['x1_normalized', 'x2_normalized']].values
y_train = train_data['y_normalized'].values

X_test = test_data[['x1_normalized', 'x2_normalized']].values
y_test = test_data['y_normalized'].values

# Train ve test verilerini yazdır
print("Train Verileri:")
print("X_train:")
print(X_train)
print("y_train:")
print(y_train)

print("\nTest Verileri:")
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)

""" Eğitim ve test setleri arasındaki fark 0.01 * 0.01 gibi aynı değerler ve 
    iki tarafında tek iki tarafında çift olduğu değerler eğitim ve test olarak 
    iki parçaya ayrılamıyor fakat onun dışındaki değerler örneğin 0.01 * 0.02 eğitim 
    iken 0.02 * 0.01 test olarak iki farklı yere ayrılabiliyor böyleliklte eğitim ve 
    test verilerimiz mümkün olduğunca orantılı dağıtılarak doğruluk değerini arttırabiliyoruz.
"""