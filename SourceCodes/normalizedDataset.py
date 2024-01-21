import pandas as pd

# CSV dosyasını oku
csv_filename = 'DataSets/dataset.csv'  # Dataset CSV dosyanızın yolu
df = pd.read_csv(csv_filename)

# Giriş ve çıkış verilerini ayır
X = df[['x1', 'x2']].values
y = df['y'].values

# Verileri 100'e bölerek normalizasyon yap
X_normalized = X / 100.0
y_normalized = y / 100.0

# Normalize edilmiş verileri yeni bir DataFrame'e kaydet
normalized_df = pd.DataFrame({'x1_normalized': X_normalized[:, 0], 'x2_normalized': X_normalized[:, 1], 'y_normalized': y_normalized})

# Normalize edilmiş DataFrame'i CSV dosyasına kaydet
normalized_csv_filename = 'DataSets/normalizedDataset.csv'
normalized_df.to_csv(normalized_csv_filename, index=False)

print(f"Normalize edilmiş dataset kaydedildi: {normalized_csv_filename}")
