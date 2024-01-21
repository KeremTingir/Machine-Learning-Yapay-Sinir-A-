import csv
import os

# Veri setini oluşturacak fonksiyon
def generate_dataset(rows, cols, output_folder):
    dataset = []
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            x1 = i
            x2 = j
            y = x1 * x2
            dataset.append((x1, x2, y))
    
    # Klasörü kontrol et ve yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return dataset

# Veri setini CSV dosyasına yazacak fonksiyon
def write_to_csv(dataset, output_folder):
    csv_filename = os.path.join(output_folder, 'dataset.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['x1', 'x2', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in dataset:
            writer.writerow({'x1': data[0], 'x2': data[1], 'y': data[2]})

# Veri setini oluştur
rows = 10
cols = 10
output_folder = 'DataSets'
dataset = generate_dataset(rows, cols, output_folder)

# Veri setini CSV dosyasına yaz
write_to_csv(dataset, output_folder)
