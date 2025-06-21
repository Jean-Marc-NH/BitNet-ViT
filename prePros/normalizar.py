import numpy as np

def process_csv_to_bin(csv_path, x_bin_path, y_bin_path):
    print(f"Procesando: {csv_path}")

    # Cargar CSV completo
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

    # Separar etiqueta (columna 0) y pixeles (columnas 1:)
    y = data[:, 0].astype(np.uint8)        # Etiquetas enteras
    X = data[:, 1:] / 255.0                # Normalización a [0.0, 1.0]

    print(f"→ {X.shape[0]} muestras, {X.shape[1]} características por muestra")

    # Guardar binarios
    X.tofile(x_bin_path)
    y.tofile(y_bin_path)

    print(f"✅ Guardado: {x_bin_path}, {y_bin_path}\n")

# Rutas a tus CSV
train_csv = "fer2013_train_shuffled.csv"
test_csv  = "fer2013_test_shuffled.csv"

# Procesar y exportar binarios
process_csv_to_bin(train_csv, "X_train.bin", "y_train.bin")
process_csv_to_bin(test_csv,  "X_test.bin",  "y_test.bin")
