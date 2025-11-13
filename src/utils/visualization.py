# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(preds_real, preds_imag, targets_real, targets_imag, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(targets_real + 1j * targets_imag, label='True', marker='o')
    plt.plot(preds_real + 1j * preds_imag, label='Predicted', marker='x')
    plt.legend()
    plt.title("Complex Predictions (Real + i Imag)")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def to_polar(real, imag):
    magnitude = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)
    return magnitude, phase


def plot_polar(preds_real, preds_imag, targets_real, targets_imag, save_path=None):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')

    true_angles = np.angle(targets_real + 1j * targets_imag)
    true_magnitudes = np.abs(targets_real + 1j * targets_imag)

    pred_angles = np.angle(preds_real + 1j * preds_imag)
    pred_magnitudes = np.abs(preds_real + 1j * preds_imag)

    ax.scatter(true_angles, true_magnitudes, label='True', alpha=0.7)
    ax.scatter(pred_angles, pred_magnitudes, label='Predicted', alpha=0.7)
    ax.legend()
    plt.title("Polar Plot of Complex Values")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_predictions_2(preds_real, preds_imag, targets_real, targets_imag):
    plt.figure(figsize=(10,5))

    # Комплексная плоскость
    plt.subplot(1, 2, 1)
    plt.scatter(targets_real, targets_imag, color='blue', label='True', alpha=0.5)
    plt.scatter(preds_real, preds_imag, color='red', label='Predicted', alpha=0.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Complex plane')
    plt.legend()

    # Реальная и мнимая части по времени
    plt.subplot(1, 2, 2)
    plt.plot(targets_real, label='True real', color='blue')
    plt.plot(preds_real, label='Pred real', color='red', linestyle='--')
    plt.plot(targets_imag, label='True imag', color='cyan')
    plt.plot(preds_imag, label='Pred imag', color='magenta', linestyle='--')
    plt.title('Real & Imag parts')
    plt.legend()

    plt.tight_layout()
    plt.show()


from utils.sheduler import get_sсheduler
from utils.early_stopping import EarlyStopping
from utils.visualization import plot_predictions, plot_polar


plot_predictions(preds_real, preds_imag, targets_real, targets_imag)
plot_polar(preds_real, preds_imag, targets_real, targets_imag)

# Сохраняем графики локально
os.makedirs("mlruns/plots", exist_ok=True)
plot_predictions(preds_real, preds_imag, targets_real, targets_imag, save_path="mlruns/plots/predictions.png")
plot_polar(preds_real, preds_imag, targets_real, targets_imag, save_path="mlruns/plots/polar.png")

# Логгируем артефакты в MLflow
mlflow.log_artifact("mlruns/plots/predictions.png")
mlflow.log_artifact("mlruns/plots/polar.png")

# Логгируем модель
mlflow.pytorch.log_model(model, "model")

# Сохраняем графики и логгируем их в MLflow
os.makedirs("mlruns/plots_finetune", exist_ok=True)
plot_predictions(preds_real, preds_imag, targets_real, targets_imag, save_path="mlruns/plots_finetune/predictions_ft.png")
plot_polar(preds_real, preds_imag, targets_real, targets_imag, save_path="mlruns/plots_finetune/polar_ft.png")

mlflow.log_artifact("mlruns/plots_finetune/predictions_ft.png")
mlflow.log_artifact("mlruns/plots_finetune/polar_ft.png")

# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(preds_real, preds_imag, targets_real, targets_imag, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(targets_real + 1j * targets_imag, label='True', marker='o')
    plt.plot(preds_real + 1j * preds_imag, label='Predicted', marker='x')
    plt.legend()
    plt.title("Complex Predictions (Real + i Imag)")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_polar(preds_real, preds_imag, targets_real, targets_imag, save_path=None):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')

    true_angles = np.angle(targets_real + 1j * targets_imag)
    true_magnitudes = np.abs(targets_real + 1j * targets_imag)

    pred_angles = np.angle(preds_real + 1j * preds_imag)
    pred_magnitudes = np.abs(preds_real + 1j * preds_imag)

    ax.scatter(true_angles, true_magnitudes, label='True', alpha=0.7)
    ax.scatter(pred_angles, pred_magnitudes, label='Predicted', alpha=0.7)
    ax.legend()
    plt.title("Polar Plot of Complex Values")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Допустим, у нас есть временной ряд
data = pd.read_csv('your_timeseries.csv')  # Например, столбец "close"
ts = data['close'].values

# Стандартизация временного ряда
scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts.reshape(-1, 1)).flatten()

# Параметры
window_size = 20
stride = 1

eigenvalues_dataset = []

for i in range(0, len(ts_scaled) - window_size, stride):
    window = ts_scaled[i:i+window_size]

    # Формируем симметричную матрицу (ковариационная)
    window_matrix = np.cov(window.reshape(-1, 1), rowvar=False)

    # Получаем собственные значения (они будут вещественными)
    eigvals = np.linalg.eigvalsh(window_matrix)

    eigenvalues_dataset.append(eigvals)

eigenvalues_dataset = np.array(eigenvalues_dataset)

print("Размерность набора собственных чисел:", eigenvalues_dataset.shape)
print("Пример λ для одного окна:", eigenvalues_dataset[0])