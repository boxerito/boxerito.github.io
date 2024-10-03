import sys
from pathlib import Path

# Añadir el directorio padre al PYTHONPATH
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from src.data import get_mouse_data, get_folds_tiers
from src.utils import get_best_model_path
from src.predictors import Predictor
from src import constants

def load_data(mouse, trial_id):
    mouse_data = get_mouse_data(mouse=mouse, splits=constants.folds_splits)
    trial_data = next((trial for trial in mouse_data['trials'] if trial['trial_id'] == trial_id), None)
    if trial_data is None:
        raise ValueError(f"No se encontró el trial_id {trial_id}")
    
    video = np.load(trial_data["video_path"])
    responses = np.load(trial_data["response_path"])
    behavior = np.load(trial_data["behavior_path"])
    pupil_center = np.load(trial_data["pupil_center_path"])
    length = trial_data["length"]
    
    return video, responses, behavior, pupil_center, length

def get_random_trial_from_fold(mouse, fold=0):
    tiers = get_folds_tiers(mouse, constants.num_folds)
    fold_trials = np.where(tiers == f'fold_{fold}')[0]
    if len(fold_trials) == 0:
        raise ValueError(f"No se encontraron trials para el fold {fold}")
    return np.random.choice(fold_trials)

def create_rasterplot(responses, title, num_neurons):
    plt.figure(figsize=(6, 4))
    cmap = plt.cm.get_cmap('viridis').reversed()
    vmin, vmax = 0, 10
    
    plt.imshow(responses, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=10)
    plt.xlabel('Time', fontsize=8)
    plt.ylabel(f'neurons\n{num_neurons}', fontsize=8)
    plt.yticks([0, responses.shape[0] - 1], ['0', str(num_neurons)])
    cbar = plt.colorbar(label='', ticks=[vmin, vmax])
    cbar.ax.set_yticklabels([str(vmin), str(vmax)], fontsize=8)
    plt.tight_layout()

def prepare_input(video, behavior, pupil_center, responses, length):
    # Recortar los datos a la longitud especificada
    video = video[:, :, :length]
    behavior = behavior[:, :length]
    pupil_center = pupil_center[:, :length]
    responses = responses[:, :length]
    
    print(f"Shapes después de recortar:")
    print(f"video: {video.shape}, behavior: {behavior.shape}, pupil_center: {pupil_center.shape}, responses: {responses.shape}")
    
    # Video tiene forma (H, W, T)
    H, W, T = video.shape
    
    # Transponemos el video a (1, T, H, W)
    video_transposed = video.transpose(2, 0, 1)[np.newaxis, :]
    
    # Ajustamos behavior y pupil_center para que tengan la forma (2, T, H, W)
    behavior_expanded = behavior[:, :, np.newaxis, np.newaxis].repeat(H, axis=2).repeat(W, axis=3)
    pupil_center_expanded = pupil_center[:, :, np.newaxis, np.newaxis].repeat(H, axis=2).repeat(W, axis=3)
    
    # Concatenamos los arrays
    input_tensor = np.concatenate([
        video_transposed,
        behavior_expanded,
        pupil_center_expanded
    ], axis=0)
    
    print(f"Shape del input_tensor: {input_tensor.shape}")
    
    return input_tensor, responses, length

def load_stored_prediction(mouse, trial_id):
    prediction_path = Path(f'/home/antoniofernandez/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/predictions/true_batch_001/out-of-fold/{mouse}/{trial_id}.npy')
    return np.load(prediction_path)

def compare_predictions(in_situ_pred, stored_pred, ground_truth):
    min_length = min(in_situ_pred.shape[1], stored_pred.shape[1], ground_truth.shape[1])
    in_situ_pred = in_situ_pred[:, :min_length]
    stored_pred = stored_pred[:, :min_length]
    ground_truth = ground_truth[:, :min_length]
    
    # Verificar NaN y valores infinitos
    print(f"NaN en in_situ_pred: {np.isnan(in_situ_pred).any()}")
    print(f"Inf en in_situ_pred: {np.isinf(in_situ_pred).any()}")
    print(f"Estadísticas de in_situ_pred: min={np.min(in_situ_pred):.4f}, max={np.max(in_situ_pred):.4f}, mean={np.mean(in_situ_pred):.4f}, std={np.std(in_situ_pred):.4f}")
    
    in_situ_stored_corr = np.corrcoef(in_situ_pred.reshape(-1), stored_pred.reshape(-1))[0, 1]
    in_situ_stored_mse = np.mean((in_situ_pred - stored_pred) ** 2)
    
    gt_in_situ_corr = np.corrcoef(ground_truth.reshape(-1), in_situ_pred.reshape(-1))[0, 1]
    gt_stored_corr = np.corrcoef(ground_truth.reshape(-1), stored_pred.reshape(-1))[0, 1]
    
    return in_situ_stored_corr, in_situ_stored_mse, gt_in_situ_corr, gt_stored_corr, min_length

def create_comparison_plot(ground_truth, in_situ_pred, stored_pred, title):
    min_length = min(ground_truth.shape[1], in_situ_pred.shape[1], stored_pred.shape[1])
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    vmin, vmax = 0, 10
    cmap = plt.cm.get_cmap('viridis').reversed()
    
    axs[0].imshow(ground_truth[:, :min_length], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth")
    
    axs[1].imshow(in_situ_pred[:, :min_length], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title("In-situ Prediction")
    
    axs[2].imshow(stored_pred[:, :min_length], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].set_title("Stored Prediction")
    
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Neurons')
    
    plt.tight_layout()
    plt.savefig(title)
    plt.close()

def main():
    mouse = constants.new_mice[0]  # Seleccionamos el primer ratón "new"
    fold = 0

    trial_id = get_random_trial_from_fold(mouse, fold)

    print(f"Analizando datos para el ratón {mouse}, trial {trial_id}, fold {fold}")
    video, responses, behavior, pupil_center, length = load_data(mouse, trial_id)

    print(f"Formas originales:")
    print(f"video shape: {video.shape}")
    print(f"responses shape: {responses.shape}")
    print(f"behavior shape: {behavior.shape}")
    print(f"pupil_center shape: {pupil_center.shape}")
    print(f"length: {length}")

    input_tensor, ground_truth, _ = prepare_input(video, behavior, pupil_center, responses, length)
    print(f"\nForma del input_tensor: {input_tensor.shape}")

    experiment_dir = Path('/home/antoniofernandez/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001')
    model = Predictor(get_best_model_path(experiment_dir / f'fold_{fold}'), device='cuda', blend_weights="ones")

    print("Realizando predicción in-situ...")
    in_situ_pred = model.predict_trial(
        video=video[:, :, :length],
        behavior=behavior[:, :length],
        pupil_center=pupil_center[:, :length],
        mouse_index=constants.mouse2index[mouse],
    )
    print(f"Forma de las respuestas predichas in-situ: {in_situ_pred.shape}")
    print(f"Estadísticas de in_situ_pred: min={np.min(in_situ_pred):.4f}, max={np.max(in_situ_pred):.4f}, mean={np.mean(in_situ_pred):.4f}, std={np.std(in_situ_pred):.4f}")

    # Cargar la predicción almacenada
    stored_pred = load_stored_prediction(mouse, trial_id)
    print(f"Forma de las respuestas predichas almacenadas: {stored_pred.shape}")

    # Comparar predicciones
    in_situ_stored_corr, in_situ_stored_mse, gt_in_situ_corr, gt_stored_corr, min_length = compare_predictions(in_situ_pred, stored_pred, ground_truth)
    print(f"\nComparación entre predicciones (usando los primeros {min_length} frames):")
    print(f"Correlación entre in-situ y almacenada: {in_situ_stored_corr:.4f}")
    print(f"Error Cuadrático Medio entre in-situ y almacenada: {in_situ_stored_mse:.4f}")
    print(f"Correlación entre Ground Truth y Predicción in-situ: {gt_in_situ_corr:.4f}")
    print(f"Correlación entre Ground Truth y Predicción almacenada: {gt_stored_corr:.4f}")

    # Limitar los valores para la visualización
    ground_truth = np.clip(ground_truth, 0, 10)
    in_situ_pred = np.clip(in_situ_pred, 0, 10)
    stored_pred = np.clip(stored_pred, 0, 10)

    # Crear rasterplots comparativos
    create_comparison_plot(ground_truth, in_situ_pred, stored_pred,
                           f"comparison_rasterplot_{mouse}_trial_{trial_id}.png")

    print("Rasterplot comparativo creado y guardado.")

if __name__ == "__main__":
    main()