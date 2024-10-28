import sys
from pathlib import Path
import random
import json
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Añadir el directorio padre a PYTHONPATH
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import argus
from src.data import get_mouse_data
from src.predictors import Predictor
from src import constants
import utils_reconstruction.image_similarity as im_sim
import utils_reconstruction.utils_reconstruction as utils

print('\n')
print('------------')

## Parámetros
track_iter = 10
plot_iter = track_iter * 1
fold_number = 0  # Definir el número de fold
data_fold = f'fold_{fold_number}'
randomize_models = False  # Si es True, se generan modelos aleatorios; si es False, se usan los modelos proporcionados
num_random_models = 2  # Número de modelos aleatorios a generar si randomize_models es True
user_model_list = np.array([0,1,2,3,4,5,6])  # Lista de modelos proporcionada por el usuario

# Generar lista de modelos
if randomize_models:
    model_list = np.random.choice(range(7), num_random_models, replace=False)
else:
    model_list = user_model_list

number_models = model_list.shape[0]
animals = range(4, 5)  # Solo incluye el índice 4
start_trial = 8  # Se refiere al índice del trial en el fold seleccionado
end_trial = 9  # No incluye este
random_trials = False  # Si es True, se eligen trials aleatorios; por defecto es False
video_length = None  # Máximo es 300, pero puede ser demasiado para algunas GPUs
load_skip_frames = 0  # En caso de que se deba omitir el inicio del video; por defecto es 0

# Validar que data_fold no está en model_list
def some_function(fold_number, model_list, check_data_fold=False):
    data_fold_index = fold_number  # Dado que data_fold es 'fold_{fold_number}'

    if check_data_fold:
        if data_fold_index in model_list:
            raise ValueError("data_fold no puede ser el mismo que cualquier valor en model_list.")

some_function(fold_number, model_list)

print(f'Model list: {model_list}')
print(f'Number of models: {number_models}')

population_reduction = 0.9  # Porcentaje de neuronas a ablar/eliminar/silenciar
optimize_given_predicted_responses = False
randomize_neurons = False  # Establecer en True para aleatorizar neuronas

subbatch_size = 32
minibatch = 8
epoch_number_first = 1000
n_steps = 1
epoch_reducer = 1
strides_all, epoch_switch = utils.stride_calculator(
    minibatch=minibatch,
    n_steps=n_steps,
    epoch_number_first=epoch_number_first,
    epoch_reducer=epoch_reducer
)
print('strides_all: ' + str(strides_all) + '; epoch_switch: ' + str(epoch_switch))

mask_update_th = 0.5
mask_eval_th = 1

vid_init = 'gray'
lr = 1000
lr_warmup_epochs = 10
loss_func = 'poisson'
use_adam = True
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-8
with_gradnorm = True
clip_grad = 1
eval_frame_skip = 32

response_dropout_rate = 0
drop_method = 'zero_pred_n_true'
input_noise = 0
pix_decay_rate = 0

save_path_suffix = 'hpc_round5'
if optimize_given_predicted_responses:
    save_path_suffix += '_fit_to_predicted'
if population_reduction > 0:
    save_path_suffix += '_popreduc'

save_path = f'reconstructions/modelfold{model_list}_data{data_fold}_pop{round(100 - population_reduction * 100)}_{save_path_suffix}/'

Path(save_path).mkdir(parents=True, exist_ok=True)
print('save path:' + save_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ' + str(device))

print('\nLoading a model')
model_path = [None] * 7
model = [None] * number_models

model_path[0] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_0/model-000-0.290928.pth').expanduser()
model_path[1] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_1/model-000-0.292576.pth').expanduser()
model_path[2] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_2/model-000-0.291243.pth').expanduser()
model_path[3] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_3/model-000-0.290196.pth').expanduser()
model_path[4] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_4/model-000-0.289216.pth').expanduser()
model_path[5] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_5/model-000-0.288470.pth').expanduser()
model_path[6] = Path('~/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001/fold_6/model-000-0.289128.pth').expanduser()

for n in range(len(model_list)):
    print('Loading model for reconstruction: ', model_path[model_list[n]])
    model[n] = argus.load_model(model_path[model_list[n]], device=device, optimizer=None, loss=None)
    model[n].eval()

if optimize_given_predicted_responses:
    predictor = [None] * len(model_path)
    for n in range(len(predictor)):
        model_path_temp = model_path[n]
        print('Predicting true responses with model: ', model_path_temp)
        predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")
else:
    predictor = [None] * number_models
    for n in range(len(predictor)):
        model_path_temp = model_path[n]
        print('Predicting true responses with model: ', model_path_temp)
        predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")

# Diagnóstico inicial
print("\nDiagnostic Information:")
print(f"Number of predictors: {len(predictor)}")
print(f"Population reduction: {population_reduction}")

json_file_path = Path('folds_trials_new_mice.json')
with open(json_file_path, 'r') as f:
    json_data = json.load(f)
print('\nGetting a batch')
for mouse_index in animals:
    mouse = constants.index2mouse[mouse_index]
    mouse_key = mouse
    mouse_data = get_mouse_data(mouse=mouse, splits=[data_fold])
    trial_data_all = mouse_data['trials']
    print('Total trials: ' + str(len(trial_data_all)))

    fold_of_interest = str(fold_number)
    available_trials = json_data[mouse_key].get(fold_of_interest, [])

    if not available_trials:
        raise ValueError(f"No trials available for mouse {mouse_key} in fold {fold_of_interest} in JSON")

    print(f"Available trials for mouse {mouse_key} in JSON in fold {fold_of_interest}: {available_trials}")

    trial_id_to_index = {trial_data['trial_id']: idx for idx, trial_data in enumerate(trial_data_all)}
    valid_trials = [t for t in available_trials if t in trial_id_to_index]

    if not valid_trials:
        raise ValueError(f"No valid trials exist both in JSON and trial_data_all for mouse {mouse_key} in fold {fold_of_interest}")

    print(f"Valid trials: {valid_trials}")

    mask = np.load(f'reconstructions/masks/mask_m{mouse_index}.npy')
    mask_update = torch.tensor(np.where(mask >= mask_update_th, 1, 0)).to(device)
    mask_eval = torch.tensor(np.where(mask >= mask_eval_th, 1, 0)).to(device)
    print('Mask shape: ', mask.shape)

    if random_trials:
        trials_to_process = random.sample(valid_trials, min(end_trial - start_trial, len(valid_trials)))
    else:
        trials_to_process = valid_trials[start_trial:end_trial]

    for trial in trials_to_process:
        torch.cuda.empty_cache()
        trial_index = trial_id_to_index[trial]
        trial_data = trial_data_all[trial_index]
        print(f'Processing trial {trial}, index {trial_index}')
        print('Trial paths:')
        print(trial_data)

        if video_length is None:
            video_length = trial_data["length"]
        inputs, responses, video, behavior, pupil_center = utils.load_trial_data(
            trial_data_all, model[0], trial_index, load_skip_frames, length=video_length)
        inputs = inputs.to(device)
        responses = responses.to(device)

        print("\nDiagnostic - Initial shapes:")
        print(f"responses shape: {responses.shape}")
        print(f"inputs shape: {inputs.shape}")

        if randomize_neurons:
            manual_seed = (trial + 1) * (mouse_index + 1)
            print(f'Random seed para aleatorizar neuronas: {manual_seed}')
            torch.manual_seed(manual_seed)
            responses = responses[torch.randperm(responses.shape[0]), :]

        if population_reduction > 0:
            manual_seed = (trial + 1) * (mouse_index + 1)
            print(f'Random seed para reducción de población: {manual_seed}')
            torch.manual_seed(manual_seed)
            population_mask = torch.rand((responses.shape[0], 1), device=device) > population_reduction
            responses = responses * population_mask.repeat(1, responses.shape[1])
            print(f"\nDiagnostic - Population mask shape: {population_mask.shape}")
        else:
            population_mask = None

        # Identificar neuronas silenciadas usando population_mask
        if population_mask is not None:
            silenced_neurons = np.where(population_mask.cpu().numpy().flatten() == 0)[0]
        else:
            silenced_neurons = []

        # Expand masks si es necesario
        mask_update_expanded = mask_update.repeat(1, 1, inputs.shape[1], 1, 1)
        mask_eval_expanded = mask_eval.repeat(1, 1, inputs.shape[1], 1, 1)
        responses_predicted_original = torch.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)), device=device)

        print("\nDiagnostic - Before predictions:")
        print(f"responses_predicted_original initial shape: {responses_predicted_original.shape}")

        for n in range(len(predictor)):
            prediction = predictor[n].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            print(f"Predicción {n} shape: {prediction.shape}")  # Debe ser (8122, 300)
            responses_predicted_original[:, :, n] = torch.from_numpy(prediction).to(device)

        print("\nDiagnostic - Before population mask application:")
        print(f"responses_predicted_original shape: {responses_predicted_original.shape}")

        if population_mask is not None:
            print("\nDiagnostic - Population mask details:")
            print(f"population_mask shape: {population_mask.shape}")
            mask_torch = population_mask  # Ya es un tensor en el dispositivo
            print(f"Forma de la máscara para multiplicación: {mask_torch.shape}")

            # Realizar la multiplicación directamente con tensores
            responses_predicted_original = responses_predicted_original * mask_torch[:, :, None]
            print("\nDiagnostic - After population mask application:")
            print(f"responses_predicted_original shape after mask: {responses_predicted_original.shape}")

        print("\nDiagnostic - Before loss calculation:")
        print(f"responses_predicted_original shape: {responses_predicted_original.shape}")

        # Calcular la media correctamente
        responses_predicted_mean = responses_predicted_original.mean(axis=2)  # Resultado: (8122, 300)
        print(f"responses_predicted_mean shape: {responses_predicted_mean.shape}")
        print(f"responses shape: {responses.shape}")

        # Verificar que las formas coinciden antes de la pérdida
        if responses_predicted_mean.shape != responses.shape:
            raise ValueError(f"Mismatch shapes: responses_predicted_mean {responses_predicted_mean.shape} vs responses {responses.shape}")

        # Calcular la función de pérdida
        loss_gt = utils.response_loss_function(
            responses_predicted_mean,
            responses.clone().detach(),
            mask=population_mask
        )
        print(f'Ground truth test loss: {loss_gt.item()}')

        # Calcular la correlación
        response_corr_gt = np.corrcoef(
            responses.cpu().detach().numpy().flatten(),
            responses_predicted_mean.cpu().detach().numpy().flatten()
        )[0, 1]
        print(f'Response correlation ground truth: {response_corr_gt}')

        if optimize_given_predicted_responses:
            alternative_response_gt = responses_predicted_mean.clone().detach()

        # Generar predicciones con pantalla gris
        responses_predicted_gray = torch.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)), device=device)
        print("\nDiagnostic - Gray predictions:")
        print(f"responses_predicted_gray initial shape: {responses_predicted_gray.shape}")

        for n in range(len(predictor)):
            prediction_gray = predictor[n].predict_trial(
                video=np.ones_like(video) * (255 / 2),
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            print(f"Predicción (gray) {n} shape: {prediction_gray.shape}")
            responses_predicted_gray[:, :, n] = torch.from_numpy(prediction_gray).to(device)

        if population_mask is not None:
            print("\nDiagnostic - Before gray population mask application:")
            print(f"responses_predicted_gray shape: {responses_predicted_gray.shape}")
            responses_predicted_gray = responses_predicted_gray * mask_torch[:, :, None]
            print(f"responses_predicted_gray shape after mask: {responses_predicted_gray.shape}")

        # Inicializar predictores con gradientes
        predictor_withgrads = [None] * number_models
        for n in range(number_models):
            predictor_withgrads[n] = utils.Predictor_JB(model[n], mouse_index, withgrads=True, dummy=False)
        print('\nSkipping gradient tracking test')

        print('Response range: ' + str(responses.min().cpu().detach().numpy()) + ' to ' + str(responses.max().cpu().detach().numpy()))
        print('Video range: ' + str(inputs[0, :].min().cpu().detach().numpy()) + ' to ' + str(inputs[0, :].max().cpu().detach().numpy()))

        # Crear figuras para visualización
        fig, axs = plt.subplots(4, 4, figsize=(20, 20), dpi=100)
        fig.suptitle(f'mouse {mouse_index} trial {trial} model {model_list}', fontsize=16)

        # Visualización de los frames inicial y final
        axs[0, 0].imshow(np.concatenate((video[:, :, 0], video[:, :, -1]), axis=1), cmap='gray')
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Frame 0 and ' + str(video.shape[2]))

        # Visualización del comportamiento
        axs[1, 0].plot(behavior[:, :].T)
        axs[1, 0].set_title('Behavior')

        # Visualización de la posición de la pupila
        axs[2, 0].plot(pupil_center[:, :].T)
        axs[2, 0].set_title('Pupil position')

        # Preparar los datos de respuestas para la visualización
        responses_visual = responses.cpu().detach().numpy().copy()

        # Crear un mapa de colores personalizado
        cmap = plt.cm.viridis  # Puedes elegir el mapa de colores que prefieras
        cmap_modified = cmap(np.arange(cmap.N))
        cmap_modified[0, :] = [0, 0, 0, 1]  # Establecer el color para el valor más bajo (neuronas silenciadas) a negro
        cmap_modified[1:, :] = cmap_modified[1:, :] * 1.2  # Aumentar brillo
        cmap_modified[1:, :] = np.clip(cmap_modified[1:, :], 0, 1)  # Mantener valores válidos
        custom_cmap = ListedColormap(cmap_modified)

        # Establecer las respuestas de las neuronas silenciadas a un valor negativo para visualizarlas en negro
        if population_mask is not None:
            responses_visual[silenced_neurons, :] = -1  # Valor negativo para distinguir en el rasterplot

        # Crear el rasterplot con actividades
        axs[1, 1].imshow(responses_visual, aspect='auto', vmin=-1, vmax=10, cmap=custom_cmap, interpolation='none')
        axs[1, 1].set_title('Responses (silenced neurons in black)')
        axs[1, 1].text(1.05, 0.5, 'Silenced Neurons', transform=axs[1, 1].transAxes,
                       fontsize=10, color='black', va='center', ha='left')

        # Respuestas predichas con superposición de máscara
        responses_predicted_mean_np = responses_predicted_mean.cpu().detach().numpy().copy()
        if population_mask is not None:
            responses_predicted_mean_np[silenced_neurons, :] = -1  # Valor negativo para distinguir en el rasterplot

        axs[2, 1].imshow(responses_predicted_mean_np, aspect='auto', vmin=-1, vmax=10, cmap=custom_cmap, interpolation='none')
        axs[2, 1].set_title('Predicted Responses (silenced neurons in black')

        # Respuestas predichas con pantalla gris y superposición de máscara
        responses_predicted_gray_np = responses_predicted_gray.mean(axis=2).cpu().detach().numpy()
        if population_mask is not None:
            responses_predicted_gray_np[silenced_neurons, :] = -1  # Valor negativo para distinguir en el rasterplot

        axs[3, 0].imshow(responses_predicted_gray_np, aspect='auto', vmin=-1, vmax=10, cmap=custom_cmap, interpolation='none')
        axs[3, 0].set_title('Predicted Responses Gray (silenced neurons in black)')

        # Energía de movimiento del video
        axs[0, 2].plot(np.sum(np.abs(np.diff(video, axis=2)), axis=(0, 1)))
        axs[0, 2].set_title('Video motion energy')

        randneurons_suffix = '_randneurons' if randomize_neurons else ''
        fig.savefig(save_path + f'reconstruction_summary_m{mouse_index}_t{trial}{randneurons_suffix}.png')

        print('\n')
        video_pred = utils.init_weights(inputs, vid_init, device)
        print('Shape of video_pred: ' + str(video_pred.shape))

        print('\n')
        video_corr = []
        video_RMSE = []
        video_iter = []
        loss_all = []
        response_corr = []
        frame_corr = []  # Lista para almacenar la correlación frame a frame
        frame_RMSE = []  # Lista para almacenar el RMSE frame a frame

        print(f'Reconstructing mouse {mouse_index} trial {trial}')
        progress_bar = tqdm(range(epoch_switch[-1]))
        start_training_time = time.time()

        for i in progress_bar:
            for n in range(n_steps):
                if i < epoch_switch[n]:
                    subbatch_shift = strides_all[n]
                    break

            number_of_subbatches = 2 + (inputs.shape[1] - subbatch_size) // subbatch_shift

            gradients_fullvid = torch.zeros_like(video_pred).repeat(number_models, number_of_subbatches, 1, 1, 1, 1).to(device).requires_grad_(False)
            gradients_fullvid = gradients_fullvid.fill_(float('nan'))
            gradnorm = torch.zeros(number_models, number_of_subbatches).to(device)
            for n in range(number_models):
                for subbatch in range(number_of_subbatches):
                    if subbatch == number_of_subbatches - 1:
                        start_frame = inputs.shape[1] - subbatch_size
                        end_frame = inputs.shape[1]
                    else:
                        start_frame = subbatch * subbatch_shift
                        end_frame = subbatch * subbatch_shift + subbatch_size
                    subbatch_frames = range(start_frame, end_frame)

                    if input_noise > 0:
                        added_video_noise = torch.randn_like(video_pred[:, :, subbatch_frames, :, :]) * input_noise
                    else:
                        added_video_noise = torch.zeros_like(video_pred[:, :, subbatch_frames, :, :])

                    input_prediction = utils.cat_video_behaviour(
                        video_pred[:, :, subbatch_frames, :, :] + added_video_noise,
                        inputs[None, 1:, subbatch_frames, :, :]
                    ).detach().requires_grad_(True)

                    responses_predicted_new = predictor_withgrads[n](input_prediction)

                    if optimize_given_predicted_responses:
                        loss = utils.response_loss_function(
                            responses_predicted_new,
                            alternative_response_gt[:, subbatch_frames].clone().detach(),
                            mask=population_mask
                        )
                    else:
                        loss = utils.response_loss_function(
                            responses_predicted_new,
                            responses[:, subbatch_frames].clone().detach(),
                            mask=population_mask
                        )

                    loss.backward()
                    gradients = input_prediction.grad

                    gradnorm[n, subbatch] = torch.norm(gradients)
                    if with_gradnorm:
                        gradients = gradients / (gradnorm[n, subbatch] + 1e-6)
                    else:
                        gradients = gradients * 100

                    gradients_fullvid[n, subbatch, :, subbatch_frames, :, :] = gradients[:, 0:1, :, :, :]

            gradients_fullvid = torch.nanmean(gradients_fullvid, axis=1, keepdim=True)
            gradients_fullvid = torch.nanmean(gradients_fullvid, axis=0, keepdim=False)
            gradients_fullvid = gradients_fullvid * mask_update_expanded

            if clip_grad is not None:
                gradients_fullvid = torch.clip(gradients_fullvid, -1 * clip_grad, clip_grad)

            if i > 0:
                if lr_warmup_epochs > 0 and i < lr_warmup_epochs:
                    lr_current = lr * min(1, i / lr_warmup_epochs)
                else:
                    lr_current = lr

                if not use_adam:
                    video_pred = torch.add(video_pred, -lr_current * gradients_fullvid[0:1, 0:1])
                else:
                    if i == 1:
                        m = torch.zeros_like(gradients_fullvid)
                    lr_t = lr_current * (1 - adam_beta2 ** (i + 1)) ** 0.5 / (1 - adam_beta1 ** (i + 1))
                    m = adam_beta1 * m + (1 - adam_beta1) * gradients_fullvid
                    m_hat = m / (1 - adam_beta1 ** (i + 1))
                    video_pred = torch.add(video_pred, -lr_t * m_hat)

            if i == 0:
                maxgrad = np.max([
                    np.abs(gradients_fullvid[0, 0].mean(axis=(0)).cpu().detach().numpy().min()),
                    np.abs(gradients_fullvid[0, 0].mean(axis=(0)).cpu().detach().numpy().max())
                ])

            video_pred = torch.clip(video_pred, 0, 255)

            if pix_decay_rate > 0:
                video_pred = ((video_pred - 255 / 2) * (1 - pix_decay_rate)) + (255 / 2)

            video_pred = video_pred.detach().requires_grad_(True)

            if i == 0:
                loss_init = loss.item()

            if i == 0 or i % track_iter == 0 or i == epoch_switch[-1] - 1:
                progress_bar.set_postfix(variable_message=f'loss: {loss.item():.3f} / {loss_init:.0f}', refresh=True)
                progress_bar.update()
                ground_truth = np.moveaxis(video, [2], [0])
                reconstruction = video_pred[0, 0].cpu().detach().numpy()
                reconstruction = reconstruction[:, 14:14 + 36, :]
                mask_cropped = mask_eval[14:14 + 36, :].cpu().detach().numpy()
                reconstruction_masked = reconstruction * mask_cropped + np.ones_like(reconstruction) * (1 - mask_cropped) * 255 / 2

                responses_predicted_full = torch.zeros((constants.num_neurons[mouse_index], video_length, number_models), device=device)

                for n in range(len(predictor)):
                    prediction = predictor[n].predict_trial(
                        video=np.moveaxis(reconstruction_masked, [0], [2]),
                        behavior=behavior,
                        pupil_center=pupil_center,
                        mouse_index=mouse_index
                    )
                    responses_predicted_full[:, :, n] = torch.from_numpy(prediction).to(device)

                responses_predicted_full = responses_predicted_full.mean(axis=2)

                if population_mask is not None:
                    responses_predicted_full = responses_predicted_full * population_mask

                loss_full = utils.response_loss_function(
                    responses_predicted_full[:, eval_frame_skip:],
                    responses[:, eval_frame_skip:].clone().detach(),
                    mask=population_mask
                )
                loss_all.append(loss_full.item())
                print(f"Full loss at iteration {i}: {loss_full.item()}")

                model_list_str = '_'.join(map(str, model_list))
                randneurons_suffix = '_randneurons' if randomize_neurons else ''

                # Guardar el archivo TIFF con el nombre actualizado
                concat_video = utils.save_tif(
                    ground_truth,
                    reconstruction,
                    save_path + f'optimized_input_m{mouse_index}_t{trial}_models_{model_list_str}{randneurons_suffix}.tif',
                    mask=mask_cropped
                )

                video_corr.append(im_sim.reconstruction_video_corr(
                    ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                video_RMSE.append(im_sim.reconstruction_video_RMSE(
                    ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                video_iter.append(i)
                response_corr_value = np.corrcoef(
                    responses.cpu().detach().numpy().flatten(),
                    responses_predicted_full.cpu().detach().numpy().flatten()
                )[0, 1]
                response_corr.append(response_corr_value)
                print(f"Response correlation at iteration {i}: {response_corr_value}")

                motion_energy_gt = im_sim.video_energy(ground_truth[eval_frame_skip:], mask_cropped)
                motion_energy_recon = im_sim.video_energy(reconstruction[eval_frame_skip:], mask_cropped)

                # Cálculo de correlación y RMSE frame a frame
                frame_corr_current = []
                frame_RMSE_current = []
                for t in range(eval_frame_skip, ground_truth.shape[0]):
                    gt_frame = ground_truth[t]
                    recon_frame = reconstruction[t]

                    # Aplicar máscara para seleccionar píxeles dentro de la región de interés
                    valid_pixels = mask_cropped > 0
                    gt_flat = gt_frame[valid_pixels].flatten()
                    recon_flat = recon_frame[valid_pixels].flatten()

                    # Correlación frame a frame
                    if np.std(gt_flat) > 0 and np.std(recon_flat) > 0:
                        corr = np.corrcoef(gt_flat, recon_flat)[0, 1]
                    else:
                        corr = np.nan  # Si la desviación estándar es cero, asignar NaN
                    frame_corr_current.append(corr)

                    # RMSE frame a frame
                    rmse = np.sqrt(np.mean((gt_flat - recon_flat) ** 2))
                    frame_RMSE_current.append(rmse)

                frame_corr.append(frame_corr_current)
                frame_RMSE.append(frame_RMSE_current)

                if i == 0 or i % plot_iter == 0 or i == epoch_switch[-1] - 1:
                    axs[0, 1].clear()
                    axs[0, 1].imshow(np.concatenate((concat_video[0], concat_video[-1]), axis=1), cmap='gray', vmin=0, vmax=255)
                    axs[0, 1].axis('off')
                    axs[0, 1].set_title('First and last frame of ground truth and prediction')

                    axs[3, 1].clear()
                    # Preparar las respuestas predichas para visualización
                    responses_predicted_full_np = responses_predicted_full.cpu().detach().numpy().copy()
                    if population_mask is not None:
                        responses_predicted_full_np[silenced_neurons, :] = -1  # Neuronas silenciadas en negro
                    axs[3, 1].imshow(responses_predicted_full_np, aspect='auto', vmin=-1, vmax=10, cmap=custom_cmap, interpolation='none')
                    axs[3, 1].set_title('Predicted responses reconstructed video (silenced neurons in black)')

                    axs[0, 2].clear()
                    axs[0, 2].plot(motion_energy_gt / np.max(motion_energy_gt), color='blue', label='Ground Truth')
                    if i > 0:
                        axs[0, 2].plot(motion_energy_recon / np.max(motion_energy_recon), color='red', label='Reconstruction')
                    axs[0, 2].set_title('Motion Energy')
                    axs[0, 2].legend()

                    axs[0, 3].clear()
                    axs[0, 3].imshow(gradients_fullvid[0, 0, eval_frame_skip:].mean(axis=(0)).cpu().detach().numpy(), vmin=-maxgrad / 5, vmax=maxgrad / 5)
                    axs[0, 3].set_title(f'Mean gradient in space (vmax {np.round(maxgrad / 2, 4)})')

                    axs[1, 2].clear()
                    axs[1, 2].plot(video_iter, loss_all)
                    axs[1, 2].axhline(y=loss_gt.item(), color='k', linestyle='--')
                    axs[1, 2].set_title('Response loss')

                    axs[1, 3].clear()
                    axs[1, 3].plot(video_iter, response_corr)
                    axs[1, 3].axhline(y=response_corr_gt, color='k', linestyle='--')
                    axs[1, 3].set_title('Response correlation')

                    axs[2, 2].clear()
                    axs[2, 2].plot(video_iter, video_corr)
                    axs[2, 2].set_title('Video correlation')

                    axs[2, 3].clear()
                    axs[2, 3].plot(video_iter, video_RMSE)
                    axs[2, 3].set_title('Video RMSE')

                    # Calcular la media de la correlación y RMSE frame a frame
                    mean_frame_corr = np.nanmean(frame_corr_current)  # np.nanmean ignora valores NaN
                    mean_frame_RMSE = np.nanmean(frame_RMSE_current)  # np.nanmean ignora valores NaN

                    # Graficar correlación frame a frame
                    axs[3, 2].clear()
                    axs[3, 2].plot(frame_corr_current)
                    axs[3, 2].axhline(y=mean_frame_corr, color='red', linestyle='--', label=f'Mean: {mean_frame_corr:.2f}')
                    axs[3, 2].set_title('Frame-by-frame correlation')
                    axs[3, 2].legend()

                    # Graficar RMSE frame a frame
                    axs[3, 3].clear()
                    axs[3, 3].plot(frame_RMSE_current)
                    axs[3, 3].axhline(y=mean_frame_RMSE, color='red', linestyle='--', label=f'Mean: {mean_frame_RMSE:.2f}')
                    axs[3, 3].set_title('Frame-by-frame RMSE')
                    axs[3, 3].legend()

                    fig.savefig(save_path + f'reconstruction_summary_m{mouse_index}_t{trial}{randneurons_suffix}.png')

        # Fin del bucle de reconstrucción
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        print(f"Training time for trial {trial}: {training_time:.2f} seconds")

        # Guardar resultados
        recon_dict = {
            'epochs': epoch_switch[-1],
            'strides': strides_all,
            'strides_switch': epoch_switch,
            'track_iter': track_iter,
            'plot_iter': plot_iter,
            'lr': lr,
            'loss_func': loss_func,
            'response_dropout_rate': response_dropout_rate,
            'mouse_index': mouse_index,
            'trial': trial,
            'load_skip_frames': load_skip_frames,
            'eval_frame_skip': eval_frame_skip,
            'video_length': video_length,
            'subbatch_size': subbatch_size,
            'subbatch_shift': subbatch_shift,
            'model_paths': model_path,
            'model_list': model_list,
            'data_fold': data_fold,
            'save_path': save_path,
            'population_reduction': population_reduction,
            'population_mask': population_mask.cpu().numpy() if population_mask is not None else None,
            'vid_init': vid_init,
            'lr_warmup_epochs': lr_warmup_epochs,
            'use_adam': use_adam,
            'adam_beta1': adam_beta1,
            'adam_beta2': adam_beta2,
            'adam_eps': adam_eps,
            'mask': mask,
            'mask_update_th': mask_update_th,
            'mask_eval_th': mask_eval_th,
            'with_gradnorm': with_gradnorm,
            'clip_grad': clip_grad,
            'device': str(device),
            'video_gt': ground_truth,
            'video_pred': reconstruction,
            'behavior': behavior,
            'pupil_center': pupil_center,
            'responses': responses.cpu().numpy(),
            'responses_predicted_gt': responses_predicted_original.cpu().numpy(),
            'responses_predicted_full': responses_predicted_full.cpu().numpy(),
            'video_iter': video_iter,
            'response_loss_gt': loss_gt.item(),
            'response_loss_full': loss_all,
            'response_corr_gt': response_corr_gt,
            'response_corr_full': response_corr,
            'video_corr': video_corr,
            'video_RMSE': video_RMSE,
            'frame_corr': frame_corr,
            'frame_RMSE': frame_RMSE,
            'motion_energy_gt': motion_energy_gt,
            'motion_energy_recon': motion_energy_recon,
            'training_time': training_time
        }
        np.save(save_path + f'reconstruction_summary_m{mouse_index}_t{trial}{randneurons_suffix}.npy', recon_dict)
        print(f"Saved reconstruction summary for mouse {mouse_index}, trial {trial}")

        print("Generating reconstruction video...")
        # Convertir model_list a una cadena para incluir en el nombre del archivo
        model_list_str = '_'.join(map(str, model_list))

        # Leer el archivo TIFF usando el mismo nombre de archivo
        tif_path = Path(save_path) / f'optimized_input_m{mouse_index}_t{trial}_models_{model_list_str}{randneurons_suffix}.tif'
        mp4_path = Path(save_path) / f'reconstruction_video_m{mouse_index}_t{trial}_models_{model_list_str}{randneurons_suffix}.mp4'

        # Asegurarse de que el archivo TIFF existe
        if not tif_path.exists():
            raise FileNotFoundError(f"The TIFF file {tif_path} does not exist.")

        tif_frames = cv2.imreadmulti(str(tif_path))[1]
        if not tif_frames:
            raise FileNotFoundError(f"No frames found in the TIFF file at {tif_path}. Please ensure the file exists and contains data.")

        height, width = tif_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(str(mp4_path), fourcc, 30, (width, height))

        for frame in tif_frames:
            video_output.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        video_output.release()

        plt.close(fig)

        print(f"\nReconstruction completed for mouse {mouse_index}, trial {trial}")
        print(f"Model used: {model_path[model_list[0]]}")
        print(f"Results saved in: {save_path}")
        print(f"Reconstruction video saved as: {mp4_path}")

print("Reconstruction process completed for all mice and trials.")
