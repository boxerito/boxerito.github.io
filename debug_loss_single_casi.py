import sys
import os
from pathlib import Path
import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
print(f"Added to PYTHONPATH: {project_root}")

from src.data import get_mouse_data
from src.utils import get_best_model_path, get_length_without_nan
from src.predictors_temp import Predictor
from src import constants
from src.indexes import IndexesGenerator

script_dir = Path(__file__).parent
catalog_path = script_dir / 'trials_by_fold.json'

def poisson_loss(predicted, target):
    epsilon = 1e-8
    # Usar torch.clamp para evitar predicciones negativas o cercanas a 0
    predicted = torch.clamp(predicted, min=epsilon)
    target = target.to(torch.float32)
    loss = predicted - target * torch.log(predicted)
    return loss.mean()

def recortar_frames(data, skip_first=50, skip_last=1, target_length=300):
    print(f"Recortar_frames: Input shape: {data.shape}")
    if data.shape[0] > target_length + skip_first + skip_last:
        data = data[skip_first:-skip_last]
    if data.shape[0] < target_length:
        pad_width = ((0, target_length - data.shape[0]), (0, 0), (0, 0))
        data = np.pad(data, pad_width, mode='constant')
    elif data.shape[0] > target_length:
        data = data[:target_length]
    print(f"Recortar_frames: Output shape: {data.shape}")
    return data

def load_trial_catalog(catalog_path):
    print(f"Loading trial catalog from: {catalog_path}")
    with open(catalog_path, 'r') as f:
        return json.load(f)

def get_random_trial_for_fold(catalog, mouse, fold):
    return random.choice(catalog[mouse][str(fold)])

def load_sample_data(mouse_data, trial_id):
    print(f"Loading data for trial_id {trial_id}")
    trial_data = next((trial for trial in mouse_data['trials'] if trial['trial_id'] == trial_id), None)
    if trial_data is None:
        raise ValueError(f"No se encontró el trial_id {trial_id}")
    
    responses = np.load(trial_data["response_path"]).astype(np.float32)
    behavior = np.load(trial_data["behavior_path"]).astype(np.float32)
    pupil_center = np.load(trial_data["pupil_center_path"]).astype(np.float32)
    
    length = get_length_without_nan(responses[0])
    
    responses = responses[:, :length]
    behavior = behavior[:, :length]
    pupil_center = pupil_center[:, :length]
    
    print(f"Shapes after trimming - responses: {responses.shape}, behavior: {behavior.shape}, pupil_center: {pupil_center.shape}")
    print(f"Responses stats - min: {np.nanmin(responses):.4f}, max: {np.nanmax(responses):.4f}, mean: {np.nanmean(responses):.4f}, std: {np.nanstd(responses):.4f}")
    print(f"Behavior stats - min: {np.nanmin(behavior):.4f}, max: {np.nanmax(behavior):.4f}, mean: {np.nanmean(behavior):.4f}, std: {np.nanstd(behavior):.4f}")
    print(f"Pupil center stats - min: {np.nanmin(pupil_center):.4f}, max: {np.nanmax(pupil_center):.4f}, mean: {np.nanmean(pupil_center):.4f}, std: {np.nanstd(pupil_center):.4f}")
    
    return responses, behavior, pupil_center

def load_original_video(mouse_data, trial_id):
    print(f"Loading original video for trial_id {trial_id}")
    trial_data = next((trial for trial in mouse_data['trials'] if trial['trial_id'] == trial_id), None)
    if trial_data is None:
        raise ValueError(f"No se encontró el trial_id {trial_id}")
    
    video = np.load(trial_data["video_path"])
    print(f"Loaded original video shape: {video.shape}")
    
    # Redimensionar de 36x64 a 64x64
    resized_video = np.zeros((video.shape[2], 64, 64), dtype=video.dtype)
    for i in range(video.shape[2]):
        resized_video[i] = cv2.resize(video[:, :, i], (64, 64), interpolation=cv2.INTER_LINEAR)
    
    print(f"Resized original video shape: {resized_video.shape}")
    return resized_video

def initialize_model(experiment_dir, fold):
    print(f"Initializing model for fold {fold}")
    model_path = get_best_model_path(experiment_dir / f'fold_{fold}')
    print(f"Loading model from: {model_path}")
    predictor = Predictor(model_path=model_path, device='cuda', blend_weights="ones")
    return predictor.model

def reconstruct_video_debug(model, responses, behavior, pupil_center, original_video, mouse_index, num_epochs=1000, learning_rate=10, output_dir=None):
    print("Starting video reconstruction...")
    print(f"Input shapes - responses: {responses.shape}, behavior: {behavior.shape}, pupil_center: {pupil_center.shape}")
    print(f"Original video shape: {original_video.shape}")
    
    video_length = original_video.shape[0]
    video = torch.full((1, 1, video_length, 64, 64), 127.5, device='cuda', dtype=torch.float32)
    video.requires_grad = True

    optimizer = torch.optim.Adam([video], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = poisson_loss
    
    window_size = 32
    stride = 8

    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Video length: {video_length}")
    print(f"Number of windows: {(video_length - window_size) // stride + 1}")

    losses = []
    correlations = []
    pred_stats = []

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    print(f"responses shape: {responses.shape}")
    print(f"behavior shape: {behavior.shape}")
    print(f"pupil_center shape: {pupil_center.shape}")
    print(f"video shape: {video.shape}")

    for epoch in range(num_epochs):
        if epoch < 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (epoch + 1) / 10

        epoch_loss = 0
        num_batches = 0
        accumulated_grad = torch.zeros_like(video)

        for i in range(0, video_length - window_size + 1, stride):
            optimizer.zero_grad()
            
            end = min(i + window_size, video_length)
            current_window_size = end - i

            input_video = video[:, :, i:end]
            input_responses = torch.tensor(responses[:, i:end], device='cuda', dtype=torch.float32)
            input_pupil_center = torch.tensor(pupil_center[:, i:end], device='cuda', dtype=torch.float32)
            input_behavior = torch.tensor(behavior[:, i:end], device='cuda', dtype=torch.float32)

            print(f"Before processing - Window {num_batches + 1} - i: {i}, end: {end}, current_window_size: {current_window_size}")
            print(f"input_video shape: {input_video.shape}")
            print(f"input_responses shape: {input_responses.shape}")
            print(f"input_pupil_center shape: {input_pupil_center.shape}")
            print(f"input_behavior shape: {input_behavior.shape}")

            try:
                input_tensor = torch.cat([
                    input_video,
                    input_pupil_center[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_window_size, 64, 64),
                    input_pupil_center[1].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_window_size, 64, 64),
                    input_behavior[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_window_size, 64, 64),
                    input_behavior[1].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_window_size, 64, 64)
                ], dim=1)
            except RuntimeError as e:
                print(f"Error en la expansión: {e}")
                print("Aplicando padding para igualar dimensiones...")
                
                target_size = window_size
                
                input_video = torch.nn.functional.pad(input_video, (0, 0, 0, 0, 0, target_size - input_video.shape[2]))
                input_responses = torch.nn.functional.pad(input_responses, (0, target_size - input_responses.shape[1]))
                input_pupil_center = torch.nn.functional.pad(input_pupil_center, (0, target_size - input_pupil_center.shape[1]))
                input_behavior = torch.nn.functional.pad(input_behavior, (0, target_size - input_behavior.shape[1]))
                
                input_tensor = torch.cat([
                    input_video,
                    input_pupil_center[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_size, 64, 64),
                    input_pupil_center[1].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_size, 64, 64),
                    input_behavior[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_size, 64, 64),
                    input_behavior[1].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_size, 64, 64)
                ], dim=1)

            print(f"Final input_tensor shape: {input_tensor.shape}")
            
            assert input_video.shape[2] == input_responses.shape[1], f"Mismatch in temporal dimension: video {input_video.shape[2]}, responses {input_responses.shape[1]}"
            assert input_tensor.shape[2] == input_video.shape[2], f"Unexpected input_tensor shape: {input_tensor.shape}"
            
            predicted_responses = model.nn_module(input_tensor, mouse_index)
            
            print(f"predicted_responses shape: {predicted_responses.shape}")
            print(f"input_responses shape: {input_responses.shape}")
            
            loss = loss_fn(predicted_responses, input_responses)
            
            # Calcular gradientes directamente para video
            video_grad = torch.autograd.grad(loss, video, retain_graph=True)[0]
            
            accumulated_grad[:, :, i:end] += video_grad[:, :, :current_window_size]
            
            epoch_loss += loss.item()
            num_batches += 1

        # Actualizar video usando los gradientes acumulados
        with torch.no_grad():
            video -= learning_rate * accumulated_grad
            video.clamp_(0, 255)
        
        losses.append(epoch_loss / num_batches)
        
        current_video = video.detach().cpu().numpy().squeeze()
        correlation = pearsonr(current_video.flatten(), original_video.flatten())[0]
        correlations.append(correlation)

        pred_stats.append((predicted_responses.min().item(), predicted_responses.max().item(), predicted_responses.mean().item()))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.4f}, Correlation: {correlation:.4f}")
            
            axs[0, 0].clear()
            axs[0, 0].plot(losses)
            axs[0, 0].set_title('Loss over iterations')
            axs[0, 0].set_xlabel('Iterations')
            axs[0, 0].set_ylabel('Loss')

            axs[0, 1].clear()
            axs[0, 1].plot([s[0] for s in pred_stats], label='Min')
            axs[0, 1].plot([s[1] for s in pred_stats], label='Max')
            axs[0, 1].plot([s[2] for s in pred_stats], label='Mean')
            axs[0, 1].set_title('Predicted Response Statistics')
            axs[0, 1].set_xlabel('Iterations')
            axs[0, 1].set_ylabel('Value')
            axs[0, 1].legend()

            axs[1, 0].clear()
            axs[1, 0].plot(correlations)
            axs[1, 0].set_title('Correlation with Original Video')
            axs[1, 0].set_xlabel('Iterations')
            axs[1, 0].set_ylabel('Correlation')

            axs[1, 1].clear()
            axs[1, 1].hist(accumulated_grad.cpu().numpy().flatten(), bins=50)
            axs[1, 1].set_title('Gradient Distribution')
            axs[1, 1].set_xlabel('Gradient Value')
            axs[1, 1].set_ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(output_dir / f'reconstruction_progress_{epoch+1}.png')

    print("Video reconstruction completed.")
    print(f"Final reconstructed video shape: {video.shape}")
    
    reconstructed_video = video.detach().cpu().numpy().squeeze()
    print(f"Reconstructed video shape before adjustment: {reconstructed_video.shape}")
    
    # Asegurar que el video reconstruido tenga 300 frames
    if reconstructed_video.shape[0] != 300:
        reconstructed_video = reconstructed_video[:300]
    
    print(f"Reconstructed video shape after adjustment: {reconstructed_video.shape}")
    print(f"Original video shape: {original_video.shape}")
    
    # Calcular correlación
    correlation = pearsonr(reconstructed_video.flatten(), original_video.flatten())[0]
    correlations.append(correlation)

    return reconstructed_video, losses, correlations

def plot_video_comparison(original_video, reconstructed_video, num_frames=5, output_dir=None):
    print("Generating video comparison plot...")
    fig, axes = plt.subplots(2, num_frames, figsize=(20, 8))
    for i in range(num_frames):
        frame_idx = i * (original_video.shape[0] // num_frames)
        
        axes[0, i].imshow(original_video[frame_idx], cmap='gray', vmin=0, vmax=255)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original Frame {frame_idx}')
        
        axes[1, i].imshow(reconstructed_video[frame_idx], cmap='gray', vmin=0, vmax=255)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstructed Frame {frame_idx}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'video_comparison.png')
    plt.close()
    print("Video comparison plot saved as 'video_comparison.png'")

def save_video_frames(video, directory):
    print(f"Saving video frames to directory: {directory}")
    os.makedirs(directory, exist_ok=True)
    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx].astype(np.uint8)
        plt.imsave(os.path.join(directory, f"frame_{frame_idx:04d}.png"), frame, cmap='gray', vmin=0, vmax=255)
    print(f"Saved {video.shape[0]} frames.")

def create_animation(frame_directory, movie_name, fps=30):
    print(f"Creating animation from frames in {frame_directory}")
    from moviepy.editor import ImageSequenceClip
    
    if not os.path.exists(frame_directory):
        raise FileNotFoundError(f"El directorio {frame_directory} no existe")
    
    frames = [(int(f.split('_')[1].split('.')[0]), f) for f in os.listdir(frame_directory) if f.endswith('.png')]
    frames.sort(key=lambda x: x[0])
    frame_paths = [os.path.join(frame_directory, f[1]) for f in frames]
    clip = ImageSequenceClip(frame_paths, fps=fps)
    file_name = f"{movie_name}.mp4"
    clip.write_videofile(file_name, codec='libx264')
    print(f"Animation saved as {file_name}")

def process_fold(fold, experiment_dir, mouse_data, trial_id, mouse_index, output_dir):
    print(f"Processing fold {fold}, trial {trial_id}")
    model = initialize_model(experiment_dir, fold)
    responses, behavior, pupil_center = load_sample_data(mouse_data, trial_id)
    original_video = load_original_video(mouse_data, trial_id)
    
    print(f"Original video shape before processing: {original_video.shape}")
    
    # Aplicar recortar_frames al video original
    original_video_300 = recortar_frames(original_video)
    
    print(f"Original video shape after processing: {original_video_300.shape}")
    
    # Asegurar que tenga exactamente 300 frames
    if original_video_300.shape[0] != 300:
        original_video_300 = original_video_300[:300]
    
    print(f"Final original video shape: {original_video_300.shape}")
    
    reconstructed_video, losses, correlations = reconstruct_video_debug(model, responses, behavior, pupil_center, original_video_300, mouse_index, output_dir=output_dir)
    
    print(f"Reconstructed video shape: {reconstructed_video.shape}")
    
    return reconstructed_video, original_video_300, losses, correlations

if __name__ == "__main__":
    try:
        print("Starting debug_loss_single.py")
        
        mouse = constants.new_mice[0]
        print(f"Selected mouse: {mouse}")
        
        fold = 0
        print(f"Selected fold: {fold}")

        mouse_index = constants.mouse2index[mouse]
        print(f"Mouse index: {mouse_index}")

        trial_catalog = load_trial_catalog(catalog_path)

        trial_id = get_random_trial_for_fold(trial_catalog, mouse, fold)
        print(f"Selected trial_id: {trial_id}")

        experiment_dir = Path('/home/antoniofernandez/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001')

        output_dir = project_root / 'scripts' / 'outputs' / 'clopath' / str(trial_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Getting mouse data...")
        mouse_data = get_mouse_data(mouse=mouse, splits=constants.folds_splits)

        print(f"Processing fold {fold}, trial {trial_id}")
        reconstructed_video, original_video_300, losses, correlations = process_fold(fold, experiment_dir, mouse_data, trial_id, mouse_index, output_dir)
        
        if reconstructed_video is not None:
            correlation = pearsonr(reconstructed_video.flatten(), original_video_300.flatten())[0]
            print(f"Fold {fold}, Trial {trial_id} final correlation: {correlation:.4f}")

            plot_video_comparison(original_video_300, reconstructed_video, output_dir=output_dir)

            save_video_frames(reconstructed_video, output_dir / 'reconstructed_frames')
            create_animation(output_dir / 'reconstructed_frames', output_dir / 'reconstructed_video')

            save_video_frames(original_video_300, output_dir / 'original_frames')
            create_animation(output_dir / 'original_frames', output_dir / 'original_video')
            
            np.save(output_dir / 'losses.npy', np.array(losses))
            np.save(output_dir / 'correlations.npy', np.array(correlations))
            
            print("Debug process completed successfully.")
        else:
            print(f"Reconstruction failed for fold {fold}, trial {trial_id}")
    
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()
