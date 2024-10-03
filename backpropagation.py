import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add the root directory of the project to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
print(f"Added to PYTHONPATH: {project_root}")

# Import original, unmodified modules
from src.predictors import Predictor
from src.data import get_mouse_data
from src import constants

def main():
    # Select mouse, fold, and trial
    mouse = constants.new_mice[0]  # We select the first mouse
    fold = 0

    # Get mouse data
    mouse_data = get_mouse_data(mouse=mouse, splits=constants.folds_splits)
    trial_id = mouse_data['trials'][0]['trial_id']  # Use a fixed trial for consistency

    print(f"Selected mouse: {mouse}, selected trial_id: {trial_id}")

    # Load video, behavior, pupil_center, etc.
    video = np.random.randn(36, 64, 300)  # Simulated video
    behavior = np.random.randn(2, 300)  # Simulated behavior
    pupil_center = np.random.randn(2, 300)  # Simulated pupil center
    responses = np.random.randn(7863, 300)  # Simulated neuronal responses

    length = 300  # Maximum length

    # Load the pretrained model from the .pth file
    model_path = project_root / 'data' / 'experiments' / 'true_batch_001' / f'fold_{fold}' / 'model-000-0.290763.pth'
    print(f"Using model from: {model_path}")

    # Initialize the Predictor with the original model path
    predictor = Predictor(model_path=model_path)

    # Set model to training mode
    predictor.model.train()

    # Create input tensor (video) that requires gradients
    input_video_frames = torch.randn((1, 1, length, video.shape[0], video.shape[1]), device='cuda', requires_grad=True)

    # Prepare behavior and pupil data
    behavior_tensor = torch.tensor(behavior, device='cuda', dtype=torch.float32)
    pupil_center_tensor = torch.tensor(pupil_center, device='cuda', dtype=torch.float32)
    behavior_data = torch.cat([behavior_tensor, pupil_center_tensor], dim=0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
    behavior_data = behavior_data.repeat(1, 1, 1, video.shape[0], video.shape[1])

    # Concatenate input_video_frames and behavior_data
    input_video = torch.cat([input_video_frames, behavior_data], dim=1)

    # Prepare target neuronal responses
    target_responses = torch.tensor(responses, device='cuda', dtype=torch.float32).unsqueeze(0)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Make predictions using the model's predict method
    predicted_responses = predictor.model.predict(input_video, mouse_index=constants.mouse2index[mouse])

    # Check if the prediction has grad_fn to confirm that it allows backpropagation
    if isinstance(predicted_responses, torch.Tensor) and predicted_responses.requires_grad:
        print("The predictions have gradients enabled.")
    else:
        print("ERROR: The predictions do not have gradients enabled.")
        return

    # Calculate the loss
    loss = criterion(predicted_responses, target_responses)

    # Perform backpropagation
    loss.backward()

    # Verify if input_video_frames has accumulated gradients
    if input_video_frames.grad is not None:
        print("Gradients calculated successfully in input_video_frames.")
        print(f"Gradient min: {input_video_frames.grad.min().item()}, max: {input_video_frames.grad.max().item()}")
    else:
        print("ERROR: No gradients were calculated in input_video_frames. This confirms that the model does not allow gradient backpropagation.")

if __name__ == "__main__":
    main()
