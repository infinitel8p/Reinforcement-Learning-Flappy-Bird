import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
    # Get the current device
    current_device = torch.cuda.current_device()
    print(f"Current device: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. PyTorch is using CPU.")
