import torch
from collections import OrderedDict

# Define the path to your original and new checkpoint
original_ckpt_path = 'models\\best_coco_segm_mAP_iter_5150.pth'
new_ckpt_path = 'models\\converted_best_coco_segm_mAP_iter_5150.pth'

print(f"Loading original checkpoint from: {original_ckpt_path}")

# Load the full checkpoint, explicitly allowing it to unpickle Python objects
# This is the key step that PyTorch 2.6 was warning you about.
original_ckpt = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)

# A checkpoint can store the state dict under 'state_dict' or at the top level.
# We check for the 'state_dict' key first.
if 'state_dict' in original_ckpt:
    state_dict = original_ckpt['state_dict']
else:
    # If not, assume the whole file is the state_dict
    state_dict = original_ckpt

# It's good practice to create a new ordered dictionary
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k] = v

# Save the new checkpoint containing ONLY the weights
torch.save(new_state_dict, new_ckpt_path)

print(f"Successfully created a new weights-only checkpoint at: {new_ckpt_path}")