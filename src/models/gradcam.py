import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cnn import MedicalResNet
from class_models import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Weight the channels by the corresponding gradients
        assert self.gradients is not None, "Gradients were not captured"
        assert self.activations is not None, "Activations were not captured"
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()

        heatmap = F.relu(heatmap)
        
        # Avoid division by zero if gradients are all zero
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.detach().cpu().numpy()

def prepare_input(device, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    input_batch = img_tensor.unsqueeze(0)
    
    return input_batch.to(device)

def save_gradcam_result(original_img_path, heatmap, output_path):
    img = cv2.imread(original_img_path)
    assert img is not None, f"Failed to load image from {original_img_path}"
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    
    heatmap = (256 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)

def plot_comparison(original_img, heatmap, output_path="result.png"):    
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    if heatmap_uint8.ndim == 3:
        heatmap_uint8 = cv2.cvtColor(heatmap_uint8, cv2.COLOR_BGR2GRAY)
    heatmap_uint8 = np.ascontiguousarray(heatmap_uint8, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create the Superimposed image
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Mammogram")
    
    axes[1].imshow(heatmap_color)
    axes[1].set_title("Grad-CAM Heatmap")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Diagnostic Overlay")
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = get_model(model_name="convnext_small.fb_in22k_ft_in1k")
model.load_state_dict(torch.load('./data/processed/pooled_Mammos/best_classification_model.pth'))
model.to(device)
model.eval()
cam = GradCAM(model,)
batch = prepare_input(device, './data/processed/BrCaWisconsin/images/0af7cc840a48436da112978750a5c151.png')
heatmap = cam.generate_heatmap(batch)
# save_gradcam_result('./data/processed/BrCaWisconsin/images/0af7cc840a48436da112978750a5c151.png', heatmap, 'example.png')
plot_comparison(cv2.imread('./data/processed/BrCaWisconsin/images/0af7cc840a48436da112978750a5c151.png', cv2.IMREAD_GRAYSCALE), heatmap, output_path="./src/models/gradcam_comparison.png")