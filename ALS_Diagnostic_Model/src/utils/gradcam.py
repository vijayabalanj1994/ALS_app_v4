from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw


def extract_output_cam_and_image(model, image_tensor, class_idx, target_layer="features.denseblock4"):

    cam_extractor = GradCAM(model.model, target_layer=target_layer)
    output = model(image_tensor)
    cam = cam_extractor(class_idx, output)[0]

    # De-normalize image
    denorm_img = image_tensor.squeeze().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_img = denorm_img * std + mean
    original_image = to_pil_image(denorm_img.clamp(0, 1))

    return output, cam, original_image

def generate_gradcam_overlay(cam, original_image):

    overlay = overlay_mask(original_image, to_pil_image(cam, mode="F"))
    return overlay

def generate_gradcam_circles(cam, original_image, threshold=0.8):

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()

    # Resize and normalize CAM
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized) / 255.0
    binary_mask = (cam_array >= threshold).astype(np.uint8)

    # Detect contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles
    circles = []
    draw = ImageDraw.Draw(original_image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        r = max(w, h) // 2 + 4
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="red", width=3)
        circles.append((cx,cy,r))

    return original_image, circles


def generate_gradcam_focus_mask(cam, original_image, threshold=0.8, alpha=0.6):

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().squeeze().numpy()

    # Resize and normalize CAM
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, Image.BILINEAR)
    cam_array = np.array(cam_resized).astype(np.float32) / 255.0

    # Create alpha mask: 0 for high activation, alpha for low
    alpha_mask = np.where(cam_array < threshold, int(255 * alpha), 0).astype(np.uint8)

    # Convert to RGBA
    image_rgba = original_image.convert("RGBA")

    # Create black image with variable alpha (based on CAM)
    black_overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    black_overlay.putalpha(Image.fromarray(alpha_mask))

    # Composite: overlay the transparent black mask on top of original image
    result = Image.alpha_composite(image_rgba, black_overlay)

    return result

def extract_color_masks(qpath_image):
    """Extract blue, yellow, and orange masks from a QuPath-annotated image."""

    # Convert to RGB numpy array
    image_np = np.array(qpath_image.convert("RGB"))

    # Define BGR-like color bounds (tune these as needed)
    masks = {
        "blue": cv2.inRange(image_np, (90, 90, 200), (150, 150, 255)),
        "yellow": cv2.inRange(image_np, (180, 180, 0), (255, 255, 150)),
        #"orange": cv2.inRange(image_np, (200, 100, 0), (255, 160, 80))
    }

    # Convert to PIL images for visualization
    return {k: Image.fromarray((v > 0).astype(np.uint8) * 255) for k, v in masks.items()}

def quantify_focus(circles, cells_mask, border_exclude=10):
    # Convert PIL image to binary NumPy mask
    cell_mask_np = np.array(cells_mask.convert("L")) > 0  # shape: (H, W)

    # Exclude borders
    h, w = cell_mask_np.shape
    cell_mask_np[:border_exclude, :] = 0              # Top
    cell_mask_np[-border_exclude:, :] = 0             # Bottom
    cell_mask_np[:, :border_exclude] = 0              # Left
    cell_mask_np[:, -border_exclude:] = 0             # Right

    hit_count, miss_count = 0, 0
    for cx, cy, r in circles:
        circle_mask = np.zeros_like(cell_mask_np, dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), r, 1, -1)

        if np.any(circle_mask & cell_mask_np):
            hit_count += 1
        else:
            miss_count += 1

    precision = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0.0
    return hit_count, miss_count, precision


