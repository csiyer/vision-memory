import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image_hue(img, angle_deg):
    """
    Rotates the hue of an image in CIELAB color space.
    """
    # 1. Convert to float32 and scale to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # 2. Convert RGB to LAB
    # Note: OpenCV uses Lab ranges: L [0,100], a [-127,127], b [-127,127]
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)
    
    # Extract a and b channels
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Flatten for matrix multiplication
    orig_shape = a.shape
    v = np.stack((a.flatten(), b.flatten()), axis=0)
    
    # 3. Define Rotation Matrix
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    # 4. Apply Rotation
    v_rotated = rotation_matrix @ v
    
    # 5. Reshape and Reconstruct
    lab[:, :, 1] = v_rotated[0, :].reshape(orig_shape)
    lab[:, :, 2] = v_rotated[1, :].reshape(orig_shape)
    
    # 6. Convert back to RGB
    rgb_rotated = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    
    # Clip and convert back to uint8
    rgb_rotated = np.clip(rgb_rotated * 255, 0, 255).astype(np.uint8)
    return rgb_rotated

# --- Visualization Loop (Equivalent to HowToRotate) ---
def show_rotations(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR
    
    plt.figure(figsize=(12, 12))
    angles = np.arange(10, 370, 10) # 10 to 360 in steps of 10
    
    for i, angle in enumerate(angles):
        plt.subplot(6, 6, i + 1)
        rotated = rotate_image_hue(img, angle)
        plt.imshow(rotated)
        plt.title(f"{angle}°", fontsize=8)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# To run:
# show_rotations('obj1.jpg')