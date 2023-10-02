import cv2
import numpy as np

def apply_lighting_augmentation(img, factor=0.1):
    """
    Apply lighting augmentation to an input image.

    :param image: Input image as a NumPy array (BGR format).
    :param brightness_factor: Factor to adjust brightness (1.0 for no change).
    :param contrast_factor: Factor to adjust contrast (1.0 for no change).
    :return: Augmented image as a NumPy array.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

# Load an example image (you can replace this with your image loading logic)
input_image = cv2.imread('2.png')

# Apply lighting augmentation with specified factors
brightness_factor = 1.2  # Increase brightness
contrast_factor = 1.5    # Increase contrast
augmented_image = apply_lighting_augmentation(input_image, brightness_factor)

# Display the original and augmented images (you can replace this with saving the image)
cv2.imwrite('Original Image.png', input_image)
cv2.imwrite('Augmented Image.png', augmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
