import cv2
import numpy as np

def apply_lighting_augmentation(image, brightness_factor=1.0, contrast_factor=1.0):
    """
    Apply lighting augmentation to an input image.

    :param image: Input image as a NumPy array (BGR format).
    :param brightness_factor: Factor to adjust brightness (1.0 for no change).
    :param contrast_factor: Factor to adjust contrast (1.0 for no change).
    :return: Augmented image as a NumPy array.
    """
    # Convert the image to float32 format for manipulation
    image = image.astype(np.float32)

    # Adjust brightness and contrast
    augmented_image = cv2.addWeighted(image, brightness_factor, image, 0, 0)
    augmented_image = cv2.multiply(augmented_image, contrast_factor)

    # Clip the values to be in the valid range [0, 255]
    augmented_image = np.clip(augmented_image, 0, 255)

    # Convert the image back to uint8 format
    augmented_image = augmented_image.astype(np.uint8)

    return augmented_image

# Load an example image (you can replace this with your image loading logic)
input_image = cv2.imread('2.png')

# Apply lighting augmentation with specified factors
brightness_factor = 1.2  # Increase brightness
contrast_factor = 1.5    # Increase contrast
augmented_image = apply_lighting_augmentation(input_image, brightness_factor, contrast_factor)

# Display the original and augmented images (you can replace this with saving the image)
cv2.imwrite('Original Image', input_image)
cv2.imwrite('Augmented Image', augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
