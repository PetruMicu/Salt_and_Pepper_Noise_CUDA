import cv2
import numpy as np

# Create a white 2K image (2048x1080)
image = np.ones((1080, 2048, 3), dtype=np.uint8) * 255

# Add some black squares (representing "objects" in the image)
cv2.rectangle(image, (400, 300), (600, 500), (0, 0, 0), -1)  # Black square
cv2.rectangle(image, (1200, 700), (1400, 900), (0, 0, 0), -1)  # Black square

# Add random noise (salt and pepper noise) to simulate a noisy image
num_salt = 10000  # Adjusted for 2K image size
num_pepper = 10000  # Adjusted for 2K image size

# Add salt (white pixels)
coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
image[coords[0], coords[1]] = [255, 255, 255]

# Add pepper (black pixels)
coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
image[coords[0], coords[1]] = [0, 0, 0]

# Save the image as a JPG file
cv2.imwrite('test_image.jpg', image)

# Optionally display it
# cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
