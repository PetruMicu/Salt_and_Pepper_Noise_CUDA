from PIL import Image
import numpy as np

# Load the generated image
image_path = "input_image.jpg"
original_image = Image.open(image_path)

# Add salt-and-pepper noise to a color image
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    np_image = np.array(image)
    total_pixels = np_image.size

    # Salt noise
    num_salt = int(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i, num_salt) for i in np_image.shape]
    np_image[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise
    num_pepper = int(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in np_image.shape]
    np_image[pepper_coords[0], pepper_coords[1]] = 0

    return Image.fromarray(np_image)

# Apply salt-and-pepper noise to the original color image
noisy_image = add_salt_and_pepper_noise(original_image)

# Save the noisy image in JPEG format
output_path = "noisy_image.jpg"
noisy_image.save(output_path, format="JPEG")
output_path
