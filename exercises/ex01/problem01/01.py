from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# IMAGE_PATH = "../BOSSBase/10.png"
IMAGE_PATH = "/home/sw/Software_Projekte/UIBK/Master_Computer_Science/Term02/EM_11B_VU_Information_Hiding/exercises/ex01/BOSSBase/10.png"

UINT8_MIN = 0
UINT8_MAX = 255

def psnr(image_array1, image_array2):
    image_array1 = image_array1.astype(np.float32)
    image_array2 = image_array2.astype(np.float32)

    mse = np.mean((image_array1 - image_array2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(float(UINT8_MAX)**2 / mse)

def a(image_array: np.ndarray):
    print("a) Increment all the pixels by 1. What PSNR do you expect? Measure the PSNR experimentally.")
    image_array = image_array.astype(np.int16) # Prevent overflows in the next step    
    modified_image_array = np.clip(image_array + 1, UINT8_MIN, UINT8_MAX).astype(np.uint8)
    modified_image = Image.fromarray(modified_image_array)
    modified_image.save("10_(all_pixels_incremented_by_1).png")

    # What PSNR do you expect?
    # => Minor change -> high PSNR value

    # Measure the PSNR experimentally
    psnr_value = psnr(image_array, modified_image_array)
    print(f"PSNR: {round(psnr_value, 3)} dB")

def plot_histogram(image_array: np.ndarray, title: str):
    plt.hist(image_array.ravel(), bins=256, range=(0, 256))
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.savefig(title + ".png")
    plt.close()

def apply_gamma_correction(image_array: np.ndarray, gamma: float):
    image_array = image_array.astype(np.float32)
    image_array = UINT8_MAX * (image_array / UINT8_MAX) ** gamma
    image_array = np.clip(image_array, UINT8_MIN, UINT8_MAX).astype(np.uint8)
    return image_array

def b(image_array: np.ndarray):
    print("\nb) Apply gamma correction, γ = 1.1, and store the resulting image in PNG. Compare the image histograms. Plot the distribution of the quantization error.")
    gamma = 1.1
    gamma_corrected_image_array = apply_gamma_correction(image_array, gamma)
    gamma_corrected_image = Image.fromarray(gamma_corrected_image_array)
    gamma_corrected_image.save("10_(gamma_correction_1.1).png")

    plot_histogram(image_array, "10_histogram")
    plot_histogram(gamma_corrected_image_array, "10_(gamma_correction_1.1)_histogram")

def main():
    print("For the following task, use the image 10.png from the BOSSBase database.")
    image = Image.open(IMAGE_PATH).convert("L")
    image_array = np.array(image, dtype=np.uint8)
    print(f"Image mode: {image.mode}\n")
    a(image_array)
    b(image_array)

if __name__ == "__main__":
    main()