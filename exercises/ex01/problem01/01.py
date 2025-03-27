from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jpeglib

IMAGE_PATH = "../BOSSBase/10.png"

UINT8_MIN = 0
UINT8_MAX = 255

def psnr(image_array1, image_array2):
    image_array1 = image_array1.astype(np.float32)
    image_array2 = image_array2.astype(np.float32)

    mse = np.mean((image_array1 - image_array2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(float(UINT8_MAX)**2 / mse)

def show_images_side_by_side(images, titles, filename):
    fig, axs = plt.subplots(1, len(images), figsize=(12, 5))
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img, cmap='gray', vmin=UINT8_MIN, vmax=UINT8_MAX)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def a(image_array: np.ndarray):
    print("a) Increment all the pixels by 1. What PSNR do you expect? Measure the PSNR experimentally.")
    image_array = image_array.astype(np.uint16) # Prevent overflows in the next step    
    modified_image_array = np.clip(image_array + 1, UINT8_MIN, UINT8_MAX).astype(np.uint8)

    show_images_side_by_side(
        images=[image_array, modified_image_array],
        titles=["Original", "All Pixels +1"],
        filename="a)_10_pixel_increment_comparison.png"
    )

    # What PSNR do you expect?
    # => Minor change -> high PSNR value

    # Measure the PSNR experimentally
    psnr_value = psnr(image_array, modified_image_array)
    print(f"PSNR: {round(psnr_value, 3)} dB")

def apply_gamma_correction(image_array: np.ndarray, gamma: float):
    image_array = image_array.astype(np.float32)
    image_array = UINT8_MAX * (image_array / UINT8_MAX) ** gamma
    image_array = np.clip(image_array, UINT8_MIN, UINT8_MAX).astype(np.uint8)
    return image_array

def plot_histograms_side_by_side(original: np.ndarray, modified: np.ndarray, filename: str, labels=("Original", "Modified")):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].hist(original.ravel(), bins=256, range=(UINT8_MIN, UINT8_MAX), color='gray')
    axs[0].set_title(f"{labels[0]} Histogram")
    axs[0].set_xlabel("Pixel value")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(modified.ravel(), bins=256, range=(UINT8_MIN, UINT8_MAX), color='gray')
    axs[1].set_title(f"{labels[1]} Histogram")
    axs[1].set_xlabel("Pixel value")
    axs[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_quantization_error(original: np.ndarray, modified: np.ndarray, filename: str):
    error = original.astype(np.int16) - modified.astype(np.int16)
    x_min = error.min() - 2
    x_max = error.max() + 2
    num_bins = x_max - x_min

    plt.figure(figsize=(6, 4))
    plt.hist(error.ravel(), bins=num_bins, range=(x_min, x_max), color='blue', edgecolor='black')
    plt.title("Quantization Error Distribution")
    plt.xlabel("Error (Original - Modified)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def b(image_array: np.ndarray):
    print("\nb) Apply gamma correction, γ = 1.1, and store the resulting image in PNG. Compare the image histograms. Plot the distribution of the quantization error.")
    gamma = 1.1
    gamma_corrected_image_array = apply_gamma_correction(image_array, gamma)

    show_images_side_by_side(
        images=[image_array, gamma_corrected_image_array],
        titles=["Original", f"Gamma {gamma}"],
        filename="b)_10_gamma_comparison.png"
    )

    plot_histograms_side_by_side(
        original=image_array,
        modified=gamma_corrected_image_array,
        filename="b)_10_gamma_comparison_histogram.png",
        labels=("Original", "Gamma 1.1")
    )

    plot_quantization_error(
        original=image_array,
        modified=gamma_corrected_image_array,
        filename="b)_10_gamma_quantization_error.png"
    )

def subsample_nearest_neighbor(image_array: np.ndarray, factor: int = 2):
    return image_array[::factor, ::factor]

def apply_linear_filter(image_array: np.ndarray, kernel: np.ndarray):
    image_array = image_array.astype(np.float32)
    output = np.zeros_like(image_array)

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the image (reflect mode reduces edge artifacts)
    padded = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Perform convolution
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    # Clip and return as uint8
    return np.clip(output, UINT8_MIN, UINT8_MAX).astype(np.uint8)

    
def c(image_array: np.ndarray):
    print("\nc) Subsample the image by nearest neighbor. Mark the areas with aliasing. Employ a linear filter prior to the subsampling to suppress the aliasing.")

    # Subsample without filtering (aliasing visible)
    subsampled_raw = subsample_nearest_neighbor(image_array)

    # Apply linear filter before subsampling
    motion_blur = (1/6) * np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ])
    filtered = apply_linear_filter(image_array, motion_blur)
    subsampled_filtered = subsample_nearest_neighbor(filtered)

    show_images_side_by_side(
        images=[image_array, subsampled_raw, subsampled_filtered],
        titles=["Original", "Subsampled (with Aliasing)", "Filtered + Subsampled"],
        filename="c)_10_subsampling_linear_filter_comparison.png"
    )

def build_dct_matrix(n):
    D = np.zeros((n, n))
    for j in range(n):
        cj = 1 / np.sqrt(2) if j == 0 else 1
        for i in range(n):
            D[i, j] = cj * np.sqrt(2 / n) * np.cos(((2 * i + 1) * j * np.pi) / (2 * n))
    return D

def compute_2d_dct(image_array: np.ndarray):
    image_array.astype(np.float32)
    h, w = image_array.shape
    D_h = build_dct_matrix(h)
    D_w = build_dct_matrix(w)
    return D_h @ image_array @ D_w.T

def manual_idct2(dct_array):
    h, w = dct_array.shape
    D_h = build_dct_matrix(h)
    D_w = build_dct_matrix(w)
    return D_h.T @ dct_array @ D_w

def d(image_array: np.ndarray):
    print("\nd) Compute the 2D DCT spectrum of the image (over the whole image, not in 8 × 8 blocks like JPEG) and apply top-left cropping. What effect does it have when transferring it back to the spatial domain?")
    
    # Compute 2D DCT
    dct_coeffs = compute_2d_dct(image_array)

    # Apply top-left cropping
    keep_fraction = 0.2
    h, w = dct_coeffs.shape
    h_keep = int(h * keep_fraction)
    w_keep = int(w * keep_fraction)

    # Zero out high frequencies
    dct_cropped = np.zeros_like(dct_coeffs)
    dct_cropped[:h_keep, :w_keep] = dct_coeffs[:h_keep, :w_keep]

    # Reconstruct
    image_reconstructed = manual_idct2(dct_cropped)
    image_reconstructed =  np.clip(image_reconstructed, UINT8_MIN, UINT8_MAX).astype(np.uint8)

    print(f"Effect of top-left cropping: Image is blurred due to the removal of high frequencies")

    show_images_side_by_side(
        images=[image_array, image_reconstructed],
        titles=["Original", "Reconstructed (Lowpass DCT)"],
        filename="d)_10_dct_cropped_comparison.png"
    )

def e(image_array: np.ndarray):
    print("\ne) Read the DCT coefficients of a JPEG-compressed version of this image, for example by using the Python package jpeglib or by calling the standard libjpeg library directly. Compare the histograms of the DC (0, 0) and AC (11, 44) modes")
    #save the image as a jpeg
    image = Image.fromarray(image_array)
    image.save("10.jpg")

    dct_coeffs = jpeglib.read_dct("10.jpg")

    # DC: extract coefficient at (0, 0) in each block
    dc_values = dct_coeffs.Y[:, :, 0, 0].flatten()

    # AC: extract coefficient at (1, 1) and (4, 4) in each block
    ac_values_1_1 = dct_coeffs.Y[:, :, 1, 1].flatten()
    ac_values_4_4 = dct_coeffs.Y[:, :, 4, 4].flatten()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(dc_values, bins=len(np.unique(dc_values)), color='blue')
    plt.title('DC Values (0,0)')

    # plt.subplot(1, 2, 2)
    # plt.hist(ac_values, bins=100, color='red')
    # plt.title('AC Values (11,44)')

    plt.savefig("e)_10_dc_ac_histograms.png")
    plt.close()

def main():
    print("For the following task, use the image 10.png from the BOSSBase database.")
    image = Image.open(IMAGE_PATH).convert("L")
    image_array = np.array(image, dtype=np.uint8)
    print(f"Image mode: {image.mode}\n")
    a(image_array)
    b(image_array)
    c(image_array)
    d(image_array)
    e(image_array)

if __name__ == "__main__":
    main()