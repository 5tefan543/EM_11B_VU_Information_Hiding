import os
from PIL import Image
import numpy as np
from dataclasses import dataclass
from sealwatch import chi2, ws

IMAGE_PATH = "../DCIM/"

@dataclass
class ImageData:
    name: str
    array: np.ndarray

def a(images: list[ImageData]):
    stego = None
    print("\na) Analyze the images for LSBR. Which of them seem(s) to carry a steganographic payload?")
    for img in images:
        estimated_embedding_rate = ws.attack(img.array)
        print(f"Analyzing {img.name} with shape {img.array.shape}: α̂  = {estimated_embedding_rate}")
        if estimated_embedding_rate > 0.2:
            stego = img
            print(f"→ {img.name} might contain a stego payload!")
    return stego

def b(stego: ImageData):
    print("\nb) Try to extract the message. It may require a pinch of detective work.")

def main():
    print("The archive “DCIM.zip” contains files that were captured from a communication of Martin and his colleague Verena.")
    images = os.listdir(IMAGE_PATH)
    image_data = [
        ImageData(name=img_name, array=np.array(Image.open(os.path.join(IMAGE_PATH, img_name))))
        for img_name in images
    ]
    stego = a(image_data)
    b(stego)

if __name__ == "__main__":
    main()
