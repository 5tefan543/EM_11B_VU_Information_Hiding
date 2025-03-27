from conseal import lsb
from sealwatch import ws
from PIL import Image
import numpy as np

IMAGE_PATH = "../BOSSBase/10.png"

def a():
    print("a) Embed the message 01010011 using sequential LSBR into the following cover:")
    # Original cover values
    cover = [105, 105, 116, 98, 105, 104, 104, 107, 101, 114]

    # Message to embed (as a binary string)
    message_bits = "01010011"
    message = [int(b) for b in message_bits]

    # Perform LSB replacement
    stego = cover.copy()
    for i in range(len(message)):
        # First clears the least significant bit of the cover byte: 01101001 & 11111110 = 01101000
        # Then sets the least significant bit to the message bit: 01101000 | 00000001 = 01101001
        stego[i] = (cover[i] & ~1) | message[i]

    # Output result
    print("Cover: ", cover)
    print("Stego: ", stego)

def c():
    print("\nc) Simulate α = 0.4 of LSBR into an image using conseal, and measure the empirical change rate β̂, embedding rate α̂, and embedding efficiency ê.")
    cover = Image.open(IMAGE_PATH)
    cover_array = np.array(cover)
    stego_array = lsb.simulate(cover_array, alpha=0.4)

    # calculate empirical change rate
    empirical_change_vector = (stego_array.flatten() != cover_array.flatten()).astype(int) # 1 if pixel changed, 0 otherwise
    empirical_change_rate = np.mean(empirical_change_vector) # β̂ : proportion of pixels that changed
    print("Empirical change rate: ", empirical_change_rate)

    # estimate embedding: alpha_hat = 2 * beta_hat
    estimated_embedding_rate = ws.attack(stego_array) # α̂ : how many bits were embedded proportionally to the total number of pixels
    print("Estimated embedding rate: ", estimated_embedding_rate)

    embedding_efficiency = estimated_embedding_rate / empirical_change_rate # ê = α̂ / β̂
    print("Embedding efficiency: ", embedding_efficiency)

def main():
    a()
    c()

if __name__ == "__main__":
    main()