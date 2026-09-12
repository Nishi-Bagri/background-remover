import numpy as np
from PIL import Image


# ============================================================
# 1. Load Image and Convert to NumPy Array
# ============================================================

image = Image.open("image.jpg")
arr = np.array(image)

print("Image shape:", arr.shape)
print("Data type:", arr.dtype)


# ============================================================
# 2. Inspect Individual Pixels
# ============================================================

print("\n--- Individual Pixels ---")

print("Pixel [0, 0]:", arr[0, 0])
print("Pixel [0, 1]:", arr[0, 1])
print("Pixel [1, 0]:", arr[1, 0])


# ============================================================
# 3. Extract RGB Channels
# ============================================================

red = arr[:, :, 0]
green = arr[:, :, 1]
blue = arr[:, :, 2]

print("\n--- Channel Shapes ---")

print("Red:", red.shape)
print("Green:", green.shape)
print("Blue:", blue.shape)


# ============================================================
# 4. Basic Statistics of RGB Channels
# ============================================================

print("\n--- RGB Statistics ---")

print("Red:", red.min(), red.max(), red.mean())
print("Green:", green.min(), green.max(), green.mean())
print("Blue:", blue.min(), blue.max(), blue.mean())

print("\n--- Standard Deviation ---")

print("Red std:", red.std())
print("Green std:", green.std())
print("Blue std:", blue.std())


# ============================================================
# 5. Compare Two Neighboring Pixels
# ============================================================

print("\n--- Single Pixel Difference ---")

red_difference = abs(
    int(red[0, 0]) - int(red[0, 1])
)

print("Red pixel difference:", red_difference)


# ============================================================
# 6. Compare All Horizontally Neighboring Pixels
# ============================================================

print("\n--- Neighboring Pixel Differences ---")

red_difference = np.abs(
    red[:, 1:].astype(np.int16) -
    red[:, :-1].astype(np.int16)
)

green_difference = np.abs(
    green[:, 1:].astype(np.int16) -
    green[:, :-1].astype(np.int16)
)

blue_difference = np.abs(
    blue[:, 1:].astype(np.int16) -
    blue[:, :-1].astype(np.int16)
)


# ============================================================
# 7. Difference Statistics
# ============================================================

print("\n--- Red Difference ---")

print("Shape:", red_difference.shape)
print("Maximum:", red_difference.max())
print("Mean:", red_difference.mean())


print("\n--- Green Difference ---")

print("Shape:", green_difference.shape)
print("Maximum:", green_difference.max())
print("Mean:", green_difference.mean())


print("\n--- Blue Difference ---")

print("Shape:", blue_difference.shape)
print("Maximum:", blue_difference.max())
print("Mean:", blue_difference.mean())


# ============================================================
# 8. Understanding NumPy Axis
# ============================================================

print("\n--- Axis Experiment ---")

print(
    "Red column means shape:",
    red.mean(axis=0).shape
)

print(
    "Red row means shape:",
    red.mean(axis=1).shape
)