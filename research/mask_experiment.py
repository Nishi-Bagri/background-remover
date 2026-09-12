import numpy as np

mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

print(mask)
print("Shape:", mask.shape)
print("Background pixels:", np.count_nonzero(mask == 0))
print("Foreground pixel:", np.count_nonzero(mask == 1))

image = np.array([
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100],
    [110, 120, 130, 140, 150],
    [160, 170, 180, 190, 200]
])

foreground = image[mask == 1]
background = image[mask == 0]

print("Foreground values:", foreground)
print("Background values:", background)