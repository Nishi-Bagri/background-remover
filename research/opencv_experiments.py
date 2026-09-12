import cv2
import numpy as np

image = cv2.imread("image.jpg")

print("Original:", image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray.shape)
print(gray.dtype)
print(gray.min())
print(gray.max())
print(gray.mean())
print(gray.std())

edges = cv2.Canny(gray, 100, 200)

print("Edges shape:", edges.shape)
print("Edges dtype:", edges.dtype)
print("Edges min:", edges.min())
print("Edges max:", edges.max())

edge_pixels = np.count_nonzero(edges)

total_pixels = edges.size

edge_density = edge_pixels/ total_pixels

print("Edge pixels:", edge_pixels)
print("Total pixels:", total_pixels)
print("Edge density:", edge_density)

edges_low = cv2.Canny(gray, 50, 100)
edges_high = cv2.Canny(gray, 150, 300)

print("Edge density (50, 100):", np.count_nonzero(edges_low) / edges_low.size)

print("Edge density (100, 200):", np.count_nonzero(edges) / edges.size)

print("Edge density (150, 300):", np.count_nonzero(edges_high) / edges_high.size)