import cv2
import numpy as np

simple = cv2.imread("simple_background.png")
complex_image = cv2.imread("complex_background.png")

simple_gray = cv2.cvtColor(simple, cv2.COLOR_BGR2GRAY)
complex_gray = cv2.cvtColor(complex_image, cv2.COLOR_BGR2GRAY)

simple_edges = cv2.Canny(simple_gray, 100, 200)
complex_edges = cv2.Canny(complex_gray, 100, 200)

simple_density = np.count_nonzero(simple_edges) / simple_edges.size
complex_density = np.count_nonzero(complex_edges) / complex_edges.size

print("Simple edge density:", simple_density)
print("Complex edge density:", complex_density)

simple_difference = np.abs(
    simple_gray[:, 1:].astype(np.int16) - simple_gray[:, :-1].astype(np.int16)
)

complex_difference = np.abs(
    complex_gray[:, 1:].astype(np.int16) -
    complex_gray[:, :-1].astype(np.int16)
)

print("Simple mean local difference:", simple_difference.mean())

print("Complex mean local difference:", complex_difference.mean())

actual = cv2.imread("image.jpg")
actual_gray = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)

actual_edges = cv2.Canny(actual_gray, 100, 200)

actual_edge_density = (
    np.count_nonzero(actual_edges) / actual_edges.size
)

actual_difference = np.abs(
    actual_gray[:, 1:].astype(np.int16) -
    actual_gray[:, :-1].astype(np.int16)
)

actual_local_variation = actual_difference.mean()

print("Actual image edge density:", actual_edge_density)

print("Actual image mean local difference:", actual_local_variation)

print("\n--- Feature ranges ---")

print("Simple edge density:", simple_density)

print("Complex edge density:", complex_density)

print("Actual edge density:", actual_edge_density)

print("Simple local variation:", simple_difference.mean())

print("Complex local variation:", complex_difference.mean())

print("Actual local variation:", actual_local_variation)

# Experimental complexity score

reference_edge = complex_density
reference_variation = complex_difference.mean()

simple_edge_normalized = simple_density / reference_edge
complex_edge_normalized = complex_density / reference_edge
actual_edge_normalized = actual_edge_density / reference_edge

simple_variation_normalized = simple_difference.mean() / reference_variation

complex_variation_normalized = complex_difference.mean() / reference_variation

actual_variation_normalized = actual_local_variation / reference_variation

simple_score = (
    simple_edge_normalized + simple_variation_normalized
)/2

complex_score = (
    complex_edge_normalized + complex_variation_normalized
)/2

actual_score = (
    actual_edge_normalized + actual_variation_normalized
) / 2

print("\n--- Experimental complexity score ---")
print("Simple score:", simple_score)
print("Complex score:", complex_score)
print("Actual image score:", actual_score)

# Realistic background experiments

background_files = [
    "realistic_smooth_wall.png",
    "realistic_grass_texture.png",
    "realistic_brick_wall.png",
    "image.jpg"
]

for file in background_files:
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.count_nonzero(edges) / edges.size

    difference = np.abs(
        gray[:, 1:].astype(np.int16) -
        gray[:, :-1].astype(np.int16)
    )

    local_variation = difference.mean()

    print("\n", file)
    print("Edge density:", edge_density)
    print("Local variation:", local_variation)

    color_std = image.std()

    print("Color variation:", color_std)

# Min-Max normalization experiment

edge_values = np.array([
        0.0,
        0.015479166666666667,
        0.025686167061611375,
        0.103984375
    ])

minimum = edge_values.min()
maximum = edge_values.max()

actual_edge = 0.025686167061611375

normalized_edge = (
    actual_edge - minimum
) / ( maximum - minimum)

print("\n--- Normalization experiment ---")
print("Minimum:", minimum)
print("Maximum:", maximum)
print("Actual normalized edge:", normalized_edge)

# Normalize all three features

edge_values = np.array([
    0.0,
    0.015479166666666667,
    0.103984375, 
    0.025686167061611375 
])

local_values = np.array([
    0.0,
    0.7364678899082568,
    5.122332444427942,
    6.819639282735613
])

color_values = np.array([
    0.2800000000000004,
    25.127826085968202,
    27.4250828415052,
    79.88913968105776
])

actual_edge = 0.025686167061611375
actual_local = 5.122332444427942
actual_color = 79.88913968105776

def normalize(value, values):
    minimum = values.min()
    maximum = values.max()

    return(value - minimum) / (maximum - minimum)

normalized_edge = normalize(actual_edge, edge_values)
normalized_local = normalize(actual_local, local_values)

normalized_color = normalize(actual_color, color_values)

print("\n---Normalized features---")
print("Normalized edge:", normalized_edge)
print("Normalized local variation:", normalized_local)
print("Normalized color variation:", normalized_color)

complexity_score = (
    normalized_edge + normalized_local + normalized_color
)/3

print("n\---Multi-Feature complexity score ---")
print("Complexity score:", complexity_score)

test_names = [
    "Smooth wall",
    "Brick wall",
    "Grass",
    "Actual image"
]

for i in range(len(test_names)):
    edge_score = normalize(edge_values[i], edge_values)
    local_score = normalize(local_values[i], local_values)
    color_score = normalize(color_values[i], color_values)

    score = (
        edge_score + local_score + color_score
    ) / 3

    print(test_names[i], "->", score)

# Weighted complexity score

edge_weight = 0.30
local_weight = 0.50
color_weight = 0.20

weighted_score = (
    normalized_edge * edge_weight + normalized_local * local_weight + normalized_color * color_weight
)

print("\n---Weighted complexity score---")
print("weighted score:", weighted_score)


# Clipping experiment

values = np.array([-0.5, 0.2, 0.7, 1.0, 1.5])

clipped_values = np.clip(values, 0, 1)

print("\n---Clipping experiment---")
print("Original:", values)
print("Clipped:", clipped_values)

# Local complexity experiment on cmp9

image = cv2.imread("complex/cmp9.jpg")

height = image.shape[0]

lower_region = image[int(height * .60):, :]

gray_lower = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)

edges_lower = cv2.Canny(gray_lower, 100, 200)

edge_density_lower = (
    np.count_nonzero(edges_lower) / edges_lower.size
)


difference_lower = np.abs(
    gray_lower[:, 1:].astype(np.int16)
    - gray_lower[:, :-1].astype(np.int16)
)

local_variation_lower = difference_lower.mean()

color_variation_lower = lower_region.std()

print("\n---Local Complexity: cmp9 lower region---")
print("Edge density:", edge_density_lower)
print("Local variation:", local_variation_lower)
print("Color variuation:", color_variation_lower)