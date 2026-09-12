import cv2
import numpy as np

def complexity_score(image, edge_max, local_max, color_max):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size

    difference = np.abs(
        gray[:, 1:].astype(np.int16) -
        gray[:, :-1].astype(np.int16)
    )

    local_variation = difference.mean()

    color_variation = image.std()

    normalized_edge = edge_density / edge_max
    normalized_local = local_variation / local_max
    normalized_color = color_variation / color_max

    normalized_edge = np.clip(normalized_edge, 0, 1)
    normalized_local = np.clip(normalized_local, 0, 1)
    normalized_color = np.clip(normalized_color, 0, 1)

    edge_weight = 0.30
    local_weight = 0.50
    color_weight = 0.20

    score = (
    normalized_edge * edge_weight
    + normalized_local * local_weight
    + normalized_color * color_weight
)

    return score

image = cv2.imread("image.jpg")

score = complexity_score(
    image,
    edge_max=0.103984375,
    local_max=6.819639282735613,
    color_max=79.88913968105776
)

print("Complexity score:", score)