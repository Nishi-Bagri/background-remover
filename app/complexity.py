import cv2
import numpy as np

# Experimental reference values calculated from our test images.
# These are used to normalize the different complexity features.
EDGE_MAX = 0.103984375
LOCAL_MAX = 6.819639282735613
COLOR_MAX = 79.88913968105776

# Weight given to each feature in the final complexity score.
EDGE_WEIGHT = 0.30
LOCAL_WEIGHT = 0.50
COLOR_WEIGHT = 0.20

def complexity_score(image):
    """
    Calculate an image complexity score between 0 and 1.

    The score is based on:
    1. Edge density
    2. Local pixel variation
    3. Color variation
     
    """

    #Convert the image from BGR to grayscale.
    #Grayscale makes it easier to analyze edges and pixel variation.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using the Canny edge detector.

    edges = cv2.Canny(gray, 100, 200)

    #Calculate what percentage of pixels are edge pixels.

    edge_density = np.count_nonzero(edges) / edges.size

    #Compare neighboring pixels to measure Local variation.
    #int16 prevents unit8 substraction from wrapping around.

    difference = np.abs(
        gray[:, 1:].astype(np.int16) -
        gray[:, :-1].astype(np.int16)
    )

    #Average neighboring-pixel difference.
    
    local_variation = difference.mean()

    #Measure overall color variation in the image.
    
    color_variation = image.std()

    #Normalize each feature to approximately 0-1

    normalized_edge = np.clip(edge_density / EDGE_MAX, 0, 1)
    normalized_local = np.clip(local_variation / LOCAL_MAX, 0, 1)
    normalized_color = np.clip(color_variation / COLOR_MAX, 0, 1)

    #Combine the three features using their assigned weights.

    score = (
        normalized_edge * EDGE_WEIGHT
        + normalized_local * LOCAL_WEIGHT
        + normalized_color * COLOR_WEIGHT
    )

    return score * 100
    