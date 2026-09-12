import cv2
import numpy as np

from app.complexity import complexity_score
from app.router import route_image
from app.background_removal import remove_background
from app.external_api import remove_background_external
from app.groq_service import analyze_complex_image
from app.passport_processing import create_passport_photo


def process_image(image):
    """
    Complete passport-photo processing pipeline.

    1. Analyze image complexity.
    2. Route simple images to local U2Net.
    3. Route complex images to external BiRefNet API.
    4. Create the final passport photo.
    """

    # Keep the original image.
    # We use this later for face detection.
    original_image = image.convert("RGB")

    # Convert the PIL image to a NumPy array.
    image_array = np.array(
        original_image
    )

    # OpenCV expects BGR format.
    opencv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------
    # STEP 1: COMPLEXITY ANALYSIS
    # --------------------------------

    score = complexity_score(
        opencv_image
    )

    route = route_image(
        score
    )

    # --------------------------------
    # STEP 2: SIMPLE IMAGE
    # --------------------------------

    if route == "local":

        # Simple images use local U2Net.
        result = remove_background(
            original_image
        )

        # Create final passport photo.
        passport_photo = create_passport_photo(
            result,
            original_image
        )

        return {
            "score": score,
            "route": route,
            "status": "processed",
            "image": passport_photo
        }

    # --------------------------------
    # STEP 3: COMPLEX IMAGE
    # --------------------------------

    # Analyze the complex image using Groq Vision.
    analysis = analyze_complex_image(
        original_image
    )

    # Complex images use EXTERNAL BiRefNet.
    result = remove_background_external(
        original_image
    )

    # --------------------------------
    # STEP 4: PASSPORT PROCESSING
    # --------------------------------

    passport_photo = create_passport_photo(
        result,
        original_image
    )

    # --------------------------------
    # STEP 5: RETURN RESULT
    # --------------------------------

    return {
        "score": score,
        "route": route,
        "status": "ai_processed",
        "image": passport_photo,
        "analysis": analysis
    }