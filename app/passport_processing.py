import cv2
import numpy as np
from PIL import Image


# Final passport-style output size.
# 413 x 531 follows a 35:45 portrait ratio.
TARGET_WIDTH = 413
TARGET_HEIGHT = 531


# Load OpenCV's built-in frontal-face detector.
FACE_CASCADE = cv2.CascadeClassifier(
    "app/haarcascade_frontalface_default.xml"
)


def add_white_background(image):
    """
    Put the transparent image onto a white background.
    """

    # Make sure the image has an alpha channel.
    image = image.convert("RGBA")

    # Create a white background with the same dimensions.
    white_background = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    # Place the foreground onto the white background.
    white_background.alpha_composite(image)

    # Return a normal RGB image.
    return white_background.convert("RGB")


def detect_face(image):
    """
    Detect the largest face in the original image.

    Returns:
        (x, y, width, height)
        or None if no face is detected.
    """

    # Convert the original image to RGB.
    image = image.convert("RGB")

    # Convert the Pillow image to a NumPy array.
    image_array = np.array(image)

    # Convert RGB to grayscale.
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces in the original image.
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Return None if no face was detected.
    if len(faces) == 0:
        return None

    # Select the largest detected face.
    return tuple(
        max(
            faces,
            key=lambda face: face[2] * face[3]
        )
    )


def create_passport_photo(
    removed_image,
    original_image
):
    """
    Create a passport-style photo.

    The original image is used for face detection.
    The background-removed image is used for the final result.
    """

    # Make sure the processed image has transparency.
    removed_image = removed_image.convert("RGBA")

    # Get the foreground transparency mask.
    alpha = removed_image.getchannel("A")

    # Find the foreground bounding box.
    subject_bbox = alpha.getbbox()

    # If background removal found nothing,
    # return a blank white image.
    if subject_bbox is None:
        return Image.new(
            "RGB",
            (TARGET_WIDTH, TARGET_HEIGHT),
            "white"
        )

    # Detect the face using the ORIGINAL image.
    face = detect_face(original_image)

    if face is not None:

        # Face coordinates from the original image.
        face_x, face_y, face_width, face_height = face

        # Calculate the centre of the face.
        face_center_x = face_x + face_width / 2

        # Create a portrait crop based on face size.
        #
        # This includes the head, shoulders and
        # part of the upper body.
        crop_height = int(face_height * 7)

        # Calculate the width using our target aspect ratio.
        crop_width = int(
            crop_height
            * TARGET_WIDTH
            / TARGET_HEIGHT
        )

        # Make sure the crop is wide enough around the face.
        crop_width = max(
            crop_width,
            int(face_width * 2.5)
        )

        # Recalculate the height to preserve the
        # 35:45 passport ratio.
        crop_height = int(
            crop_width
            * TARGET_HEIGHT
            / TARGET_WIDTH
        )

        # Centre the crop around the face.
        crop_left = int(
            face_center_x - crop_width / 2
        )

        # Leave some space above the head.
        crop_top = int(
            face_y - face_height * 0.8
        )

        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        # Get dimensions of the processed image.
        image_width, image_height = removed_image.size

        # Keep the crop inside the image.
        if crop_left < 0:
            crop_right -= crop_left
            crop_left = 0

        if crop_top < 0:
            crop_bottom -= crop_top
            crop_top = 0

        if crop_right > image_width:
            shift = crop_right - image_width
            crop_left -= shift
            crop_right = image_width

        if crop_bottom > image_height:
            shift = crop_bottom - image_height
            crop_top -= shift
            crop_bottom = image_height

        # Final safety limits.
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(image_width, crop_right)
        crop_bottom = min(image_height, crop_bottom)

        # Crop the background-removed image.
        cropped = removed_image.crop(
            (
                crop_left,
                crop_top,
                crop_right,
                crop_bottom
            )
        )

    else:

        # If the face detector cannot find a face,
        # use the complete foreground as a fallback.
        cropped = removed_image.crop(subject_bbox)

    # Resize the cropped image to the final passport dimensions.
    cropped = cropped.resize(
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )

    # Create a clean white passport background.
    passport = Image.new(
        "RGBA",
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        ),
        (255, 255, 255, 255)
    )

    # Place the processed subject on the white background.
    passport.alpha_composite(
        cropped,
        (0, 0)
    )

    # Return the final RGB image.
    return passport.convert("RGB")