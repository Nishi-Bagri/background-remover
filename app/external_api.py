import os
import tempfile

from PIL import Image
from gradio_client import Client, handle_file
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Read Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")


# Check that the token exists
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set in the .env file."
    )


# Connect to the external BiRefNet Space
client = Client(
    "ZhengPeng7/BiRefNet_demo",
    token=HF_TOKEN
)


def remove_background_external(image):
    """
    Remove the background from a complex image
    using external BiRefNet.
    """

    # Create a temporary JPG file.
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp_file:

        temp_path = temp_file.name

    try:

        # Convert image to RGB.
        image = image.convert("RGB")

        # Reduce large images before uploading.
        image.thumbnail(
            (1024, 1024)
        )

        # Save as compressed JPEG.
        image.save(
            temp_path,
            format="JPEG",
            quality=85
        )

        # Send image to external BiRefNet.
        result = client.predict(
            handle_file(temp_path),
            "1024x1024",
            "Portrait",
            api_name="/image"
        )

        # The API can return multiple values.
        if isinstance(
            result,
            (list, tuple)
        ):
            result = result[-1]

        # If result is already a PIL image.
        if isinstance(
            result,
            Image.Image
        ):
            return result.convert("RGBA")

        # Otherwise open the returned image.
        return Image.open(
            result
        ).convert("RGBA")

    finally:

        # Delete temporary file.
        if os.path.exists(temp_path):
            os.remove(temp_path)