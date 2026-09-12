import os
import base64
import json

from groq import Groq
from dotenv import load_dotenv


# Load environment variables from the .env file.
load_dotenv()


# Read the Groq API key from the environment.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Create the Groq client.
client = Groq(
    api_key=GROQ_API_KEY
)


def analyze_complex_image(image):
    """
    Send a complex image to Groq Vision for AI analysis.

    Groq identifies:
    - the main subject
    - the type of background
    - difficult areas for segmentation
    - the overall background-removal difficulty

    The actual pixel-level background removal is performed
    later by BiRefNet.
    """

    # Import BytesIO here because the image only needs
    # to exist temporarily in memory.
    from io import BytesIO


    # Create an in-memory buffer for the image.
    image_buffer = BytesIO()


    # Convert the image to RGB and save it as JPEG.
    image.convert("RGB").save(
        image_buffer,
        format="JPEG",
        quality=95
    )


    # Convert the image bytes to Base64.
    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")


    # Ask the vision model to understand the image
    # specifically from a background-removal perspective.
    prompt = """
Analyze this image for an AI background-removal pipeline.

Identify:

1. The main foreground subject.
2. The type of background.
3. Whether the background contains:
   - textures
   - objects
   - people
   - vegetation
   - buildings
   - shadows
   - overlapping elements
4. The difficult areas around the subject, such as:
   - hair
   - shoulders
   - clothing edges
   - hands
   - transparent objects
   - overlapping objects
5. The expected segmentation difficulty.

Return ONLY valid JSON in this format:

{
    "subject": "...",
    "background": "...",
    "complexity": "low | medium | high",
    "difficult_areas": [
        "...",
        "..."
    ],
    "segmentation_notes": "..."
}
"""


    # Send the image and prompt to the Groq
    # vision-capable model.
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",

        messages=[
            {
                "role": "user",

                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{encoded_image}"
                            )
                        }
                    }
                ]
            }
        ],

        # We only need a short structured analysis.
        max_completion_tokens=400
    )


    # Get the text returned by Groq.
    analysis_text = (
        response
        .choices[0]
        .message
        .content
    )


    # Try to convert Groq's JSON response into
    # a Python dictionary.
    try:

        analysis = json.loads(
            analysis_text
        )

    except json.JSONDecodeError:

        # If the model returns something that is not
        # valid JSON, keep the original response
        # rather than crashing the application.
        analysis = {
            "subject": "Unknown",
            "background": "Unknown",
            "complexity": "high",
            "difficult_areas": [],
            "segmentation_notes": analysis_text
        }


    # Return the structured AI analysis.
    return analysis