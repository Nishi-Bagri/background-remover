from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from app.processor import process_image


# Create the FastAPI application.
app = FastAPI(
    title="Passport Background Remover",
    description="Background removal and passport photo processing API"
)


# Serve the frontend files from the static folder.
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)


@app.get("/")
def home():
    """
    Open the main passport-photo application.
    """

    # Return the HTML page.
    from fastapi.responses import FileResponse

    return FileResponse(
        "app/static/index.html"
    )


@app.post("/process")
async def process_photo(file: UploadFile = File(...)):
    """
    Receive an uploaded image and process it.
    """

    # Read the uploaded file.
    contents = await file.read()

    # Convert the uploaded bytes into a Pillow image.
    image = Image.open(
        io.BytesIO(contents)
    )

    # Send the image through our processing pipeline.
    result = process_image(image)

    # Return the processed image.
    if result["status"] in [
        "processed",
        "ai_processed"
    ]:

        # Create an in-memory output buffer.
        output = io.BytesIO()

        # Save the final image as PNG.
        result["image"].save(
            output,
            format="PNG"
        )

        # Move the pointer back to the beginning.
        output.seek(0)

        # Return the image to the browser.
        return StreamingResponse(
            output,
            media_type="image/png"
        )


    # Return information if the image wasn't processed.
    return {
        "filename": file.filename,
        "complexity_score": result["score"],
        "route": result["route"],
        "status": result["status"],
        "message": result.get("message"),
        "analysis": result.get("analysis")
    }