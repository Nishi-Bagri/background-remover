from PIL import Image

from app.external_api import remove_background_external


image = Image.open("image.jpg")

result = remove_background_external(image)

result.save("test_result.png")

print("Background removal successful!")

print("Saved as: test_result.png")