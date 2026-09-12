from rembg import remove
from PIL import Image

input_image = Image.open("image.jpg")

output_image = remove(input_image)

output_image.save("removed_background.png")

print("Background removal completed.")