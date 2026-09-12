from rembg import remove
from PIL import Image


def remove_background(image):
    return remove(image)


# Test
input_image = Image.open("image.jpg")

output_image = remove_background(input_image)

output_image.save("test_removed.png")

print("Background removal completed.")