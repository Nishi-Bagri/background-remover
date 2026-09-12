from PIL import Image

image = Image.open("image.jpg")

print(image.format , image.size, image.mode)

image = image.convert("RGBA")

print(image.mode)

image.save("image.png")

alpha = image.getchannel("A")

print(alpha.mode)
print(alpha.size)