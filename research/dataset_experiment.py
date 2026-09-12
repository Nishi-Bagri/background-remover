from pathlib import Path
import cv2
from complexity_function import complexity_score

folder = Path("complex")

print(folder)

image_files = folder.glob("*.jpg")

for file in image_files:
    image = cv2.imread(str(file))

    score = complexity_score(
        image,
        edge_max=0.103984375,
        local_max=6.819639282735613,
        color_max=79.88913968105776
    )

    print(file, "->", score)