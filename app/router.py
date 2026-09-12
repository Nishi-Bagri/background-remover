# Score at or above this value is considered complex.
COMPLEXITY_THRESHOLD = 60


def route_image(score):
    # Scores below 60 use the local background-removal pipeline.
    if score < COMPLEXITY_THRESHOLD:
        return "local"

    # Scores at or above 60 use the external pipeline.
    return "external"