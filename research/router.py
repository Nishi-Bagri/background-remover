COMPLEXITY_THRESHOLD = 0.45

def route_image(score):
    if score < COMPLEXITY_THRESHOLD:
        return "simple"
    else:
        return "complex"

# Test
score = 0.649663279251838

route = route_image(score)

print("Complexity score:", score)
print("Route:", route)