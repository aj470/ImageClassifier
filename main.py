
import images
# dimensions in characters
FACE_WIDTH = 60
FACE_HEIGHT = 70

DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28


test_faces = images.load_images("digitdata/testimages", DIGIT_HEIGHT)
test_faces[1].print()
