import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg', 'missile.jpg', 'missile3.png']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

# https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
# https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0) # 특정 키 입력 시까지 대기


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)





# STEP 1: Import the necessary modules. 패키지 가져오는 부분
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)


# STEP 3: Load the input image.
image = mp.Image.create_from_file(f"images\{IMAGE_FILENAMES[3]}")

# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
# print(classification_result)

print(classification_result.classifications[0].categories[0])
print(classification_result.classifications[0].categories[1])
print(classification_result.classifications[0].categories[2])
print(classification_result.classifications[0].categories[3])
# for i in range(len(classification_result)):
#     arr = [classification_result.classifications[i], categories[]]

# # STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
print(f"top category: {top_category.category_name} ({top_category.score:.2f})")


