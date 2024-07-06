# STEP 1: Import the necessary modules. 패키지 가져오는 부분
from fastapi import FastAPI, File, UploadFile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)


app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()



    # content => jpg 파일인데, http통신에서는 파일이 character type(텍스트 형태) 으로 왔다갔다 함. (바이너리 아님)
    # 1. text -> binary     : io.BytesIO(text)
    # 2. binary -> PIL Image

    # STEP 3: Load the input image.

    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)


    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"top category: {top_category.category_name} ({top_category.score:.2f})"

    return {"result": result}