from fastapi import APIRouter
from starlette.responses import JSONResponse
from fastapi import UploadFile
from iris.iris_classifier import IrisClassifier
from iris.models import Iris
import json


router = APIRouter()


@router.post("/upload")
def upload(file: UploadFile):
    content = file.file.read()
    content_json = json.loads(content)
    print(content_json)
    return {"filename": file.filename}


@router.post("/classify")
def classify(inputs: UploadFile):

    try:
        # Parse inputs into JSON
        iris_features = json.loads(inputs.file.read())

        # Output
        classified_iris = []

        # Init classifier
        iris_classifier = IrisClassifier()

        # Classify all input instances
        for input in iris_features:
            iris = Iris(**input)
            result = iris_classifier.classify_iris(iris)
            classified_iris.append({"features": input, "output": result})

        return JSONResponse(classified_iris)

    except Exception as e:
        return JSONResponse({"error": str(e)})
