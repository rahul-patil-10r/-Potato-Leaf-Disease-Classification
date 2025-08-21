from fastapi import FastAPI ,File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import keras
from fastapi.responses import JSONResponse

model = tf.keras.models.load_model('/Users/rahulpatil/Desktop/potato_project/modelh5/model_1.h5')

app = FastAPI()


class_name=["early","normal","late"]

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Correctly use await to read the file and get bytes
    img_bytes = await file.read()
    
    # Convert the image bytes into a PIL Image object
    image = Image.open(io.BytesIO(img_bytes))
    
    # Convert the image into a numpy array
    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)
   
    model_ans=model.predict(img_array)
    confi=float(np.max(model_ans))
    model_ans=np.argmax(model_ans)

    ans=class_name[model_ans]
    

    
   
    return {
        "class": ans,
        "confidence": confi
    }

    

if __name__=="__main__" :
    uvicorn.run(app,host='localhost',port=8080)