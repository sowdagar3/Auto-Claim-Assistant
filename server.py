import sys
from fastapi import FastAPI, HTTPException,File,UploadFile
from fastapi.responses import FileResponse
import uuid
from pathlib import Path
import shutil
import io
import cv2
import numpy as np
from predict_damage import *
import uvicorn
import os

app = FastAPI(title="Insurance Claim Assistant")


@app.get("/")
async def read_main():
    return {"msg": "Hello Let's claim your insurance !!!"}


# In[3]:


@app.post("/upload-image-predict-damage")
async def upload_and_predict_tf(file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    file.filename=f"{uuid.uuid4()}.png"
    SERVER_DIR="server_database"
    os.makedirs(SERVER_DIR,exist_ok=True)
    temp_file = os.path.join(os.getcwd(),SERVER_DIR,file.filename)
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite(f'{temp_file}', image)
    #prediction = tf_run_classifier(temp_file)
    prediction = predict(temp_file)
#     json_response={"status_code": 200,
#             "predicted_label": prediction["predicted_label"],
#             "probability": prediction["probability"]}
    return prediction



if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9001, log_level="debug",
                proxy_headers=True, reload=True)


# In[ ]:




