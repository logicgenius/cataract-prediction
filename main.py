from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 載入TFLite模型
interpreter = tflite.Interpreter(model_path="cataract.tflite")
interpreter.allocate_tensors()

# 讀出模型輸入／輸出細節
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 自動取得模型期望的影像高、寬
# input_shape[0] = batch size (1)，[1]=height，[2]=width，[3]=channels
_, input_h, input_w, _ = input_details[0]['shape']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀檔與轉成 RGB
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    # 改用模型實際需要的寬高 resize
    image = image.resize((input_w, input_h))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_mapping = {0: "正常", 1: "不正常", 2: "術後"}
    result = class_mapping[predicted_class]

    return {"prediction": result}