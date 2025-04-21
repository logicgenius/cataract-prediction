# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import numpy as np
# import io
# from PIL import Image
# import tflite_runtime.interpreter as tflite

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # 載入TFLite模型
# interpreter = tflite.Interpreter(model_path="cataract.tflite")
# interpreter.allocate_tensors()

# # 讀出模型輸入／輸出細節
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # 自動取得模型期望的影像高、寬
# # input_shape[0] = batch size (1)，[1]=height，[2]=width，[3]=channels
# _, input_h, input_w, _ = input_details[0]['shape']

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # 讀檔與轉成 RGB
#     content = await file.read()
#     image = Image.open(io.BytesIO(content)).convert("RGB")
#     # 改用模型實際需要的寬高 resize
#     image = image.resize((input_w, input_h))

#     img_array = np.array(image, dtype=np.float32) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     interpreter.set_tensor(input_details[0]['index'], img_array)
#     interpreter.invoke()

#     prediction = interpreter.get_tensor(output_details[0]['index'])
#     predicted_class = np.argmax(prediction, axis=1)[0]

#     class_mapping = {0: "正常", 1: "不正常", 2: "術後"}
#     result = class_mapping[predicted_class]

#     return {"prediction": result}

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --------- 載入並初始化 TFLite Interpreter ---------
interpreter = tflite.Interpreter(model_path="cataract.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 取得模型期望的輸入高、寬與資料型別
_, input_h, input_w, _ = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# 類別對應
class_mapping = {0: "正常", 1: "不正常", 2: "術後"}

# --------- 首頁路由 ---------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------- 預測路由 ---------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 讀圖並轉 RGB，調整到模型輸入尺寸
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image = image.resize((input_w, input_h))

    # 2. 轉成 numpy array
    arr = np.array(image)

    # 3. 依模型輸入 dtype 做前處理
    if input_dtype == np.float32:
        # EfficientNetV2 預處理：把像素從 [0,255] 映到 [-1,1]
        arr = arr.astype(np.float32)
        arr = (arr - 127.5) / 127.5
    elif input_dtype == np.uint8:
        # 量化版模型：直接 cast，無 normalization
        arr = arr.astype(np.uint8)
    else:
        # 其他型別：直接 cast
        arr = arr.astype(input_dtype)

    # 4. 加入 batch 維度
    input_data = np.expand_dims(arr, axis=0)

    # 5. 推論
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # 6. 取 argmax 並回傳
    pred_idx = int(np.argmax(output_data))
    result = class_mapping[pred_idx]

    # 7. 回傳預測結果與原始分數 (debug 用)
    return {
        "prediction": result,
        "scores": {
            "正常": float(output_data[0]),
            "不正常": float(output_data[1]),
            "術後": float(output_data[2])
        }
    }
