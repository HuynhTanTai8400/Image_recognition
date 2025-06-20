from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv # Giữ lại nếu bạn có các biến môi trường khác
import os
import io
import json
import base64
from PIL import Image

# Thư viện cho model Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image # Cần cho image.img_to_array
import numpy as np

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Khởi tạo FastAPI app
app = FastAPI()

# ==============================================================================
# CẤU HÌNH VÀ TẢI MODEL TỪ LOCAL (SỬ DỤNG GIT LFS)
# ==============================================================================

# Đường dẫn tới file model của bạn trong project (sau khi Git LFS đã tải về)
# Đảm bảo file này nằm trong thư mục 'models/' trong repository GitHub của bạn
MODEL_PATH = "models/base_model_trained.keras"

# Tải model vào bộ nhớ khi ứng dụng khởi động
keras_model = None # Biến global để giữ model
try:
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI KHẨN CẤP: Không tìm thấy file model tại {MODEL_PATH}.")
        print("Vui lòng đảm bảo model đã được push lên GitHub bằng Git LFS và được tải về thành công.")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    keras_model = load_model(MODEL_PATH)
    print("Mô hình Keras đã được load thành công vào bộ nhớ.")
except Exception as e:
    print(f"LỖI KHẨN CẤP: Không thể load model từ {MODEL_PATH} vào bộ nhớ.")
    print(f"Chi tiết lỗi: {e}")
    # Đảm bảo ứng dụng sẽ thất bại nếu không load được model
    raise RuntimeError(f"Không thể load model từ {MODEL_PATH}. Vui lòng kiểm tra file model và Git LFS.") from e

# Load tên các lớp (class names) - Từ đoạn code Flask của bạn
classes = [
    'Bánh bèo', 'Bánh bột lọc', 'Bánh căn', 'Bánh canh', 'Bánh chưng',
    'Bánh cuốn', 'Bánh đúc', 'Bánh giò', 'Bánh khọt', 'Bánh mì',
    'Bánh pía', 'Bánh tét', 'Bánh tráng nướng', 'Bánh xèo', 'Bún bò Huế',
    'Bún đậu mắm tôm', 'Bún mắm', 'Bún riêu', 'Bún thịt nướng', 'Cá kho tộ',
    'Canh chua', 'Cao lầu', 'Cháo lòng', 'Cơm tấm', 'Gỏi cuốn',
    'Hủ tiếu', 'Mì Quảng', 'Nem chua', 'Phở', 'Xôi xéo'
]

# ==============================================================================

# Load mock metadata từ JSON/local file
try:
    with open("restaurant_data.json", "r", encoding="utf-8") as f:
        restaurants_by_label = json.load(f)
    print("Dữ liệu nhà hàng đã được load thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'restaurant_data.json'. Vui lòng đảm bảo file này tồn tại.")
    restaurants_by_label = {} # Khởi tạo rỗng để tránh lỗi tiếp theo
except json.JSONDecodeError:
    print("Lỗi: Không thể parse file 'restaurant_data.json'. Vui lòng kiểm tra định dạng JSON.")
    restaurants_by_label = {}

# Mount thư mục 'static' để phục vụ các file tĩnh (CSS, JS, hình ảnh)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo Jinja2Templates để render HTML
templates = Jinja2Templates(directory="templates")

# Hàm tiền xử lý hình ảnh cho model Keras (từ Flask code, điều chỉnh cho FastAPI)
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Tiền xử lý hình ảnh đầu vào để phù hợp với model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Thay đổi kích thước về 300x300 (phù hợp với model của bạn)
    img = img.resize((300, 300))
    img_array = image.img_to_array(img) / 255.0 # Chuyển đổi thành array và chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch
    return img_array

# Hàm dự đoán bằng model Keras
async def predict_with_keras_model(image_bytes: bytes):
    if keras_model is None:
        print("Lỗi: Mô hình Keras chưa được tải hoặc bị lỗi.")
        return "Món ăn không xác định (Lỗi model)", 0.0
    
    try:
        img_input = preprocess_image(image_bytes)
        preds = keras_model.predict(img_input)[0]

        index = int(np.argmax(preds))
        label = classes[index]
        confidence = float(preds[index]) * 100
        
        return label, confidence
    except Exception as e:
        print(f"Lỗi khi dự đoán bằng model Keras: {e}")
        return "Món ăn không xác định (Lỗi dự đoán)", 0.0

# Hàm chuyển đổi ảnh sang base64 để hiển thị trên web
def image_to_base64(image_bytes: bytes, content_type: str) -> str:
    # Đảm bảo định dạng đầu ra phù hợp với content_type của ảnh gốc
    format = content_type.split('/')[-1].upper()
    if format == 'JPG': # PIL dùng 'JPEG' thay vì 'JPG'
        format = 'JPEG'
    
    img = Image.open(io.BytesIO(image_bytes))
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Không có file nào được tải lên.")

    try:
        image_bytes = await file.read()
        
        # Chuyển đổi ảnh sang base64 để hiển thị trên web
        img_base64 = image_to_base64(image_bytes, file.content_type)

        # Phân tích hình ảnh bằng model Keras cục bộ
        predicted_label, confidence = await predict_with_keras_model(image_bytes)
        
        # Tìm các nhà hàng dựa trên nhãn dự đoán
        suggestions = restaurants_by_label.get(predicted_label, [])

        # URL hình ảnh để hiển thị trên web
        img_url_display = f"data:image/{file.content_type.split('/')[-1]};base64,{img_base64}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "label": predicted_label,
                "confidence": f"{confidence:.2f}%", # Hiển thị độ tự tin
                "restaurants": suggestions,
                "image": img_url_display
            }
        })
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": {
                "label": "Lỗi xử lý", 
                "confidence": "N/A", 
                "restaurants": [], 
                "image": None,
                "error_message": f"Đã xảy ra lỗi khi xử lý ảnh: {e}"
            }
        })

# Để chạy ứng dụng này trên local, sử dụng lệnh:
# uvicorn main:app --reload --host 0.0.0.0 --port 5000
# Khi deploy lên Render.com, lệnh Start Command sẽ là:
# uvicorn main:app --host 0.0.0.0 --port $PORT