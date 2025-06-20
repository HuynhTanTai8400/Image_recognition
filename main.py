from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import io
import json
import base64
from PIL import Image
from openai import OpenAI

# Tải biến môi trường từ file .env
load_dotenv()

# Khởi tạo FastAPI app
app = FastAPI()

# Lấy OpenAI API key từ biến môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY không được tìm thấy trong tệp .env")

# Khởi tạo OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Mount thư mục 'static' để phục vụ các file tĩnh (CSS, JS, hình ảnh)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo Jinja2Templates để render HTML
templates = Jinja2Templates(directory="templates")

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

# Hàm chuyển đổi ảnh sang base64
def image_to_base64(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG") # Hoặc PNG tùy định dạng ảnh gốc
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Hàm gọi OpenAI Vision API để phân tích hình ảnh
async def analyze_image_with_openai(base64_image: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Hoặc "gpt-4-vision-preview" cho các tác vụ phức tạp hơn
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What Vietnamese dish is in this image? Provide only the name of the dish in Vietnamese, for example: 'Phở'. If it's not a Vietnamese dish, say 'Món ăn không xác định'."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lỗi khi gọi OpenAI Vision API: {e}")
        return "Món ăn không xác định"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Không có file nào được tải lên.")

    try:
        image_bytes = await file.read()
        
        # Chuyển đổi ảnh sang base64 để gửi tới OpenAI
        img_base64 = image_to_base64(image_bytes)

        # Phân tích hình ảnh bằng OpenAI Vision API
        predicted_label = await analyze_image_with_openai(img_base64)
        
        # Tìm các nhà hàng dựa trên nhãn dự đoán
        suggestions = restaurants_by_label.get(predicted_label, [])

        # URL hình ảnh để hiển thị trên web
        img_url_display = f"data:image/{file.content_type.split('/')[-1]};base64,{img_base64}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "label": predicted_label,
                "confidence": "Được xác định bởi AI", # Không có confidence từ OpenAI Vision API theo cách này
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

# Để chạy ứng dụng này, sử dụng lệnh: uvicorn main:app --reload --port 5000