# Base image Python 3.10 slim tối ưu dung lượng
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho PyTorch & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip và dependencies
COPY app/requirements.txt /app/requirements.txt

# Cài đặt torch CPU-only trước để tối ưu dung lượng image và tăng tốc độ build CI/CD
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy toàn bộ code và model weights vào container
COPY app/ /app/

# Mở cổng 7860 cho Web GIS / FastAPI
EXPOSE 7860

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

