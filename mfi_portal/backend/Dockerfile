# MFI Portal API — needs LoRA models and src/ from project root
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mfi_portal/backend/requirements.txt mfi_portal/backend/requirements.txt
RUN pip install --no-cache-dir -r mfi_portal/backend/requirements.txt

COPY src/ src/
COPY mfi_portal/backend/ mfi_portal/backend/

WORKDIR /app/mfi_portal/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
