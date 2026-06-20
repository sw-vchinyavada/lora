# LoRA Credit Scoring - Production Image
FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/

# Pre-generate Zimbabwe synthetic data on build (optional)
RUN python -c "from src.data import load_dataset; load_dataset('zimbabwe_synthetic', n_samples=1000)" || true

EXPOSE 7860

# Default: run UI; override with train for one-shot training
CMD ["python", "-m", "app.main"]
