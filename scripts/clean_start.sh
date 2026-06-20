#!/bin/bash
# Clean Start — Run all systems from scratch
# Usage: ./scripts/clean_start.sh [--train] [--mfi]

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "  LoRA Credit Scoring — Clean Start"
echo "=============================================="

# 1. Kill any running services
echo ""
echo "[1/6] Stopping existing processes..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "gradio" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 2

# 2. Clean generated files
echo ""
echo "[2/6] Cleaning generated files..."
rm -f models/baseline/*.pkl 2>/dev/null || true
rm -f models/lora/*.pt models/lora/*.bin 2>/dev/null || true
rm -f models/preprocessor.pkl models/dataset_name.txt models/score_calibration.json 2>/dev/null || true
rm -f results/metrics/*.json results/metrics/*.csv 2>/dev/null || true
rm -f results/figures/*.png results/reports/*.html 2>/dev/null || true
rm -f data/raw/*.csv data/raw/*.zip data/raw/*.data* data/raw/Index data/raw/german.doc 2>/dev/null || true
rm -f data/synthetic/*.csv 2>/dev/null || true
rm -f mfi_portal/data/*.db 2>/dev/null || true
find . -type d -name __pycache__ -not -path ./venv/* -exec rm -rf {} + 2>/dev/null || true
echo "   Done."

# 3. Ensure venv
echo ""
echo "[3/6] Activating venv..."
if [ ! -d "venv" ]; then
    echo "   Creating venv..."
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install -q -r requirements.txt
pip install -q sqlalchemy python-jose[cryptography] bcrypt
echo "   Done."

# 5. Train (optional)
TRAIN=0
for arg in "$@"; do
    case $arg in
        --train) TRAIN=1 ;;
    esac
done

if [ "$TRAIN" -eq 1 ]; then
    echo ""
    echo "[5/6] Training models (5K samples, 10 epochs)..."
    python scripts/train.py --dataset zimbabwe_synthetic --n-samples 5000 --epochs 10
    echo "   Done."
else
    echo ""
    echo "[5/6] Skipping training (use --train to run)"
fi

# 6. Summary
echo ""
echo "=============================================="
echo "  Clean Start Complete"
echo "=============================================="
echo ""
echo "Run in separate terminals (activate venv first):"
echo ""
echo "  Demo UI:      make run              → http://localhost:7860"
echo "  MFI API:      make mfi-api          → http://localhost:8000"
echo "  MFI Portal:   make mfi-front        → http://localhost:5174"
echo ""
echo "  Train models: ./scripts/clean_start.sh --train"
echo ""
