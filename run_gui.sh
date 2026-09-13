#!/usr/bin/env bash
# Run the ICU Ventilator Digital Twin dashboard on Linux / macOS / Pi.
# Opens in browser at http://localhost:8050
set -e
cd "$(dirname "$0")"

# First-time deps check
python3 -c "import dash, plotly, joblib, xgboost, sklearn, pandas, numpy" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install --break-system-packages dash plotly joblib xgboost scikit-learn \
        pandas numpy python-pptx imbalanced-learn
}

echo ""
echo "========================================================="
echo "  ICU Ventilator Digital Twin - Dashboard"
echo "  Open:  http://localhost:8050"
echo "  Stop:  Ctrl+C"
echo "========================================================="
echo ""

python3 gui/app.py
