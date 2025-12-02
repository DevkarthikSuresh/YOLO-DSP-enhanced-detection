import cv2
import json
import os
from src.preprocess import enhance
from src.detector import load_model, run_detect
from src.compare import summarize

# --- Paths ---
IMG_PATH = "examples/forestfireschernobyl.png"
WEIGHTS_PATH = "models/best.pt"   # user downloads and places here

def main():

    # Load model
    model = load_model(WEIGHTS_PATH)

    # Load raw image
    raw = cv2.imread(IMG_PATH)
    if raw is None:
        raise SystemExit(f"Error: Could not load image at {IMG_PATH}")

    # Apply DSP enhancement
    enh = enhance(raw)

    # Run detections
    raw_out, raw_vis = run_detect(model, raw)
    enh_out, enh_vis = run_detect(model, enh)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save overlays
    cv2.imwrite("results/raw_overlay.png", raw_vis)
    cv2.imwrite("results/enh_overlay.png", enh_vis)

    # Save JSON outputs
    with open("results/raw_detections.json", "w") as f:
        json.dump(raw_out, f, indent=2)

    with open("results/enh_detections.json", "w") as f:
        json.dump(enh_out, f, indent=2)

    # Print metrics
    print("RAW :", summarize(raw_out))
    print("ENH :", summarize(enh_out))

if __name__ == "__main__":
    main()
