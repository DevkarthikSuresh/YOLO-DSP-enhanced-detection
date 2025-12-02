import cv2
import numpy as np

def enhance(img_bgr):
    """
    Apply DSP-based enhancement to improve YOLO detection confidence.
    Steps:
        1. Non-local means denoising
        2. LAB color space CLAHE for contrast
        3. Gaussian blur + sharpening
    """
    # 1. Denoise
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)

    # 2. Convert to LAB + CLAHE on L-channel
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 3. Sharpen
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharp = cv2.addWeighted(contrast, 1.3, blur, -0.3, 0)

    return sharp
