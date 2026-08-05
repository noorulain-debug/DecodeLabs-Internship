import os
import cv2
import pytesseract

# 1. POINT TO TESSERACT INSTALL 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2. LOAD A SAMPLE IMAGE 
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "sample_text.jpg")

image = cv2.imread(image_path)

if image is None:
    print(f"Could not load image at '{image_path}'. Check the file exists in this folder.")
else:
    # 3. GRAYSCALE CONVERSION — collapses 3D color matrix into 1D intensity matrix
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. GAUSSIAN BLUR — smooths noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. ADAPTIVE THRESHOLDING — forces pure black/white for max contrast
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. RUN OCR — extract text, using PSM 6 (single uniform block of text)
    extracted_text = pytesseract.image_to_string(thresh, config="--psm 6")

    # 7. DISPLAY RESULTS CLEARLY
    print("=== Extracted Text ===")
    text = extracted_text.strip()
    if text:
        print(text)
    else:
        print("No text detected.")

    # 8. SAVE THE PROCESSED IMAGE SO YOU CAN VISUALLY CONFIRM THE PIPELINE WORKED
    cv2.imwrite("processed_output.png", thresh)
    print("\nProcessed (thresholded) image saved as processed_output.png")