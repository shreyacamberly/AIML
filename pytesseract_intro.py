import pytesseract
from PIL import Image

img_file = "Downloads/ss3.jpg"
no_noise = "Downloads/nonoise_ss3.jpg"

img = Image.open(no_noise)
ocr_result = pytesseract.image_to_string(img)
print(ocr_result)
