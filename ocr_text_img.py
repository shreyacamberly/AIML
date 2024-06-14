#open an image

import cv2
import matplotlib.pyplot as plt
image_file = "Downloads/ss3.jpg"
img = cv2.imread(image_file)

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    # Check if the image is grayscale or color
    if len(im_data.shape) == 2:  # Grayscale image
        height, width = im_data.shape
        figsize = width / float(dpi), height / float(dpi)
    else:  # Color image
        height, width, depth = im_data.shape
        figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(im_data, cmap='gray' if len(im_data.shape) == 2 else None)
    plt.show()

display(image_file)

#inverting the image
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("Downloads/inverted_ss3.jpg", inverted_image)
display("Downloads/inverted_ss3.jpg")


#rescaling ...later 
binarisation(converting an image in black n white):
#before binarising make it a grayscale image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = grayscale(img)
cv2.imwrite("Downloads/gray_ss3.jpg", gray_image)
display("Downloads/gray_ss3.jpg")


#binaarization

thresh, im_bw = cv2.threshold(gray_image, 200,230, cv2.THRESH_BINARY)
cv2.imwrite("Downloads/binary_ss3.jpg", im_bw)
display("Downloads/binary_ss3.jpg")


#noise removal

def noise_removal(image):
    import numpy as np 
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)
    return(image)
  no_noise = noise_removal(im_bw)
cv2.imwrite("Downloads/nonoise_ss3.jpg",no_noise)
display("Downloads/nonoise_ss3.jpg")

#dilation and erosion(when font is too thick/thin --> adjustment)
def thin_font(image):
    import numpy as np 
    image=cv2.bitwise_not(image) #invertin the image
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image) #inverts again after the task (thinning is done)
    return(image)
  eroded_image = thin_font(no_noise)
cv2.imwrite("Downloads/eroded_ss3.jpg", eroded_image)
display("Downloads/eroded_ss3.jpg")

#dilation (thickening of fomt)
def thick_font(image):
    import numpy as np 
    image=cv2.bitwise_not(image) #invertin the image
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image) #inverts again after the task (thickening is done)
    return(image)
  dilated_image = thick_font(no_noise)
cv2.imwrite("Downloads/Dilated_ss3.jpg", dilated_image)
display("Downloads/Dilated_ss3.jpg")


# Rotation n Desweing (usually take up codes codes from net n use em to do the same)
#(do this after uve removed the borders of the pic)

new =cv2.imread("Downloads/skewedpic.jpg")
display("Downloads/skewedpic.jpg")

#the func we'll be using:
import numpy as np

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage
  # Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

fixed = deskew(new)
cv2.imwrite("Downloads/fixed_ss3.jpg",fixed)
display("Downloads/fixed_ss3.jpg")


#removing inconsistent borders:
display("Downloads/nonoise_ss3.jpg")
#used when inconsistent borders otherwise normal borders ke liye use pdf editor etc
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)
no_borders = remove_borders(no_noise)
cv2.imwrite("Downloads/noborders_ss3.jpg", no_borders)
display("Downloads/noborders_ss3.jpg") 

#adding borders(used when few words r at da edge so we add a bordeer so dat they cld be included in ocr)
color = [255,255,255]
top, bottom, left, right = [150]*4
image_with_border = cv2.copyMakeBorder(no_borders,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)
cv2.imwrite("Downloads/borderimage_ss3.jpg",image_with_border)
display("Downloads/borderimage_ss3.jpg")




















