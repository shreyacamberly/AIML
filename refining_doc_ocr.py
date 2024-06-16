import pytesseract
import cv2

image = cv2.imread("Downloads/ss4.jpg")
base_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("Downloads/gray_ss4.jpg",gray)

blur = cv2.GaussianBlur(gray, (7,7), 0)
cv2.imwrite("Downloads/blur_ss4.jpg", blur)

thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("Downloads/thresh_ss4.jpg", thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
cv2.imwrite("Downloads/kernel_ss4.jpg", kernel)

dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite("Downloads/dilate_ss4.jpg", dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])


results = [] #empty list created so that all of the ocr can be performed on da text together
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h>200 and w>20 :#condn for selecting the whole columns only
        
        roi = image[y:y+h, x:x+h]
        cv2.imwrite("Downloads/roi_ss4.jpg", roi)
        cv2.rectangle(image, (x,y), (x+w, y+h), (36,225,12), 2)
        
        ocr_result = pytesseract.image_to_string(roi)
        ocr_result = ocr_result.split("\n") #seprates text by one line
        for item in ocr_result:
            results.append(item)
cv2.imwrite("Downloads/bbox_ss4.jpg", image)
print(results)

#now we'll iterate in this list to find the names on or classify all the characters uniquely
entities = []
for item in results:
    item = item.strip().replace("\n","") #here in replace meh we replace /n by "" so dat line brks 
                                         #gets negated or exchanged by no line brks this makes it more
                                         #more wrkable for us
    item = item.split(" ")[0] #0 shows the 1st index
    if len(item)>2:
        #we can c dat nos. r also getting printed n we don't want em so we make another rule to tackle dis
        if item[0] =="A" and "-" not in item:
            item = item.split(".")[0].replace(",","").replace(";","")
            entities.append(item)
        #now a list named entities is made (start meh)
print(entities)

#to get rid of the duplicates use(set which is a duplicate free list)
entities = list(set(entities))
print(entities)

#for making it in alphbtical order:
entities.sort()
print(entities)

