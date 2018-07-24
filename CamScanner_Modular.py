
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


# In[3]:


## Reorder target contour
def reorder(h):
    h = h.reshape((4,2))

    
    hnew = np.zeros((4,2), dtype=np.float32)
    
    add = h.sum(axis=1)
    hnew[3] = h[np.argmin(add)]
    hnew[1] = h[np.argmax(add)]
    
    diff = np.diff(h, axis=1)
    hnew[0] = h[np.argmin(diff)]
    hnew[2] = h[np.argmax(diff)]
    
    return hnew


# In[4]:


#img_path is path of the image needed to 
def convert_img_text(img_path):
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.resize(img, (1500, 800))
    
    ## Image Blurring
    orig = img.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    ## Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()
    
    
    ## Contours Extraction
    _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, reverse=True, key=cv2.contourArea)
    
    ## Best Contour Selection
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*p, True)
    
        if len(approx) == 4:
            target = approx
            break
            
    #reordering points        
    reorderd = reorder(target)
    
    ## Project to a fixed size screen
    input_represent = reorderd
    output_map = np.float32([[0,0],[800,0],[800,800],[0,800]])
    
    M = cv2.getPerspectiveTransform(input_represent, output_map)
    ans = cv2.warpPerspective(orig, M, (800,800))
    
    
    #blurring again to increase readability
    ans2 = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
    temp = cv2.GaussianBlur(ans2, (1,1), 0)
    now = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    
    #converting into text
    img=Image.fromarray(now)
    img.save("refined_receipt.jpg")
    text = pytesseract.image_to_string(Image.open("refined_receipt.jpg"))
    print(text)
    


# In[6]:


convert_img_text('./sample.jpg')

