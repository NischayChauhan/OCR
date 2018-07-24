
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
# In[2]:


im_path = './sample.jpg'
img = cv2.imread(im_path)
print img.shape
img = cv2.resize(img, (1500, 800))
print img.shape
plt.imshow(img)
plt.show()


# ## Image Blurring
# ## Edge Detection
# ## Contour Extraction
# ## Best contour selection
# ## Projection to fixed size screen

# In[3]:


## Image Blurring
orig = img.copy()
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()
blurred = cv2.GaussianBlur(gray, (5,5), 0)
plt.imshow(blurred)
plt.show()
print blurred.shape


# In[4]:


regen = cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)
plt.imshow(orig)
plt.show()
plt.imshow(regen)
plt.show()


# In[5]:


## Edge Detection
edged = cv2.Canny(blurred, 0, 50)
orig_edged = edged.copy()

plt.imshow(orig_edged)
plt.show()


# In[6]:


## Contours Extraction
_, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print len(contours)
contours = sorted(contours, reverse=True, key=cv2.contourArea)


# In[7]:


## Best Contour Selection
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)
    
    if len(approx) == 4:
        target = approx
        break

print target.shape


# In[8]:


## Reorder target contour
def reorder(h):
    h = h.reshape((4,2))
    print h
    
    hnew = np.zeros((4,2), dtype=np.float32)
    
    add = h.sum(axis=1)
    hnew[3] = h[np.argmin(add)]
    hnew[1] = h[np.argmax(add)]
    
    diff = np.diff(h, axis=1)
    hnew[0] = h[np.argmin(diff)]
    hnew[2] = h[np.argmax(diff)]
    
    return hnew


# In[9]:


reorderd = reorder(target)
print '----------'
print reorderd


# In[10]:


## Project to a fixed size screen
input_represent = reorderd
output_map = np.float32([[0,0],[800,0],[800,800],[0,800]])


# In[11]:


M = cv2.getPerspectiveTransform(input_represent, output_map)
ans = cv2.warpPerspective(orig, M, (800,800))


# In[12]:


plt.imshow(ans)
plt.show()


# In[13]:


ans2 = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
temp = cv2.GaussianBlur(ans2, (3,3), 0)
temp2 = cv2.GaussianBlur(ans2, (5,5), 0)
temp3 = cv2.GaussianBlur(ans2, (1,1), 0)
plt.imshow(temp)
plt.show()

plt.imshow(temp2)
plt.show()

plt.imshow(temp3)
plt.show()


# In[14]:


now = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
plt.imshow(now)
plt.show()

