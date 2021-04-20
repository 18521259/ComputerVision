import numpy as np
import cv2
from matplotlib import pyplot as plt

#============BLENDING 2 IMAGES=============================================================
img = cv2.imread(r'C:\UIT\HK5\introCV_CS231\HW5\girl.jpg')
effect = cv2.imread(r'C:\UIT\HK5\introCV_CS231\HW5\fire.jpg')
effect = cv2.cvtColor(effect, cv2.COLOR_BGR2RGB)
ALPHA = 0.5

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# rect = (start_x, start_y, width, height)
rect = (0,5,366,545)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
mask2 = mask2[:,:,np.newaxis]

bld_img = ((1 - ALPHA)*img + (ALPHA*effect*mask2)).astype(np.uint8)

# maxValue = np.amax(bld_img)
# minValue = np.amin(bld_img)
# print(minValue)
# print(maxValue)
# bld_img = np.clip(bld_img, 0, 1)

plt.subplot(131).set_title('Original')
plt.imshow(img)
plt.subplot(132).set_title('Effect')
plt.imshow(effect)
plt.subplot(133).set_title('Blended Image')
plt.imshow(bld_img)
plt.show()

#============BLENDING AN IMAGE WITH A VID=============================================================
# img = cv2.imread(r'C:\UIT\HK5\introCV_CS231\HW5\girl.jpg')
# ALPHA = 0.5
# mask = np.zeros(img.shape[:2],np.uint8)

# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)

# # rect = (start_x, start_y, width, height)
# rect = (0,5,366,545)

# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
# mask2 = mask2[:,:,np.newaxis]

# cap = cv2.VideoCapture(r'C:\UIT\HK5\introCV_CS231\HW5\smoke.avi')

# while True: 
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (367, 550))
#     frame = ((1 - ALPHA)*img + (ALPHA*frame*mask2)).astype(np.uint8)
#     cv2.imshow('frame', frame)
#     if(cv2.waitKey(10) & 0xFF == ord('b')):
#             break

# cap.release()
# cv2.destroyAllWindows()