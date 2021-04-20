import numpy as np
import cv2
from matplotlib import pyplot as plt

def blend_img_img(img, effect, rect, ALPHA):
    effect = cv2.cvtColor(effect, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    mask2 = mask2[:,:,np.newaxis]

    bld_img = ((1 - ALPHA)*img + (ALPHA*effect*mask2)).astype(np.uint8)

    plt.subplot(131).set_title('Original')
    plt.imshow(img)
    plt.subplot(132).set_title('Effect')
    plt.imshow(effect)
    plt.subplot(133).set_title('Blended Image')
    plt.imshow(bld_img)
    plt.show()

def blend_img_vid(img, effect, rect, ALPHA):
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    mask2 = mask2[:,:,np.newaxis]

    while True: 
        ret, frame = effect.read()
        if not ret:
            break
        frame = cv2.resize(frame, (367, 550))
        frame = ((1 - ALPHA)*img + (ALPHA*frame*mask2)).astype(np.uint8)
        cv2.imshow('frame', frame)
        if(cv2.waitKey(10) & 0xFF == ord('b')):
                break

    effect.release()
    cv2.destroyAllWindows()

img = cv2.imread(r'C:\UIT\HK6\HW-20210420T132435Z-001\HW\HW5\girl.jpg')
effect = cv2.imread(r'C:\UIT\HK6\HW-20210420T132435Z-001\HW\HW5\fire.jpg')
vid = cv2.VideoCapture(r'C:\UIT\HK6\HW-20210420T132435Z-001\HW\HW5\smoke.avi')
alpha = -0.5
# rect = (start_x, start_y, width, height)
rect = (0,5,366,545)

# blend_img_img(img, effect, rect, alpha)
blend_img_vid(img, vid, rect, alpha)