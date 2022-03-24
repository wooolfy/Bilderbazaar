
import cv2
import numpy
from pymatting import cutout
from PIL import Image
from os import listdir
from os.path import isfile, join

mypath="./testpdf"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in range (len(onlyfiles)):
    try:
        picture = cv2.imread("./testpdf/"+onlyfiles[i])
        mask = cv2.imread("./outputmasks/mask_"+onlyfiles[i])
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print('processing image: '+ onlyfiles[i])  
        masked = cv2.bitwise_and(picture, picture, mask = gray_mask)
    
        # cv2.imshow("Detected Edges", masked)
        # cv2.waitKey(0)
        cv2.imwrite("./cutout/"+onlyfiles[i], masked)
    except Exception:
        print('No data. Either no file or mask was empty')
        continue

