import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
#imports




def sketch(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    canny_edges = cv2.Canny(img_gray_blur,10,70)
    ret , mask = cv2.threshold(canny_edges, 70,255 , cv2.THRESH_BINARY_INV)
    return mask
#function declaration,defintion




cap = cv2.VideoCapture(0)



while True:
    ret , frame = cap.read()
    cv2.imshow("Live sketch",sketch(frame) )
    if cv2.waitKey(1) == 27:   #27 is esc
        break
cv2.destroyAllWindows()
