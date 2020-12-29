import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
               help="Path to input image")
ap.add_argument("-p","--prototxt",required=True,
               help="Path to caffe deploy file")
ap.add_argument("-m","--model",required=True,
               help="Path to Pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.5,
               help="minimum probablity for filters")
args = vars(ap.parse_args())



#Loading the model
print("Loading model")
net = cv2.dnn.readNetFromCaffe(args['prototxt'],args['model'])

#loading image and resizing
image = cv2.imread(args['image'])
(h,w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,
                            (300,300),(104.0,177.0,123.0))




                            #detecting face 

print("Computing face detections")

net.setInput(blob)

detections = net.forward()




#loop over all detections

for i in range(0,detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the predictions
    confidence = detections[0,0,i,2]
    
    # filter out weak detections by ensuring the `confidence` is greater than the minimum one
    if confidence>args['confidence']:
        
        #compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")
        # draw the bounding box of the face along with the associated probablity
        text = "{:.2f}%".format(confidence*100)
        y = startY-10 if startY-10>10 else startY+10
        cv2.rectangle(image,(startX,startY),(endX,endY),
                     (0,0,255),2)
        cv2.putText(image,text,(startX,y),
                   cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
        
cv2.imshow("Output",image)
cv2.waitKey(0)
        
        
