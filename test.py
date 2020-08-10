from darknet import loadNets, TestImage
import cv2 as cv
import numpy as np

imagePath ="data/1.jpg"       
image = cv.imread(imagePath)
net, meta = loadNets()
detections = TestImage(net,meta, image, 0.25)
imcaption = []
for detection in detections:
                label = detection[0]
                confidence = detection[1]
                print(label)
                pstring = label.decode("utf-8") +": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                image = cv.rectangle(image, (xCoord, yCoord) ,(xCoord + xEntent, yCoord + yExtent), (255, 0, 0) , 2)
                image = cv.putText(image, pstring,(xCoord, yCoord),cv.FONT_HERSHEY_SIMPLEX,1 ,(255, 0, 0) , 2)

            
cv.imshow("result",image)
cv.waitKey(0)
detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
