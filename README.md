# Suspicious Object Detection (SOD)
This project detects suspicious objects (Guns, Knives, bags etc.) in the context of Tower Surveillance.

## Training YOLO on custom dataset
You can train YOLO from scratch if you want to play with different training regimes, hyper-parameters, or datasets. Here's how to get it working on the COCO dataset.

## Download darknet
Run the following commands in your choice of directory to download the darknet repository and install it.

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

## Modify files for SOD
* Now go to your Darknet directory. 
* Clone this repository in the Darknet directory. We have to add our own config files to point to your data. 
* Modify the train.txt and test.txt, according to your own requirement otherwise, just leave it as it is and copy the dataset as mentioned in the next section. 

## Download the Dataset
Download the dataset from the following link into the darknet directory. 
https://drive.google.com/file/d/1a2fWctBJWRB277pcGV1UBVtVv2n7Zpyb/view?usp=sharing

## Running the code
* Download the weights for suspicious object detection into the darknet directory.
https://drive.google.com/file/d/1VNk3oZ0qS_bfCAgrqip1cfa0G853JXYW/view?usp=sharing
* Run the code for testing any video/image through the following command:

```./darknet detector test yolo/obj.data yolo/yolov3-custom.cfg *.weights *.jpg```
