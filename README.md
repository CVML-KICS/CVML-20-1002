# Yolo-v3 based Tower Surveillance
### (neural network for object detection)

More details: http://pjreddie.com/darknet/yolo/



#### Pre-trained models

There are weights-file for different cfg-files (trained for local dataset):
* weights https://drive.google.com/file/d/1C-MEm-owdoVbgrrloXjANPPse8Km706h/view?usp=sharing
* place these weights in the root folder. 

#### Installation
Following are the instructions to install project. 
**cd ~/TowerSurveillance** 
**make**
**make install**

#### Test model

To test the model on single image run the command (set the varibale **imagePath** to input image name) 

**python3 test.py**

#### Train model

1. Download pre-trained weights for the convolutional layers (154 MB): http://pjreddie.com/media/files/darknet53.conv.74 and put to the directory `build\darknet\x64`

2. Prepare dataset on PASCAL VOC standard

3. Create file `obj.names` in the directory `ROOT`, with objects names - each in new line

4. Create file `obj.data` in the directory `ROOT`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = train.txt
  valid  = test.txt
  names =  obj.names
  backup = backup/
  ```

5. Put image-files (.jpg) of your objects in the directory `build\darknet\x64\data\obj\`













 

 
    



