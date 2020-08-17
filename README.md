# Yolo-v3 based Tower Surveillance
### (neural network for object detection)

More details: http://pjreddie.com/darknet/yolo/



#### Pre-trained models

There are weights-file for different cfg-files (trained for local dataset):
* weights https://drive.google.com/file/d/1C-MEm-owdoVbgrrloXjANPPse8Km706h/view?usp=sharing
* place these weights in the root folder. 

#### Installation
Following are the instructions to install project. 
`cd ~/TowerSurveillance`
Open Makefile
Set values of GPU and CUDNN to 1 or 0, 1 for using GPU and 0 for using CPU
Set OPENCV = 1 if you want to compile with opencv 
```
make
make install
```

#### Run model

To run the model on single image run the command (set the varibale **imagePath** to input image name) 

`python3 test.py`

#### Run executable


#### Train model for Tower Surveillance


1. Dwonload Tower Surveillance dataset from https://drive.google.com/file/d/1X6plbvkq3tBAqxC31h15lhi_POoGQvK6/view?usp=sharing
2. Place images and corresponding annotations in folder `dataset/train` and `dataset/test`. 
3. Create `train.txt` by using following code. 
```
import glob
imgs_path = glob.glob("path/to/train/folder/*.JPG")
file = open("train.txt", "w")
for i in imgs_path:
  file.write(i + "/n")
file.close()
```
4. Create `test.txt` by using following code. 
```
import glob
imgs_path = glob.glob("path/to/test/folder/*.JPG")
file = open("test.txt", "w")
for i in imgs_path:
  file.write(i + "/n")
file.close()
```
5. Modify model meffa.cfg for training instead of testing and set hyperparameters batch size, input size etc
run 
`python3 train.py`


If you face any troubles in compiling files, configure Makefile according to your system configuration or just google the error.



