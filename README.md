# CVML-1002-20

# Tower Surveillance:

The repo is about Automatic Tower Surveillance. Person and rack open/closed classes are targeted to monitor the security of any tower premises. 

## Introduction:

The aim of this project is to watch equipment and detect the unauthorized person in telecommunication tower area, and alarm the authority for timely rescue.

## Dataset

Custom dataset was used to detect the un authorized people and equipment in tower area. For Tower surveillance dataset was collected from Different CCTV cameras that was installed at various locations and angles at tower. The collected data was basically based on the recorded video dataset of tower. The details of the used dataset are following:
-	CCTV camera recorded videos from multiple locations
-	Video Length: 35-40 minutes
-	Video Quantity: 3-4 clips

## Preprocessing

The preprocessing steps of the proposed project are following:
-	Extract Image Frames from Videos
-	Annotate the Extracted Image Frame
-	Annotation Criteria
-	Box open (If telecom cabinet, betray bank, generator’s door opened)
-	Box closed (If telecom cabinet, betray bank, generator’s door closed)
-	Person authorized (If authorized team worker enters in the premises)
-	Person unauthorized (If unauthorized team worker enters in the premises)

## Model Training 

For the detection of equipment and unauthorized persons in the tower area, Yolo V3 model was trained with the annotated images. The details of the model training are following:
-	Used 3000 Annotated Images

## Results

-	Used 300 Annotated Samples for Evaluation
-	Calculate Mean Average Precision (MAP)
-	Got 0.7% value of MAP for validation samples

