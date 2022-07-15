
# SDCND : Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking. 

In this project, you'll fuse measurements from LiDAR and camera and track vehicles over time. You will be using real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

<img src="img/img_title_1.jpeg"/>

The project consists of two major parts: 
1. **Object detection**: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. 
2. **Object tracking** : In this part, an extended Kalman filter is used to track vehicles over time, based on the lidar detections fused with camera detections. Data association and track management are implemented as well.

The following diagram contains an outline of the data flow and of the individual steps that make up the algorithm. 

<img src="img/img_title_2_new.png"/>

Also, the project code contains various tasks, which are detailed step-by-step in the code. More information on the algorithm and on the tasks can be found in the Udacity classroom. 

## Project File Structure

📦project<br>
 ┣ 📂dataset --> contains the Waymo Open Dataset sequences <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> plot functions for tracking visualization and RMSE calculation<br>
 ┃ ┣ helpers.py --> misc. helper functions, e.g. for loading / saving binary files<br>
 ┃ ┗ objdet_tools.py --> object detection functions without student tasks<br>
 ┃ ┗ params.py --> parameter file for the tracking part<br>
 ┃ <br>
 ┣ 📂results --> binary files with pre-computed intermediate results<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> data association logic for assigning measurements to tracks incl. student tasks <br>
 ┃ ┣ filter.py --> extended Kalman filter implementation incl. student tasks <br>
 ┃ ┣ measurements.py --> sensor and measurement classes for camera and lidar incl. student tasks <br>
 ┃ ┣ objdet_detect.py --> model-based object detection incl. student tasks <br>
 ┃ ┣ objdet_eval.py --> performance assessment for object detection incl. student tasks <br>
 ┃ ┣ objdet_pcl.py --> point-cloud functions, e.g. for birds-eye view incl. student tasks <br>
 ┃ ┗ trackmanagement.py --> track and track management classes incl. student tasks  <br>
 ┃ <br>
 ┣ 📂tools --> external tools<br>
 ┃ ┣ 📂objdet_models --> models for object detection<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> functions for light-weight loading of Waymo sequences<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## Installation Instructions for Running Locally
### Cloning the Project
In order to create a local copy of the project, please click on "Code" and then "Download ZIP". Alternatively, you may of-course use GitHub Desktop or Git Bash for this purpose. 

### Python
The project has been written using Python 3.7. Please make sure that your local installation is equal or above this version. 

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. You may either install them one-by-one using pip or you can use the following command to install them all at once: 
`pip3 install -r requirements.txt` 

### Waymo Open Dataset Reader
The Waymo Open Dataset Reader is a very convenient toolbox that allows you to access sequences from the Waymo Open Dataset without the need of installing all of the heavy-weight dependencies that come along with the official toolbox. The installation instructions can be found in `tools/waymo_reader/README.md`. 

### Waymo Open Dataset Files
This project makes use of three different sequences to illustrate the concepts of object detection and tracking. These are: 
- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to register with Waymo Open Dataset first: [Open Dataset – Waymo](https://waymo.com/open/terms), if you have not already, making sure to note "Udacity" as your institution.

Once you have done so, please [click here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) to access the Google Cloud Container that holds all the sequences. Once you have been cleared for access by Waymo (which might take up to 48 hours), you can download the individual sequences. 

The sequences listed above can be found in the folder "training". Please download them and put the `tfrecord`-files into the `dataset` folder of this project.


### Pre-Trained Models
The object detection methods used in this project use pre-trained models which have been provided by the original authors. They can be downloaded [here](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) (darknet) and [here](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing) (fpn_resnet). Once downloaded, please copy the model files into the paths `/tools/objdet_models/darknet/pretrained` and `/tools/objdet_models/fpn_resnet/pretrained` respectively.

### Using Pre-Computed Results

In the main file `loop_over_dataset.py`, you can choose which steps of the algorithm should be executed. If you want to call a specific function, you simply need to add the corresponding string literal to one of the following lists: 

- `exec_data` : controls the execution of steps related to sensor data. 
  - `pcl_from_rangeimage` transforms the Waymo Open Data range image into a 3D point-cloud
  - `load_image` returns the image of the front camera

- `exec_detection` : controls which steps of model-based 3D object detection are performed
  - `bev_from_pcl` transforms the point-cloud into a fixed-size birds-eye view perspective
  - `detect_objects` executes the actual detection and returns a set of objects (only vehicles) 
  - `validate_object_labels` decides which ground-truth labels should be considered (e.g. based on difficulty or visibility)
  - `measure_detection_performance` contains methods to evaluate detection performance for a single frame

In case you do not include a specific step into the list, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be loaded using [this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing) link. Please use the folder `darknet` first. Unzip the file within and put its content into the folder `results`.

- `exec_tracking` : controls the execution of the object tracking algorithm

- `exec_visualization` : controls the visualization of results
  - `show_range_image` displays two LiDAR range image channels (range and intensity)
  - `show_labels_in_image` projects ground-truth boxes into the front camera image
  - `show_objects_and_labels_in_bev` projects detected objects and label boxes into the birds-eye view
  - `show_objects_in_bev_labels_in_camera` displays a stacked view with labels inside the camera image on top and the birds-eye view with detected objects on the bottom
  - `show_tracks` displays the tracking results
  - `show_detection_performance` displays the performance evaluation based on all detected 
  - `make_tracking_movie` renders an output movie of the object tracking results

Even without solving any of the tasks, the project code can be executed. 

The final project uses pre-computed lidar detections in order for all students to have the same input data. If you use the workspace, the data is prepared there already. Otherwise, [download the pre-computed lidar detections](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB), unzip them and put them in the folder `results`.

## External Dependencies
Parts of this project are based on the following repositories: 
- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License
[License](LICENSE.md)
## Midterm Project Overview
This is Udacity Self Driving Nanodegree second course 3D Object Detection Midterm Project.
## 3D Object Detection
As in the previous course, Waymo Open Dataset will be used for this project. The Dataset contains data collected in real world and LiDAR cloud points will be used for 3D object detection.
Dataset can be found here: https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files
There are requirements to achieve these goals:
1. Converting dataset ranges channel to 8bit to visualize range and intensity image
2. Create Bird Eye View (BEV) perspective for the points cloud which then lidar intensity values and normalized height values are applied to each BEV frame
3. For 3d object detection, Complex Yolo(https://paperswithcode.com/paper/complex-yolo-real-time-3d-object-detection-on) and Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds (https://github.com/maudzung/SFA3D) will be considered in the excersise to configure and initiate the model for Vehicles detections. Then applying Bounding Box and Polygon for each detected object
4. Using pretrained data in tfrecords format files, calculate the Intersection over Union (IoU) with required minimum threshold, True/False Positive/Negative detections, Precision and Recall values.

To use the project: execute file "python loop_over_dataset.py" with python3.
Project reprequisite:
python3
numpy
opencv-python
protobuf
easydict
torch
pillow
matplotlib
wxpython
shapely
tqdm
open3d

## Step 1: Compute LiDar cloud points from Range Image
# ID_S1_EX1: 
This task requires a preview of provided range images then convert range and intensity channel to 8bit format. Then using OpenCV to stack vertically and visualize  them in a single image, cropped it to +/- 90 degree to the foward x-axis. The implementation of function is done in "objdet_pcl.py"
```
# visualize range image
def show_range_image(frame, lidar_name)
```
To test for this function, in "python loop_over_dataset.py" , config as follow:
```
exec_detection = [] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] # options are 'perform_tracking'
exec_visualization = ['show_range_image'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
```
Here is the result:
<img width="960" alt="ID_S1_EX1_cropped" src="https://user-images.githubusercontent.com/36104217/177055526-0ae86895-b676-4cd1-b848-0299a349922d.png">

# ID_S1_EX2:
Using Open3D library to visual a lidar cloud point on a 3D Model View and analyze what inside.
The implementation of function is done in "objdet_pcl.py"
```
# visualize lidar point-cloud
def show_pcl(pcl):
```
To test for this function, in "python loop_over_dataset.py" , config as follow:
```
exec_detection = [] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] # options are 'perform_tracking'
exec_visualization = ['show_pcl'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
```
Here are results:

<img width="602" alt="ID_S1_EX2" src="https://user-images.githubusercontent.com/36104217/177055725-96ea9ffc-76f9-4093-abbd-8d8a923da10e.png">
<img width="960" alt="ID_S1_EX2_7" src="https://user-images.githubusercontent.com/36104217/177055732-db4ef017-9369-413f-8ac0-edc9bcf2064e.png">
<img width="960" alt="ID_S1_EX2_8" src="https://user-images.githubusercontent.com/36104217/177055662-3a6c139a-7f1b-481b-a200-afa80c1ce00d.png">
<img width="960" alt="ID_S1_EX2_9" src="https://user-images.githubusercontent.com/36104217/177055663-0fb19e75-0750-42f8-acfd-a57816716822.png">
<img width="603" alt="ID_S1_EX2_2" src="https://user-images.githubusercontent.com/36104217/177055664-95745ded-8414-4f54-a7bd-29f245720e68.png">
<img width="602" alt="ID_S1_EX2_3" src="https://user-images.githubusercontent.com/36104217/177055665-a63914a5-353e-4d7e-a362-4af4e1d56ddc.png">

A very close by vehicle can be observed below because of large reflection points:

<img width="606" alt="ID_S1_EX2_4" src="https://user-images.githubusercontent.com/36104217/178330307-c956697d-65cd-43a9-952e-b5ba94909c65.png">

A long vehicle, probably truck carrying a trailer:

<img width="604" alt="ID_S1_EX2_5" src="https://user-images.githubusercontent.com/36104217/178330314-d98c4b47-ad91-41db-8b6f-c13314f4f519.png">

In the next frame, the mentioned above truck is now ahead, while in the other side of the road, multiple vehicles are moving in opposite direction are clearly visible with Windshields and side mirrors can also be detected via Intensity channel:

<img width="923" alt="ID_S1_EX2_6" src="https://user-images.githubusercontent.com/36104217/178330316-e1507f62-4e82-4ffe-907f-ac2867ba8047.png">


From the results, objects like cars, traffic lights, trees  or buildings can be found. These objects are detected over intensity channel, so most of the time cars on the road are clearly be seen with rear and front directions. This will be very helpful in later task when applying object detection into these images.

## Step-2: Creaate BEV from Lidar PCL
In this step requires:
1. Converting the coordinates to pixel values
2. Assigning lidar intensity values to BEV mapping
3. Using sorted point cloud lidar from the previous task
4. Normalizing the height map in the BEV
5. Compute and map the intensity values
The implementation of function is done in "objdet_pcl.py"
```
# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):
```
To test for this function, in "python loop_over_dataset.py" , config as follow:
```
exec_detection = ['bev_from_pcl'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] # options are 'perform_tracking'
exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
```
Here are results:
Lidar Cloud Point

<img width="380" alt="ID_S2_EX1_1" src="https://user-images.githubusercontent.com/36104217/177056155-537d2d51-0061-4070-8c7a-c81ca1a39253.png">
<img width="333" alt="ID_S2_EX1_2" src="https://user-images.githubusercontent.com/36104217/177056156-8f962703-bb44-4bed-b257-6c363d413a6f.png">

Intensity Channel

<img width="455" alt="ID_S2_EX2_1" src="https://user-images.githubusercontent.com/36104217/177056195-1b4c1bfc-d1cc-4e5c-9569-2fbeb847c82c.png">
<img width="458" alt="ID_S2_EX2_2" src="https://user-images.githubusercontent.com/36104217/177056197-d8265ab3-1e1e-4fd4-a958-d50a7724e4e2.png">

Normalized height Channel

<img width="458" alt="ID_S2_EX3_2" src="https://user-images.githubusercontent.com/36104217/177056246-5ec4b24e-0323-457f-ad64-53bcc5ea83e0.png">
<img width="458" alt="ID_S2_EX3_1" src="https://user-images.githubusercontent.com/36104217/177056247-7f1443f1-0dc7-4497-9ed2-7cf72af54692.png">

## Step-3: Model Based Object Detection in BEV Image
In this step, SFA3D model will be configured and initialized with a pretrained data to perform vehicle detections and apply bounding boxes and 3D polygons. Futher exploration to use this model can be found here: https://github.com/maudzung/SFA3D
1. Instantiating the FPN resnet model from the cloned repository configs
2. Extracting 3d bounding boxes from the responses
3. Transforming the pixel to vehicle coordinates
4. Model output tuned to the bounding box format [class-id, x, y, z, h, w, l, yaw]
The implementation of function is done in "objdet_detect.py"
```
# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None)

# create model according to selected model type
def create_model(configs)

# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs)

```
To test for this function, in "python loop_over_dataset.py" , config as follow:
```
exec_detection = ['bev_from_pcl', 'detect_objects'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] # options are 'perform_tracking'
exec_visualization = ['show_objects_in_bev_labels_in_camera'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
```
Here are results:

<img width="419" alt="ID_S3_EX1-5_1" src="https://user-images.githubusercontent.com/36104217/178267764-68e378c0-41a1-47f0-b0a4-01a4a156aa73.png">
<img width="457" alt="ID_S3_EX1-5_2" src="https://user-images.githubusercontent.com/36104217/178267767-db91140c-1db3-4df0-9b68-f31001099bb7.png">
<img width="458" alt="ID_S3_EX1-5_3" src="https://user-images.githubusercontent.com/36104217/178267756-2b4f34f5-5b89-453c-b2c9-493dba5b73bb.png">

The detected objects will be provided with coordinates and properties in the three-channel BEV coordinate space since the model input is a three-channel BEV map. Therefore, the detections must be converted into metric coordinates in vehicle space before they can proceed in the processing pipeline.

<img width="721" alt="ID_S3_EX1-5_4" src="https://user-images.githubusercontent.com/36104217/178267810-efa6d623-b443-4d50-8e7c-cc8c7feea806.png">
<img width="457" alt="ID_S3_EX1-5_5" src="https://user-images.githubusercontent.com/36104217/178267815-777e14a8-f374-4660-9ee6-364f7e7a889c.png">

## Step-4: Performance detection for 3D Object Detection
In this excersice, the IOUs of labels and detections are calculated to get the false positive and false negative values, then the performance is computed. The task is to compute the geometric overlap between the bounding boxes of labels and the detected objects as follow:

1. If calculated IOU exceeds minimum threshold, add the detection with label
2. Computing the degree of geometric overlap
3. Detections of multiple objects that greater than set threshold
4. Computing the false negative and false positive values
5. Computing precision and recall with above the false positive and false negative values
In "objdet_eval.py" where the precision and recall are calculated as functions of false positives and negatives:
```
    # compute positives and negatives for precision/recall
    ## step 1 : compute the total number of positives present in the scene
    all_positives = labels_valid.sum()
    
    ## step 2 : compute the number of false negatives
    true_positives = len(iou)
    false_negatives = all_positives- true_positives

    ## step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives
```
precision = 0.9979381443298969, recall = 0.6453333333333333 

With calculated values can be plotted as these graphs below:

<img width="497" alt="ID_S4_EX1_1" src="https://user-images.githubusercontent.com/36104217/178269123-e12bbe7d-0f47-4fd0-bf90-6d175ee636cf.png">

By setting "configs_det.use_labels_as_objects=True", results in precision and recall values as 1.This is shown in the following image:

<img width="476" alt="ID_S4_EX1_2" src="https://user-images.githubusercontent.com/36104217/178269288-95d7d502-0fe1-490a-bc08-8fe2ff8ff62c.png">

### Summarization of Lidar based 3D Object Detection

This project gives a basic idea of using Lidar data and visualizing in 3D plane for object detection. For further analysis, it is crucial to convert range data to a point cloud using spatial volumes, points, or CNN networks. For 3D object detection, the use of resnet/darknet and YOLO to convert these high dimensional point cloud representations into object detections through bounding boxes is crucial. Understanding the effectiveness of Lidar based detection requires evaluating the performance using maximal IOU mapping, mAP, and representing the precision/recall of the bounding boxes.

## Final Project Overview
This is Udacity Self Driving Nanodegree second course 3D Object Detection Final Project.

## Sensor Fusion and Object detection

Using Waymo's real-world data and applied an extended Kalman fusion filter for tracking several vehicles in this project. The following are the tasks completed:
- Building Kalman Filter system to track an object
- Object tracking and updating tracks (creating and deleting)
- Understanding the association between the data (sensor)
- Added camera sensor fusion based on lidar fusion 

Dataset can be found here: https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files
To use the project: execute file "python loop_over_dataset.py" with python3.
Project reprequisite:
python3
numpy
opencv-python
protobuf
easydict
torch
pillow
matplotlib
wxpython
shapely
tqdm
open3d

## Step-1: Extended Kalman Filter

The first step involves implementing Extended Kalman update for a single track. By setting up the required matrices and implementation EKF equation in the predict fucntion in filter.py file.
+ Design the system states [x, y, z, vx, vy, vz], process model, and constant velocity model.
+ Calculate the matrix (system matrix) for the 3D process models with constant velocity and noise covariances. This is required for computing state h(x) and Jacobian H.
+ For current state calculation, h(x), and the Jacobian H function are evaluated.
+ The Kalman gain is computed and is used for updating the state and covariance.
The images below show the analysis of RMSE with current time for single track:

<img width="809" alt="Final_ID_S1_1" src="https://user-images.githubusercontent.com/36104217/179194375-d583e39a-aaff-4e81-928e-5cfcb197b535.png">
<img width="916" alt="Final_ID_S1_2" src="https://user-images.githubusercontent.com/36104217/179194379-2b97237a-dc1b-47d2-9725-abd675233cda.png">

## Step-2: Track Management

In this second step, implemntation of track management for a single track then the measurement is associated with the track. If the measurement is consistent then it will become a confirmed track. Otherwise, it's considered as tentative and quickly get deleted. If the score is below certain three-point and the state balance is greater than a threshold , then the track is not removed for further consideration. 
The results show a slight reductiong of RSME between ground truth and tracked vehicle:

<img width="904" alt="Final_ID_S2_1" src="https://user-images.githubusercontent.com/36104217/179195274-94869840-e832-4b53-ba07-5ae2e8d81849.png">
<img width="885" alt="Final_ID_S2_2" src="https://user-images.githubusercontent.com/36104217/179195277-42b66801-8db3-49ec-a985-c3075eb9bb60.png">

## Step-3: Data Association

In this step, the closest neighbor association correctly matches several measurements to several tracks. In association.py, data association is introduced. Tracking is extended to multiple tracks which are aligned to each respective measurement according to Mahalanobis distance between the tracks and the measurements. Later gating is applied to filter out measurement association to the track.

<img width="905" alt="Final_ID_S3_2" src="https://user-images.githubusercontent.com/36104217/179195979-50d69a40-f2c4-4fcc-98d7-ae1f55e48e1c.png">
<img width="910" alt="Final_ID_S3_4" src="https://user-images.githubusercontent.com/36104217/179195983-ed95c870-6725-45e7-9304-86275cb3f9cf.png">

This RSME is taken for later comparision with applying sensor fusion between lidar and camera in step 4 
<img width="909" alt="Final_ID_S3_1" src="https://user-images.githubusercontent.com/36104217/179195984-cd02d00f-e258-4e91-89be-661ecd7e7bc7.png">

## Step-4: Camera Sensor fusion

Finally, camera measurement is added to the measurement update step. This helps improving the tracking accuracy. The first order of filtering of track association is by checking the current tracked object is visible to the camera. Then it will be passed along witht the lidar measurement for updating the track.

The implementation consists of projection matrix which converts the points from 3d space into 2d geometry. The partial derivatives (x,y,z) are used for measuring the model in parameters (u,v). The noise is also measured (R).If the tracking status is in FOV(Field of View) then the measurement-track pair can be accepted else it will be rejected.

RSME of tracking with Camera mesurement:

<img width="905" alt="Final_ID_S4_1" src="https://user-images.githubusercontent.com/36104217/179197140-b235a996-9cb0-4784-9050-59ca992a0e3a.png">

## Benefits in Camera-Lidar Fusion tracking over Lidar-only tracking

Theoretically, adding more sensors will result in a better accuracy due to the reduction in uncertainty with each measurement update. As can be seen in these above results, there is no significant improvements when comparing between performance of Lidar Only system and Camera - Lidar Sensor Fusion system.

## Difficulties

All the steps in this project are well descripted by the lecturers and student can get the ideas of what to be done.
However, it's quite challenge for implementing camera measurement, when projecting a 3D point to 2D point and transformation in the camera axis. This requires a thorough understanding of the concepts mentioned in previous courses.

In the provided loop_over_dataset.py, there are some outdated functions leading to some bugs as: requires different number of arguments or missing variable defintitions, etc. But through reading command line output from Python shell, it's can easily be debugged and fixed.

## Real-life challenges

Both camera and lidar sensor depend on the environment for measurement. As observed during the project, the scenario is quite bright, clear and very less noises due by foggy or rainy conditions.
All of the sensors reference posistions might be adjusted to have a best accurate calculation, since vehicle might have misalignment or deformation leading to unexpected sensors positions.

## Future Improvement

There could be a possibility to combine multiple cameras into a unified system that could provide a depth dimension into its original 2D data. This will eliminate the need of projection from 3D to 2D of Lidar data, hence improve the optimization of calculation performance.
Or with Vehicle to X (V2X) system, multiples vehicles in the same region can communicate and tranfer sensor data between which can drastically improve the data accuracy. Futhermore, tracking performance can be push to the limit due to real time position of each vehicle is updated and broadcasted to others.





