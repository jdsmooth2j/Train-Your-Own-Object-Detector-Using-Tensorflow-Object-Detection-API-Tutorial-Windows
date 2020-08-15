# Train-Your-Customized-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows
Train Your Own Object Detection Classifier Using TensorFlow on Windows
Credits goes out to *Edje Electronics* and *Dat Tran* for their work relating to this tutorial.

## Quick Summary

This repository explores the step by step process for training your own object detection classifier using TensorFlow's object detection API to detect multiple objects on Windows 10, 8, or 7. (As per Edje Electronics, it will also work on Linux-based OSes with some minor changes. Haven't tried it yet.) The tutorial was originally written using *TensorFlow version 1.5.0*. (This will also work for newer versions of TensorFlow.)

Try to watch the excellent YouTube video tutorial made by Edje Electronics on the link below which walks through every step of the way of this tutorial.
"How To Train an Object Detection Classifier Using TensorFlow (GPU) on Windows 10": http://www.youtube.com/watch?v=Rgpfk6eYxJA 

This readme exlpores every step required to get you through with your customized object detection classifier:
```
1. Installing Anaconda, CUDA, and cuDNN
2. Setting up the Object Detector Directory Structure and Anaconda Virtual Environment
3. Gathering and Labeling Images
4. Generating Training data
5. Creating a Label Map and Configuring Training
6. Training
7. Exporting the Inference Graph
8. Testing and Using Your Newly Trained Object Detection Classifier
```
The repository have all the files needed to train an "insect detector" that can accurately detect *whiteflies* and *eggplant fruit and shoot borer (EFSB)*. You can replace these files with your own files to train an object detection classifier for whatever concept you have in mind. This tutorial also have python scripts to test your own object detector out on an *image*, on a *video*, or on a *webcam* feed.

![Insect Detector Tested on Image File](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/test_output_05_threshold%3D0.6.png)

![Insect Detector Tested on Video](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=-c-vxMn5tns)

## Introduction

The tutorial's sole purpose is to describe how to train your customized convolutional neural network object detectior for multiple objects, starting *from scratch*. After this tutorial, you will have your own object detection program that can identify and draw boxes around specific objects in *images*, *videos*, or in your *webcam feed*.

There are lots of tutorials available over the Internet on how to use TensorFlow’s Object Detection API to train a customized object classifier for multipple objects, Edje Electronics is the best one I have seen. This tutorial is written on *Windows 8.1*, and it should also work for *Windows 7, 8 and 10*. The version of tensorflow I used on this tutorial was *TensorFlow v1.5.0* (it will likely work for future versions of TensorFlow)

*TensorFlow-GPU* versions allows your PC to use the 'graphics/video card' to "provide extra processing power" while performing training, it is recommended to install tensorflow-GPU version for this purpose. Using TensorFlow-GPU instead of a regular TensorFlow "reduces the training time" significantly, as a matter of fact based on Edje Electronics's experience, he documented a *reduced traning time by a factor of about 8* (3 hours to train instead of 24 hours). The CPU-only version of TensorFlow can also be used for this tutorial, but it will take much much longer. As a matter of fact, it took me about 15 hours to train my Insect Detector in 5,000 steps using the TensorFlow-CPU version and using the Faster R-CNN Inception v2 COCO model. If you choose to use the TensorFlow-CPU version, you do not need to install CUDA and cuDNN in Step 1.

## Procedure
### 1. Installing Anaconda, CUDA, and cuDNN (if using GPU)
I recommend Mark Jay's YouTube video (https://www.youtube.com/watch?v=RplXYjxgZbw) which shows the steps in installing Anaconda, CUDA, and cuDNN. The video is made for *TensorFlow-GPU v1.4*, so download and install the CUDA and cuDNN versions for the latest TensorFlow version you have, rather than *CUDA v8.0* and *cuDNN v6.0* as instructed in the video. The websites below shows which versions of CUDA and cuDNN are needed for the latest version of TensorFlow.
* https://www.tensorflow.org/install/source#tested_build_configurations
* https://www.tensorflow.org/install/gpu
* https://www.tensorflow.org/install/
* https://developer.nvidia.com/cuda-toolkit-archive
* https://developer.nvidia.com/rdp/cudnn-archive

If you have an older version of TensorFlow installed, ensure you use the CUDA and cuDNN versions that are compatible with the TensorFlow version installed on your computer. Below is a table showing which version of TensorFlow requires which versions of CUDA and cuDNN.
https://www.tensorflow.org/install/source#tested_build_configurations

Make sure you have installed Anaconda as described by Mark Jay's YouTube video, because we will utilize the anaconda virtual environment for the rest of this tutorial. (The anaconda environment I've used in this tutorial have **Python 3.6** installed)

Go ahead, explore TensorFlow's website for further installation details, including how to install it on other operating systems (such as *Linux*). Also, the object detection repository itself has the installation instructions.

### 2. Organize/Set up the TensorFlow Directory and Anaconda Virtual Environment
The TensorFlow Object Detection API requires using the specific directory structure provided on its GitHub repository. It also requires several additional Python packages, specific additions to the *PATH* and *PYTHONPATH* variables, and a few extra setup commands to get everything set up to run or train an object detection model.

Follow the instructions closely because it is fairly meticulous and an improper setup can cause cumbersome errors as you go along.

#### 2A. Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in **C:** and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

Get the full TensorFlow object detection API repository at https://github.com/tensorflow/models and download by clicking the “*Clone* or *Download*” button and then downloading the zip file. Extract the downloaded zip file “models-master” folder directly into the "C:\tensorflow1" directory you have just made and rename “models-master” to just “models”.

Note: The TensorFlow models repository's code (which contains the object detection API) is continuously updated by the developers. Sometimes they make changes that break functionality with old versions of TensorFlow. It is always best to use the latest version of TensorFlow and download the latest models repository. If you are not using the latest version, clone or download the commit for the version you are using as listed in the table below.

If you are using an older version of TensorFlow, here is a table showing which GitHub commit of the repository you should use. You can look for this by going to the release branches for the models repository and getting the commit before the last commit for the branch. (They remove the research folder as the last commit before they create the official version release.)

**TensorFlow version**  |  **GitHub Models Repository Commit**
------------------      |  -------------------------------   
TF v1.7	                |  https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f
TF v1.8	                |  https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc
TF v1.9	                |  https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b
TF v1.10	            |  https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df
TF v1.11	            |  https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43
TF v1.12	            |  https://github.com/tensorflow/models/tree/r1.12.0
TF v1.13	            |  https://github.com/tensorflow/models/tree/r1.13.0
Latest version	        |  https://github.com/tensorflow/models

Note:
This tutorial was based using **TensorFlow v1.5** and this GitHub commit of the TensorFlow Object Detection API. If portions of this tutorial do not work, it may be necessary to install TensorFlow v1.5 and use this exact commit rather than the most up-to-date version.

#### 2B. Download the Faster R-CNN Inception V2 Model from TensorFlow's Model Zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) on its model zoo. I tried the **SSD-MobileNet COCO v2 model** on my first try. This model allows faster detection but have less accuracy, while on my second try I used the **Faster-RCNN Inception v2 model** and it gives "slower detection" but have a "significant increase in accuracy" (*more accurate* than SSD Mobilenet COCO v2 model by far but with a noticeably *slower speed*).

![Test Results: SSD Mobilenet v2 COCO Model vs Faster R-CNN Inception v2 Model](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/SSD_v2_VS_FasterRCNN_Inception_v2.png?raw=true)

You can choose whichever model you want to train your own objection detection classifier. If you are planning to deploy the object detector on a device with 'low computational power' (such as Raspberry Pi and a decent laptop/PC), use the SDD-MobileNet model. Otherwise use a 'higher-end' laptop or desktop PC if you plan to deploy it using the Faster R-CNN models.

This tutorial utilizes the Faster-RCNN-Inception-V2 model. Download the model here:
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Open the just downloaded *faster_rcnn_inception_v2_coco_2018_01_28.tar.gz* file with a file archiver and extractor such as *WinRAR*, *WinZip* or *7-Zip* and extract the 'faster_rcnn_inception_v2_coco_2018_01_28' folder to the "C:\tensorflow1\models\research\object_detection" folder. 
Note: The model date and version will likely change in the future, but it should still work with this tutorial.

#### 2C. Download this tutorial's repository from my GitHub
Download the full repository located on this page (scroll to the top and click 'Clone' or 'Download') and extract all the contents directly into the "C:\tensorflow1\models\research\object_detection" directory. (You can overwrite the existing *README.md* file.) This establishes a specific directory structure that will be used for the rest of the tutorial.
Note: put *frozen_inference_graph.pb* file to inference_graph folder in the "C:\tensorflow1\models\research\object_detection" directory afer downloading.

Here is what "C:\tensorflow1\models\research\object_detection" folder should look like:

![/object_detection Folder Snippet](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/directory_object_detection.png?raw=true)

This tutorial repository contains the images, annotation data, *.csv* files, and tfrecord files needed to train an "Insect Detector". You can use these images and data to practice making your own Insect Detector. It also has Python scripts that are used to generate the training data and it has scripts to test the object detection classifier out on images, on videos, or on a webcam feed.

If you want to practice training your own insect detector, you can leave all the files in there as they are. You can follow along with this tutorial to see how each of the files were generated, and then run the training. You will still need to generate the TFRecord files (train.record and test.record) as described in procedure 4.

You can also download the frozen inference graph for my already trained insect detector from this Dropbox link
https://www.dropbox.com/s/itirt3saq2duhr8/inference_graph.rar?dl=0
and extract the contents to "\object_detection\inference_graph" folder. The inference graph you have just downloaded will work "out of the box" and you can test it out after all the setup instructions in procedures 2A - 2F has been completed by running the *object_detection_image.py* or _video.py or _webcam.py script.

If you wish to train your own object detector, delete the following files below (**do not delete the folders**):
* All files in "\object_detection\images\train" and "\object_detection\images\test"
* The *test_labels.csv* and *train_labels.csv* files in "\object_detection\images"
* All files in "\object_detection\training"
* All files in "\object_detection\inference_graph"

And now, you are all set to *start training your own object detector from scratch*. Assuming all the files listed above were deleted, this tutorial will explore on and describe how to generate the files for your own training dataset.

#### 2D. Setup a Separate Anaconda Virtual Environment
Search for the Anaconda Prompt utility from the Start menu on Windows. Right click on it, and click “Run as Administrator”. Windows will ask you if you would like to allow it to make changes for your computer, click "Yes".

In the command line interface that pops up and after clicking Yes, create a new virtual environment called “tensorflow1” by running the following command:
```
C:\> conda create -n tensorflow1 pip python=3.5
```
Activate the environment:
```
C:\> activate tensorflow1
```
Update pip:
```
(tensorflow1) C:\>python -m pip install --upgrade pip
```

Install tensorflow-gpu in this environment by issuing:
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```
Note: You can also use the TensorFlow CPU version, but it will run much slower. If you wish to proceed with the CPU-only version, just use "tensorflow" instead of "tensorflow-gpu" in the previous command.

Install the other necessary packages by issuing the following commands:
```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
 ```
Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by Tensorflow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.

#### 2E. Configure the "PYTHONPATH" Environment Variable
A *PYTHONPATH* variable must be created that points to the "\models", "\models\research", and "\models\research\slim directories". Do this by issuing the following commands (from any directory):
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
(Note: Every time you exit the *tensorflow1* virtual environment, the PYTHONPATH variable will be 'reset' and *you have to be set it up again*. To check if it has been set or not, run "*echo %PYTHONPATH%*".)

#### 2F. Compile Protobufs Files and Run "setup.py"
Compiled Protobuf files are used by TensorFlow to configure model and training parameters. Unfortunately, the 'short protoc' compilation command posted on TensorFlow’s Object Detection API installation page does not work on "Windows". Every *.proto* file in the "\object_detection\protos" directory must be called out individually by the command.

In the Anaconda Command Prompt, change directories to the "\models\research" directory:
```
(tensorflow1) C:\> cd C:\tensorflow1\models\research
```
Then copy and paste the following command into the command line and press Enter:
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
This will create a *name_pb2.py* file from every *name.proto* file in the "\object_detection\protos folder".

(Note: TensorFlow occassionally adds new .proto files to the \protos folder. If you get an error saying "ImportError: cannot import name *something_something_pb2*", you may need to update the protoc command to include the new *.proto* files.)

Lastly, run the following commands from the C:\tensorflow1\models\research directory:
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

#### 2G. Test TensorFlow Setup to Verify if it Works
You now have the TensorFlow object detection API all set up to use pre-trained models for object detection, or to train a new one. To test and verify your installation (if it is working), run the *object_detection_tutorial.ipynb* from the "\object_detection directory" as shown below:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
This opens up the Jupyter Notebook script in your default web browser and allows you to step through the code one section at a time. You can proceed through each section by typing *Shift+Enter* or by clicking the *Run* button in the upper toolbar. The section is done running when the “In [ * ]” text next to the section populates with a number (e.g. “In [3]”).
Note: A portion of the script downloads the *ssd_mobilenet_v1* model from GitHub, which is about 74MB and this means it will take some time to complete the section, so *please be patient*.

Once you have went all the way through the script, you should see two labeled images at the bottom section of the page. If you see this, then everything is working properly! Otherwise, the bottom section will report any errors encountered. 

Refer to the below link for the list of errors Edje Electronics encountered while setting up.
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors

Note: If you don't get any errors while running the full Jupyter Notebook, and the labeled pictures still don't appear, try this and go to "object_detection/utils/visualization_utils.py" and edit the import statements around lines 29 and 30 that include matplotlib using *IDLE*, *Notepad++* or open another Jupyter Notebook. Then, try re-running it again.

![Sample Tutorial to Test if Tensorflow is Working](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/object_detection_tutorial_jupyter_notebook_dogs.jpg?raw=true)

### 3. Gather and Label Images
Now that we have the TensorFlow Object Detection API all set up and ready to go, we need to provide the images it will use to train a new object detection classifier.

#### 3A. Gather All Images Needed
TensorFlow requires 'hundreds of images' of an object to obtain and train a good detection classifier. The training images should have 'random' objects in the image along with the desired objects, and should have a *variety of backgrounds and lighting conditions to train a strong classifier*. There should be some images where the desired object is partially unclear, overlapped with something else, or only halfway in the picture.

For my "insect detector", I have two different objects I want to detect (one is whitefly, and the other is eggplant fruit and shoot borer or simply EFSB). I gathered all the image samples from **Google Images**. For my training dataset, I have about 130 images for whiteflies and about 100 images for EFSB, with a variety of other non-desired objects in some of those pictures.

![Training Images](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/training_images_folder.png?raw=true)

You can use either your phone to take pictures of the objects you want to train or download images of those objects from Google Image Search and try to add an extension to your browser to collect all images at once. I recommend the "Download All Images" Chrome extension to 'gather all images in one zip file'. I have about 230 training images and 55 test images for this tutorial.

According to Edje Electronics, make sure the images aren’t 'too large' and they should be 'less than 200KB each', and their resolution shouldn’t be more than '720x1280'. *The larger the images are, the longer it will take to train the classifier*. You can use the *resizer.py* script in this repository to reduce the size of the images.

After you have gathered all the pictures you needed, **move 20% of them to the "\object_detection\images\test"** directory, and **80% of them to the "\object_detection\images\train"** directory. Make sure there are a *variety of pictures* in both the "\test" and "\train" directories.

##### 3B. Label Images
With all the images/pictures gathered, it’s time to label the desired objects in every picture. For this tutorial, I have used **LabelImg**. LabelImg is a helpful tool for labeling images, and its GitHub page has very clear instructions on how to install and use it. Below are the links where you can get LabelImg.
* LabelImg GitHub link: https://github.com/tzutalin/labelImg
* LabelImg download link: https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. This is kind of rigorous but is rewarding after the process.

![Labeling-EFSB](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/labels_EFSB_many.png?raw=true)

![Labeling-whitefly](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/labels_whitefly_many.png?raw=true)

LabelImg saves a *.xml* file containing the 'label data' for each image and these .xml files will be 'used to generate TFRecords', which are one of the inputs to the Tensorflow trainer. Once you have labeled and saved each images, there will be one .xml file for each image in the "\test" and "\train" directories.

#### 4. Generate the Training Data
After the labeling all images, generate the TFRecords (which serves as input data to the TensorFlow training model). This tutorial uses the *xml_to_csv.py* and *generate_tfrecord.py* scripts from *Dat Tran’s Raccoon Detector dataset*, with slight modifications made by Edje Electronics to work with our directory structure.

First, the image *.xml* data will be used to create *.csv* files containing all the data for the train and test images. From the "\object_detection" folder, run the following command in the Anaconda command prompt:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
This produces a *train_labels.csv* and *test_labels.csv* file in the "\object_detection\images" folder.

Then, open the *generate_tfrecord.py* file in a text editor (such as *Notepad++*). Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used in configuring the *labelmap.pbtxt* file in Step 5B. As an example, assuming you are training a classifier to detect a 'sofa', a 'person' and a 'pillow'. You will replace the following code in *generate_tfrecord.py*:
```
# TO-DO replace this with label map
 def class_text_to_int(row_label):
    if row_label == 'whitefly':
        return 1
    elif row_label == 'EFSB':
        return 2
    else:
        None
```        
With the following:        
```
# TO-DO replace this with label map
 def class_text_to_int(row_label):
    if row_label == 'sofa':
        return 1
    elif row_label == 'person':
        return 2
    elif row_label == 'pillow':
        return 3
    else:
        None
```
Afterwards, generate the *TFRecord files* by running the following commands from the "\object_detection" folder:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
(tensorflow1) C:\tensorflow1\models\research\object_detection> python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These generate a *train.record* and a *test.record* file in "\object_detection" folder. These will be used to train the new object detection classifier.

### 5. Create the Label Map and Configure the Training
The last thing to do before running the training is to create a *label map* and edit the training configuration file.

#### 5A. Create the Label Map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor (such as *Notepad++*) to create a new file and save it as *labelmap.pbtxt* in the "C:\tensorflow1\models\research\object_detection\training" folder. (Make sure the file type is *.pbtxt*, not *.txt* !) 
From the text editor, copy or type in the label map in the format below (the example below is the label map for my Insect Detector):
```
item {
  id: 1
  name: 'whitefly'
}

item {
  id: 2
  name: 'EFSB'
}
```
The label map ID numbers should be the same as what is defined in the *generate_tfrecord.py* file. For the sofa, person, and pillow detector example mentioned in Step 4, the *labelmap.pbtxt* file will look like:
```
item {
  id: 1
  name: 'sofa'
}

item {
  id: 2
  name: 'person'
}

item {
  id: 3
  name: 'pillow'
}
```

#### 5B. Configure Training
Lastly, the object detection training 'pipeline' must be configured. This determines which model and what parameters will be used for the training. It is the last step before running the training.

Go through "C:\tensorflow1\models\research\object_detection\samples\configs" and copy the *faster_rcnn_inception_v2_pets.config* file into the \"object_detection\training" directory. Then, open the file with a text editor (such as *Notepad++*). There are several changes to establish the *.config* file, mainly changing the 'number of classes' and 'examples', and adding the file paths to the training data.

Set the following changes to the *faster_rcnn_inception_v2_pets.config* file. Note: The paths must be entered with single forward slashes "/" (NOT backslashes "\"), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in "double quotation marks" ("path" e.g. "C:/tensorflow1/models/research/object_detection/train.record"), NOT 'single quotation marks' ('path' e.g. 'C:/tensorflow1/models/research/object_detection/train.record').
* Line 9. Change *num_classes* to the number of different objects you want the classifier to detect. For the above sofa, person, and pillow object detector, it would be 
    num_classes : **3**.
* Line 106. Change *fine_tune_checkpoint* to:
    fine_tune_checkpoint : **"C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"**
* Lines 123 and 125. In the 'train_input_reader' section, change *input_path* and *label_map_path* to:
    input_path : **"C:/tensorflow1/models/research/object_detection/train.record"**
    label_map_path: **"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"**
* Line 130. Change *num_examples* to the number of images you have in the "\images\test" directory. In my case, I had 53 test images. 
    num_examples: **53**
* Lines 135 and 137. In the eval_input_reader section, change *input_path* and *label_map_path* to:
    input_path : **"C:/tensorflow1/models/research/object_detection/test.record"**
    label_map_path: **"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"**

After setting it all up, save the file after the changes have been made. Awesome! Now, the training job is all configured and is all ready to go!

### 6. Perform the Training
**UPDATE 9/26/18:** As of *version 1.9*, TensorFlow has "deprecated" the *train.py* file and replaced it with *model_main.py* file. Fortunately, the train.py file is still available in the "/object_detection/legacy" folder. Simply move *train.py* from "/object_detection/legacy" folder into the "/object_detection" folder and then continue the following steps below.

All set?! From the "\object_detection" directory, issue the following command to begin the training:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

After considering or doing everything correctly, TensorFlow will initialize the training and the initialization can take up to 30 seconds before the actual training begins. When the training begins, it will look like this:

![Ongoing Training](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/training.jpg?raw=true)

Each step of the training reports the *loss*. This loss will *start high* but will *get lower as the training progresses*. In my training, using *faster_rcnn_inception_v2_coco_2018_01_28* model, the training loss started between 2 to 3 and then *gradually decreases* as the training goes by. *Edje Electronics* suggests 'allowing your model to train until the loss consistently drops below **0.05**, which will take *about 40,000 steps*, or *about 2 hours* (**depending on how powerful your CPU and GPU are**)'. 
Note: *The loss numbers will be different if a different model is used*. In my traing, using SSD MobileNet COCO v2 model, it starts with a loss of about 15, and should be trained until the loss is consistently under 2 as recommended by *Edje Electronics*.

If you want to view the progress of the training, use **TensorBoard**. To get this done, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, change the directory to "C:\tensorflow1\models\research\object_detection", and run the following command:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```    
This will create a webpage on your local machine at "*YourPCName:6006*", which can be viewed through a web browser(such as *Chrome or Firefox*). The TensorBoard page provides information and graphs that show how the training goes as each goes by. One graph to look at is the *Loss Graph*, which shows the *overall loss of the classifier over time*.

![Train Loss Graph](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/insect_detector_graph_training_loss_10000steps_01.png?raw=true)

![Total Loss Graph](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/insect_detector_graph_training_loss_10000steps_02.png?raw=true)

The training job *regularly saves checkpoints about every five minutes*. You can terminate the training by pressing *Ctrl+C* while in the command prompt window if you want to stop the training and test or start the training again later, and *it will restart/resume from the last saved checkpoint*. The checkpoint at the highest number of steps will be used to generate the 'frozen inference graph'.

### 7. Exporting Inference Graph
After the training is complete, the last step is to generate the 'frozen inference graph' (*.pb file*). From the "\object_detection" folder, issue the following command, where “XXXX” in *model.ckpt-XXXX* should be replaced with the 'highest-numbered .ckpt file' in the training folder (In my case, I have "XXXX" equals "10000"):
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-10000 --output_directory inference_graph
```
This produces a *frozen_inference_graph.pb* file in the "\object_detection\inference_graph" folder. The *.pb* file contains the object detection classifier.

### 8. Test Your Newly Trained Object Detection Classifier
The tasks were quite meticulous and rigorous, but after considering or doing everything, we'll be pleased we did it. The object detection classifier should work! Use the attached Python scripts in the "\object_detection" folder or in my GitHub repository to test your newly trained object detector on an *image file*, a *video file*, or a *webcam feed*.

Before running the said Python scripts, you need to modify the *NUM_CLASSES* variable first in the script to equal the number of classes you want to detect. (For my Insect Detector, there are two insects I want to detect, so NUM_CLASSES = **2**.)

To modify and run any of the scripts type 'idle' in the Anaconda Command Prompt (make sure the “tensorflow1” virtual environment is *activated* and make sure you are in the "\object_detection" directory)
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> idle
``` 
To test your object detector on an *image file*, copy the "path" of the image file you want to test and paste it to '*IMAGE_NAME* variable (example is shown below) in the 'object_detection_image.py' script. Save the changes in the script and then run it.
    IMAGE_NAME = **"C:/tensorflow1/models/research/object_detection/test_image_05.jpg"**
or in this format
    IMAGE_NAME = **r"C:\tensorflow1\models\research\object_detection\test_image_05.jpg"**
    
Another way is to move a picture of the object or objects you want to test into the "\object_detection" folder, and then change the *IMAGE_NAME* variable in the *object_detection_image.py* to match the file name of the picture. As an example:
    IMAGE_NAME = **'test_image_05.jpg'**

To test your object detector on a *video file*, copy the "path" of the video file you want to test and paste it to *VIDEO_NAME* variable (example is shown below) in the *object_detection_video.py* script. Save the changes in the script and then run it. 
    VIDEO_NAME = **"C:/tensorflow1/models/research/object_detection/insects_testrun_01.mp4"**
or in this format
    VIDEO_NAME = **r"C:\tensorflow1\models\research\object_detection\insects_testrun_01.mp4"**
    
Another way is to move a picture of the object or objects you want to test into the "\object_detection" folder, and then change the *IMAGE_NAME* variable in the *object_detection_video.py* to match the file name of the picture. As an example:
    VIDEO_NAME = **'insects_testrun_01.mp4'**

To test your object detector from a *webcam feed*, just plug in a USB webcam (such as a *Logitech* webcam, *raspberry pi camera*) and point it at the objects you want to test. 
NOTE: To open default camera using default backend just pass 0 in the "video = cv2.VideoCapture()" line in the *object_detection_webcam.py* script as shown below.
    video = cv2.VideoCapture(**0**)

Alternatively, you can just run the them in the Anaconda Command Prompt to test your objector (make sure the “tensorflow1” virtual environment is *activated* and make sure you are in the "\object_detection" directory):
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python object_detection_image.py
(tensorflow1) C:\tensorflow1\models\research\object_detection> python object_detection_video.py
(tensorflow1) C:\tensorflow1\models\research\object_detection> python object_detection_webcam.py
```
Assuming everything is *working properly*, the object detector will initialize for about a few seconds and then display a window showing any objects it has detected in the image file, in the video file or in the webcam you have fed. Shown below is a snippet of the test run.

![Sample Output](https://github.com/jdsmooth2j/Train-Your-Own-Object-Detector-Using-Tensorflow-Object-Detection-API-Tutorial-Windows/blob/master/detection_4.png?raw=true)

If you have encounter some errors, check the GitHub link below for common errors while setting up the object detection classifier. 
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors)
You can also try Googling the errors and there are lots of useful information over the Internet, most notably on Stack Exchange and on GitHub.

Thank you! Hope you like it! :)
