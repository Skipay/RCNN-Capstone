----------------------------------------------------------------------------------------------------------------------
How to Train and Test MRCNN model
----------------------------------------------------------------------------------------------------------------------
Images were collected by a Drone looking straight down at a height of 200 ft.

Masking images are done in Labelme for the COCO files while the VGG files are done in makesense.ai.

To train the AI one needs to have tensorflow downloaded which can be done with: pip install tensorflow. 

Following that one needs to install the requirements this is done with: pip install -r requirements.txt 
Then to train the weights call: python car.py train --dataset=path/to/dataset --weights=coco
The masks will output to the logs folder as a mask_rcnn_car_00xx.h5 file with x representing the epoch count.

To make use of your NVIDIA GPU to train the model use CUDA. Use the CUDA toolkit found on the website to download the neccessary
files. From that you need to find the correct .dll files that are seperate from CUDA download. https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
Please follow this documentation.

CUDA is not required to train the model and you will see alot of warning regarding that but they are just warnings and can 
be ignored.

To test the mask run the video_demo.py after you have changed the the model_DIR with the path to your last epoch .h5 file.

----------------------------------------------------------------------------------------------------------------------
How to Run Roboflow
----------------------------------------------------------------------------------------------------------------------
To run the roboflow executable just click on it and in a command prompt it will ask you for the path to the 
video you wish to run as your input.

Altenatively to run python from the script file call use the command: python scriptname.py path/to/video

Please note that to run roboflow from the command line you need to: pip install roboflow

----------------------------------------------------------------------------------------------------------------------
File Explanations
----------------------------------------------------------------------------------------------------------------------
The dataset folder has 2 different sets of training data. One COCO and one VGG.

The car.py is configured to train witht the VGG fole format while the car2.py is configured to train with the COCO dataset

Sources

https://www.youtube.com/watch?v=faZtmLI8pEw
https://github.com/matterport/Mask_RCNN
https://pysource.com/2021/08/10/train-mask-r-cnn-for-image-segmentation-online-free-gpu/
https://www.youtube.com/watch?v=tcu4pr948n0

