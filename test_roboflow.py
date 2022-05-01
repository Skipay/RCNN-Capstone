import sys
import cv2
import os
from roboflow import Roboflow
rf = Roboflow(api_key="UKHUplpQaZkAGU2Od2VD")
project = rf.workspace("tristan-rolling-gmail-com").project("the-beach-cars-capstone")
version = project.version(1)
model = version.model
video = ""

vidcap = cv2.VideoCapture(sys.argv[1])
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("vidSplit_frame.jpg", image) #save frame as JPEG file
    frame = ""
    
    prediction = model.predict("vidSplit_frame.jpg")
    #prediction.plot()
    try:
        prediction.save(output_path="processedFrame.jpg")
        frame = cv2.imread("processedFrame.jpg")
    except:
        frame = cv2.imread("vidSplit_frame.jpg")
        
    frameHeight, frameWidth, frameLayers = frame.shape
    
    if(count == 0):
        video = cv2.VideoWriter("out.avi", 0, 24, (frameWidth, frameHeight))
    video.write(frame)
    
    os.remove("vidSplit_frame.jpg")
    if(os.path.isfile("processedFrame.jpg")):
        os.remove("processedFrame.jpg")
    
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1

cv2.destroyAllWindows()
video.release()