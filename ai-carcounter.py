###################################################################
# AI CarCounter                                                   #
# Brandon Garcia, Donovan Dahlin, Jackson Justus, Tristan Rolling #
# Senior Capstone Project - 5/3/2022                              #
#-----------------------------------------------------------------#
# This program is used to count the number of cars on a beach     #
# from drone footage. It cuts the inputed video into frames, and  #
# analyses those frames using a specially trained model hosted on #
# www.roboflow.com. Analysed frames are annotated with bounding   #
# boxes and stitched into a 30 fps mp4 video. On program          #
# complition/interruption the total number of cars counted is     #
# outputted to the terminal.                                      #
#-----------------------------------------------------------------#
# Example Usage:                                                  #
# python vidSplit.py myVideo.mp4                                  #
###################################################################

#imports
#import sys  #command line arguments
import cv2  #image and video file manipulation
import os   #creating and deleting tempory files/navigating paths
import json #inferenced data analysis
from roboflow import Roboflow #access roboflow-hosted model API

#establish connection to roboflow-hosted model
rf = Roboflow(api_key="UKHUplpQaZkAGU2Od2VD")
project = rf.workspace("tristan-rolling-gmail-com").project("the-beach-cars-capstone")
version = project.version(1)
model = version.model

#set up video and counting variables
video = ""
#vidcap = cv2.VideoCapture(sys.argv[1])
videoPath = input("Enter Input Video Path/File Name: ")
vidcap = cv2.VideoCapture(videoPath)
frameReadSuccessfully, image = vidcap.read() #read the first frame of the video.
frameCount = 0 #count the number of frames processed
carCount = 0 #count the number of cars detected

#while frames are successfully read...
while frameReadSuccessfully:
    print("Processing Frame: " + str(frameCount)) #print frame count
    cv2.imwrite("vidSplit_frame.jpg", image) #temporarily save current frame to a JPEG file.
    outputVideoFrame = ""
    
    #proccess frame
    prediction = model.predict("vidSplit_frame.jpg") #send the temporary JPEG to roboflow for object-detection.
    #prediction.plot() #<-- Debugging Tool. opens the processed frame in a seperate window.
    try: #frames without any detected objects raise an exception from the roboflow API. this try block catches any of those exceptions.
        prediction.save(output_path="processedFrame.jpg") #temporarily save the processed frame to a JPEG file.
        if(frameCount % 270 == 0): #if the frame is far enough into the video to contain a completely new screenworth of content (~9 secs of 30fps video time), count all the cars in the frame using roboflow's json response.
            frameData = json.loads(str(prediction.json()).replace("'", "\""))
            carCount += len(frameData["predictions"])
            print("\n-------------------------------------------\ncarCount Checkpoint! Cars Counted: " + str(carCount) + "\n-------------------------------------------\n")
        outputVideoFrame = cv2.imread("processedFrame.jpg") #
    except:# Exception as e:
        #print(e)
        outputVideoFrame = cv2.imread("vidSplit_frame.jpg")
    
    #write frame to output video (out.mp4)
    frameHeight, frameWidth, frameLayers = outputVideoFrame.shape
    if(frameCount == 0):
        video = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (frameWidth, frameHeight))
    video.write(outputVideoFrame)
    
    #remove temporary files.
    os.remove("vidSplit_frame.jpg")
    if(os.path.isfile("processedFrame.jpg")):
        os.remove("processedFrame.jpg")
    
    #read next frame of input video.
    frameReadSuccessfully, image = vidcap.read()
    frameCount += 1

#clean up and end program.
cv2.destroyAllWindows()
video.release()

print("\n\nTotal Cars Counted: " + str(carCount))
exit = input("Press enter to end the program.")