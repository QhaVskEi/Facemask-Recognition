# import necessary packages and libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from threading import Thread
import raspberryModule as RPM
import DatabaseModule as DBM
import numpy as np
import cv2 as cv
import argparse
import time
import os

# initialize number of person
person = 0
num = 1
# define a function to process temperature and other sensors
def create_process(label, capture, percent, distance, startime):
    global person
    # if infrared sensor detect a hand it get temperature
    while RPM.getInfrared() != 1:
        # check if person inside doesn't exceed in 15 people
        temperature = RPM.getTemperature()
        if person < 15:
            person += num
            print("----------------------------------")
            print("Number of person inside %i" %person)
            # publish condition, do not allowed if high temperature and no mask
            if (label == "Mask"):
                if temperature >= 37:
                    RPM.getHighNoMask()
                    RPM.getDisplay("High Temperature",temperature)
                    DBM.sendtemp(temperature)
                    endtime = time.time()
                    totaltime = endtime - startime
                    print("Mask Percentage: %s" %percent)
                    print("High Temperature: %s" %temperature)
                    print("Distance: %i" %distance)
                    print(f"Total time taken: {totaltime:.2f} seconds")
                    os.system('pico2wave -l en-US -w say.wav "High temperature detected, you are prohibited to get inside" && aplay say.wav')
                elif percent < 90:
                    RPM.getHighNoMask()
                    RPM.getDisplay("Adjust your facemask",temperature)
                    endtime = time.time()
                    totaltime = endtime - startime
                    print("Facemask wear incorrectly")
                    print("Mask Percentage: %s" %percent)
                    print("Temperature: %s" %temperature)
                    print("Distance: %i" %distance)
                    print(f"Total time taken: {totaltime:.2f} seconds")
                    os.system('pico2wave -l en-US -w say.wav "Wear your mask properly" && aplay say.wav')
                else:
                    RPM.getWelcome()
                    RPM.getDisplay("Facemask Detected",temperature)
                    DBM.mask()
                    DBM.sendtemp(temperature)
                    endtime = time.time()
                    totaltime = endtime - startime
                    print("Mask Percentage: %s" %percent)
                    print("Temperature: %s" %temperature)
                    print("Distance: %i" %distance)
                    print(f"Total time taken: {totaltime:.2f} seconds")
                    os.system("mosquitto_pub -t RaspberryPi/Inload -m 1")
                    os.system("mosquitto_pub -t RaspberryPi/Inload -m 0")
                    os.system('pico2wave -l en-US -w say.wav "Mask detected, welcome. Door will open in a seconds" && aplay say.wav')
            else:
                RPM.getHighNoMask()
                RPM.getDisplay("No Mask Detected",temperature)
                DBM.noMask()
                DBM.captureImage(capture)
                DBM.sendtemp(temperature)
                endtime = time.time()
                totaltime = endtime - startime
                print("No facemask Percentage: %s" %percent)
                print("Temperature: %s" %temperature)
                print("Distance: %i" %distance)
                print(f"Total time taken: {totaltime:.2f} seconds")
                os.system('pico2wave -l en-US -w say.wav "No mask detected, you are prohibited to get inside" && aplay say.wav')
        else:
            RPM.getHighNoMask()
            RPM.getDisplay("Exceed room capacity",temperature)
            print("----------------------------------")
            print("Number of person inside %i" %person)
            print("Exceeding room capacity")
            os.system('pico2wave -l en-US -w say.wav "Exceeding room capacity, only 15 people can get inside." && aplay say.wav')

# define the funtion that create location labels and prediction
def create_label(capture, box, pred, distance, startime):
    # write a box and prediction label
    (startX, startY, endX, endY) = box
    (mask, withoutmask) = pred
    # determine the class label and color we'll use to draw the 
    # bounding box and text
    percent = max(mask, withoutmask)*100
    label = "Mask" if mask > withoutmask else "No Mask"
    color = (0,255,0) if label == "Mask" else (0,0,255)
    label_out = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
    # display the label and bounding box rectangle on the output frame
    cv.putText(capture, label_out, (startX, startY - 10), cv.FONT_HERSHEY_TRIPLEX, 0.45, color, 1)
    cv.rectangle(capture, (startX, startY), (endX, endY), color, 2)
    # pass the label into function that controling other sensors
    Thread(target=create_process, args=(label, capture, percent, distance, startime)).start()

# define the function the create frame for detection
def frame_create(capture, faceNet, maskNet, distance, startime):
    # detect faces in the frame and determine if they wearing a mask or not
    (locs, preds) = detect_predict_mask(capture, faceNet, maskNet)
    # loop over the detected face and their corresponding location
    for (box, pred) in zip(locs, preds):
        Thread(target=create_label, args=(capture, box, pred, distance, startime)).start()

# define the loop function
def create_loop(capture, faceNet, maskNet):
    global person
    distance = RPM.getDistance()
    # if person detect from sensor in the back, door will open
    if RPM.getInfrared2() != 0:
        #if person detect from range 80cm, will capture the faces and prediction
        startime = time.time()
        if distance <= 80:
            #frame_create(capture, faceNet, maskNet)
            Thread(target=frame_create(capture, faceNet, maskNet, distance, startime)).start()
    else:
        person -= num
        print("----------------------------------")
        print("Number of people inside %i" %person)
        os.system("mosquitto_pub -t RaspberryPi/Inload -m 1")
        os.system("mosquitto_pub -t RaspberryPi/Inload -m 0")
        os.system('pico2wave -l en-US -w say.wav "Door Open, thank you and come again" && aplay say.wav')

# define detection and prediction function
def detect_predict_mask(capture, faceNet, maskNet):

    # grab the dimension of the frame then construct blob from it
    (h, w) = capture.shape[:2]
    blob = cv.dnn.blobFromImage(capture, 1.0, (224,224), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain face detection
    faceNet.setInput(blob)
    detection = faceNet.forward()
    # initialize the list of prediction, faces and location from our network
    faces = []
    locs = []
    preds = []
    # looping over the detection
    for i in range(0, detection.shape[2]):
        # extract the level of confidence
        confidence = detection[0,0,i,2]
        # filter out the weak dectection by ensuring the confidence
        # is greater than the minimum confidence
        if confidence > 0.5:
            #compute the x and y coordinate of the bounding box
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            #ensure the bounding box fall within the dimension of frame
            (startX, startY) = (max(0,startX),max(0,startY))
            (endX, endY) = (min(w-1,endX),min(h-1,endY))
            #extract the face ROI, convert from BGR to RGB
            #ordering, resize it to 224x224 and process it
            face = capture[startY:endY,startX:endX]
            face = cv.cvtColor(face,cv.COLOR_BGR2RGB)
            face = cv.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            #add the face and bounding box to their respective list
            faces.append(face)
            locs.append((startX,startY,endX,endY))
    #only make prediction if at least one face detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size = 32)
    #return tuple of the face location and their corresponding locations
    return (locs, preds)

if __name__ == "__main__":
    
    #load our serialized facemask detector model from files
    ap = argparse.ArgumentParser()
    ap.add_argument("-f","--face",type=str,default="face_detector",help="path to face detector model directory")
    ap.add_argument("-m","--model",type=str,default="mask_detector.model",help="path to trained mask detector model")
    ap.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probability to filter weak detection")
    args = vars(ap.parse_args())
    #load our facemask model
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(args["model"])
    
    #initialize the video stream
    print("System Info: Starting Video Stream")
    cap = cv.VideoCapture(0)
    #starting loop and capturing images
    while True:
        success, capture = cap.read()
        capture = cv.resize(capture, (900, 800))
        #if success, measure the distance of person from the system
        if success:
            # throw to the thread function a model and images
            Thread(target=create_loop(capture, faceNet, maskNet)).start()
        #display or show the output frame
        cv.imshow("Smart Door Automation", capture)
        key = cv.waitKey(1) & 0xFF
        #if the 'q' key was pressed, break the loop
        if key == ord("q"):
            break
    #cleaning resources
    cv.destroyAllWindows()
    RPM.getClean()
    cap.release()


