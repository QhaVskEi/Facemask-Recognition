import pyrebase
import datetime
import cv2


config = {
    'apiKey': "AIzaSyAvUQtpoIvlJv-TurTDMBYU-2_UnBqobLo",
    'authDomain': "smart-door-automation.firebaseapp.com",
    'databaseURL': "https://smart-door-automation-default-rtdb.firebaseio.com",
    'projectId': "smart-door-automation",
    'storageBucket': "smart-door-automation.appspot.com",
    'messagingSenderId': "233324917439",
    'appId': "1:233324917439:web:e0b44073c8cc645a99cc0a",
    'measurementId': "G-HGPYDH4ZJJ"
}

firebase = pyrebase.initialize_app(config)
database = firebase.database()
storage = firebase.storage()

print("System info: firebase database running")

facemask = 0
nofacemask = 0
picture = 0

def mask():
    global facemask
    num = int(facemask + 1)
    facemask = num
    database.child("ProjectData").child("Mask").set(facemask)
    print("Facemask Sent to datebase")

def noMask():
    global nofacemask
    num = nofacemask + 1
    nofacemask = num
    database.child("ProjectData").child("NoMask").set(nofacemask)
    print("No facemask sent to database")

def captureImage(image):
    global picture
    num = picture + 1
    picture = num
    image_name = "no-mask%i.jpg" %picture
    cv2.imwrite(image_name, image)
    storage.child(image_name).put(image_name)
    print("Image Sent to database")

def sendtemp(data):
    database.child("ProjectsTemperature").push(data)
    print("Temperature Sent to database")
    
