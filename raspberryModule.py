# import all necessary packages and libraries
from mlx90614 import MLX90614
from smbus2 import SMBus 
import I2C_LCD_DRIVER
import RPi.GPIO as GPIO
import time

# initialize and setup components in their respective gpio pins
# setup red and green LED, set warning to False and set mode to Board
LEDred = 29
LEDgreen = 31
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LEDred, GPIO.OUT)
GPIO.setup(LEDgreen, GPIO.OUT)
# setup Ultrasonic Sensor
trigger = 33
echo = 35
GPIO.setup(trigger, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)
GPIO.output(trigger, False)
# setup buzzer and IR sensor
buzzer = 12
IRsensor = 36
IRsensor2 = 38
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setup(IRsensor, GPIO.IN)
GPIO.setup(IRsensor2, GPIO.IN)
# initialize thermal sensor
bus = SMBus(1)
thermal = MLX90614(bus, address=0x5a)
#initialize lcd screen
lcd = I2C_LCD_DRIVER.lcd()
# define a function that can get a body temperature
def getTemperature():
    return thermal.get_object_1()
# define a function that can measure a ultrasonic distance
def getDistance():
    GPIO.output(trigger, True)
    time.sleep(0.00001)
    GPIO.output(trigger, False)
    while GPIO.input(echo) == 0:
        startTime = time.time()
    while GPIO.input(echo) == 1:
        endTime = time.time()
    duration = endTime - startTime
    distance = duration*17150
    distance = round(distance, 2)
    time.sleep(0.5)
    return distance 
# define a function that will display text on lcd
def getDisplay(text,num):
    lcd.lcd_display_string("%s" %text, 2)
    lcd.lcd_display_string("Temperature: %.2f" %num, 3)
    time.sleep(2)
    lcd.lcd_clear()
# define a function to on buzzer and red LED for high tempature and no mask
def getHighNoMask():
    GPIO.output(buzzer, True)
    GPIO.output(LEDred, True)
    time.sleep(1)
    GPIO.output(buzzer, False)
    GPIO.output(LEDred, False)
# define a function to on green LED and buzzer
def getWelcome():
    GPIO.output(LEDgreen, True)
    time.sleep(1)
    GPIO.output(LEDgreen, False)
# define a function can for infrared sensor
def getInfrared():
    return GPIO.input(IRsensor)
def getInfrared2():
    return GPIO.input(IRsensor2)
# define a function for GPIO clean up
def getClean():
    GPIO.cleanup()
