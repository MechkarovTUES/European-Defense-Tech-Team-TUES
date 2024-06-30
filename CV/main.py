import time
from pymavlink import mavutil
import cv2
import subprocess
import os
import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
import pickle
import io

#MQTT INIT
MQTT_SERVER = "4bc1ede6cf764b56ad9895704e35b258.s1.eu.hivemq.cloud"
MQTT_PATH = "Image"
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)
    # The callback for when a PUBLISH message is received from the server.

client = mqtt.Client()
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
client.username_pw_set("badabum", "y&Ttg)~A9u@44W$")
client.connect(MQTT_SERVER, 8883, 60)
client.subscribe("Image")

# client.loop_start()

#Packet formating
def packet(image1_path, image2_path, gps_coordinates, timestamp, output_file):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Create a dictionary to store all components
    data = {
        'image1': image1.tobytes(),
        'image1_size': image1.size,
        'image1_mode': image1.mode,
        'image2': image2.tobytes(),
        'image2_size': image2.size,
        'image2_mode': image2.mode,
        'gps_coordinates': gps_coordinates,
        'timestamp': timestamp
    }

    # Serialize the data to a binary format
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)



#Initialize Flir One Pro
# subprocess.run("./ThermalSetup.sh")
# proc = subprocess.Popen(["./ThermalSetup.sh"])


#GPS
connection = mavutil.mavlink_connection('tcp:10.41.1.1:5790')
connection.wait_heartbeat()

#Thermal/Origin Camera
origin = cv2.VideoCapture(0)
thermal = cv2.VideoCapture(3)

if not origin.isOpened():
    print("Error: Could not open video device")
    exit()

# Set resolution
origin.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
origin.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gps_raw():
    while True:
        msg = connection.recv_match(blocking=False)
        if not msg:
            continue
        if msg.msgname == "GPS_RAW_INT":
            # print(f"lat: {msg.lat}, lon: {msg.lon}")
            return f"{msg.lat / 1e7}, {msg.lon / 1e7}"
        



start = time.time()
counter = 0
if __name__ == "__main__":
    print(time.strftime("%H:%M:%S", time.localtime()))
    while time.time() - start <= 4:
        
        ret1, frameOrigin = origin.read()
        ret2, frameThermal = thermal.read()

        if not ret1 and not ret2:
            print("Error: Could not read frame")
            break

        cv2.imwrite(f'thermal/thermal_{counter}.jpg', frameThermal)
        cv2.imwrite(f'origin/origin_{counter}.jpg', frameOrigin)

        coordinates = gps_raw()
        print(coordinates)
        
        packet(f"./thermal/thermal_{counter}.jpg", f"origin/origin_{counter}.jpg", coordinates, time.strftime("%H:%M:%S", time.localtime()), f"combined/comb_{counter}.pkl")
        f=open(f"combined/comb_{counter}.jpg", "rb")
        fileContent = f.read()
        byteArr = bytearray(fileContent)
        client.publish("Image", byteArr, qos=1)

        # client.loop_forever()
        
        time.sleep(1)
        counter += 1
    origin.release()
    thermal.release()
    client.loop_forever()
    # client.loop_stop()
    #proc.terminate()
