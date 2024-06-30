import time
import cv2
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



start = time.time()
counter = 0

def get_files(folder):
    jpg_files = [file for file in os.listdir(folder) if file.lower().endswith('.jpg')]
    jpg_files_sorted = sorted(jpg_files, key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)))
    for file in jpg_files_sorted:
        coordinates = "48.1378896, 11.6883406"
        packet(f"./Thermal2/t{file}", f"origin/{file}", coordinates, time.strftime("%H:%M:%S", time.localtime()), f"combined/comb_{counter}.pkl")
        f=open(f"combined/comb_{counter}.jpg", "rb")
        fileContent = f.read()
        byteArr = bytearray(fileContent)
        client.publish("Image", byteArr, qos=1)

        # client.loop_forever()
        
        time.sleep(1)
    # return len(jpg_files)


if __name__ == "__main__":
    print(time.strftime("%H:%M:%S", time.localtime()))
        
    get_files("Thermal2")
    
    origin.release()
    thermal.release()
    client.tls_set(tls_version=mqtt.set.PROTOCOL_TLS)
    client.loop_forever()
    # client.loop_stop()
    #proc.terminate()
