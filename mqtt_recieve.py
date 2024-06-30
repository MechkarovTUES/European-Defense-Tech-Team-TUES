import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
import random
import string

MQTT_SERVER = "4bc1ede6cf764b56ad9895704e35b258.s1.eu.hivemq.cloud"
MQTT_PATH = "Image"

def generate_random_filename(extension):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.' + extension

def decombine(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    image1 = Image.frombytes(data['image1_mode'], data['image1_size'], data['image1'])
    image2 = Image.frombytes(data['image2_mode'], data['image2_size'], data['image2'])

    gps_coordinates = data['gps_coordinates']
    timestamp = data['timestamp']

    return image1, image2, gps_coordinates, timestamp

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_PATH)

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('data'):
        os.makedirs('data')

    output_pkl_path = os.path.join('data', 'combined_data.pkl')
    with open(output_pkl_path, "wb") as f:
        f.write(msg.payload)
    print("Data Received")

    try:
        image1, image2, gps_coordinates, timestamp = decombine(output_pkl_path)
        image1_path = os.path.join('images', generate_random_filename('jpg'))
        image2_path = os.path.join('images', generate_random_filename('jpg'))
        image1.save(image1_path)
        image2.save(image2_path)

        print(f"Decombined successfully: {image1_path}, {image2_path}, {gps_coordinates}, {timestamp}")

        if None in (image1_path, image2_path, gps_coordinates, timestamp):
            print("Decombination failed due to missing metadata.")
        else:
            print(f"Decombined successfully: {image1_path}, {image2_path}, {gps_coordinates}, {timestamp}")
    except IOError as e:
        print(e)

client = mqtt.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
client.username_pw_set("badabum", "y&Ttg)~A9u@44W$")
client.connect(MQTT_SERVER, 8883, 60)
client.subscribe("Image")

client.loop_forever()
