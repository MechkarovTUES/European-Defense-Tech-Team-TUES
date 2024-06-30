## to be ran inside Salience-DETR
import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
import random
import string

import torch
import cv2
from configs.salience_detr.salience_detr_focalnet_large_lrf_800_1333 import model
from util.utils import load_state_dict


MQTT_SERVER = "4bc1ede6cf764b56ad9895704e35b258.s1.eu.hivemq.cloud"
MQTT_PATH = "Image"
CONF_THRESHOLD = 0.5
PATH_TO_PTH = "./best_ap50.pth"


def load_model():
    weight = torch.load(PATH_TO_PTH, map_location=torch.device('cpu'))
    load_state_dict(model, weight)
    return model.eval()

ourmodel = load_model()

def predict_image(model, image):
    torch_image = torch.tensor(image.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        predictions = model(torch_image)[0]
    
    conf_boxes = [box.tolist() for i, box in enumerate(predictions['boxes']) if predictions['scores'][i] > CONF_THRESHOLD]
    
    print(conf_boxes)
    return conf_boxes

def generate_random_filename(extension):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.' + extension

def decombine(input_file): # for recieving from module
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    image1 = Image.frombytes(data['image1_mode'], data['image1_size'], data['image1'])
    image2 = Image.frombytes(data['image2_mode'], data['image2_size'], data['image2'])

    gps_coordinates = data['gps_coordinates']
    timestamp = data['timestamp']

    return image1, image2, gps_coordinates, timestamp

def combine(image1_path, image2_path, timestamp, output_file, boxes, geo_loc): # for sending to website
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # w, h = im.size
    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size
    
    lat, lon = geo_loc.split(',')

    # Create a dictionary to store all components
    data = {
        'image1': image1.tobytes(),
        'image1_size': image1.size,
        'image1_mode': image1.mode,
        'image1_height': image1_height,
        'image1_width': image1_width,
        'image2': image2.tobytes(),
        'image2_size': image2.size,
        'image2_mode': image2.mode,
        'image2_height': image2_height,
        'image2_width': image2_width,
        'mines_coordinates': boxes,
        'latitude': lat,
        'longitude': lon,
        'fov': 50,
        'heading': 1.0,
        'timestamp': timestamp,
    }

    # Serialize the data to a binary format
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
        return pickle.dumps(data)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_PATH)

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg, aimodel=ourmodel):
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('outgoing'):
        os.makedirs('outgoing')

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
        
        image1_cv2 = cv2.imread(image1_path)
        boxes = predict_image(model, image1_cv2)
        print(boxes)
        
        rnd_out_name = generate_random_filename("pkl")
        client.publish("Website", payload=combine(image1_path, image2_path, timestamp, f"outgoing/{rnd_out_name}", boxes, gps_coordinates), qos=1)
    except IOError as e:
        # print(e)
        print("Couldunt decombine file")

client = mqtt.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
client.username_pw_set("badabum", "y&Ttg)~A9u@44W$")
client.connect(MQTT_SERVER, 8883, 60)
client.subscribe("Image")

client.loop_forever()
