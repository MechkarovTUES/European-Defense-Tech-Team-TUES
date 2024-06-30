import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
import pickle
import io
MQTT_SERVER = "4bc1ede6cf764b56ad9895704e35b258.s1.eu.hivemq.cloud"
MQTT_PATH = "Image"

def combine_images_and_text(image1_path, image2_path, gps_coordinates, time, output_image_path):
    # Load the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Combine the images side by side
    combined_width = image1.width + image2.width
    combined_height = max(image1.height, image2.height)
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    
    # Create a draw object to add text
    draw = ImageDraw.Draw(combined_image)
    
    # Define the font size
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Add GPS coordinates and time to the image
    text = f"GPS: {gps_coordinates}\nTime: {time}"
    text_position = (10, combined_height - 40)  # Position at bottom left corner
    
    draw.text(text_position, text, font=font, fill="white")
    
    # Save the combined image
    combined_image.save(output_image_path)
    
    # Serialize the data (GPS coordinates and time)
    metadata = {
        'gps_coordinates': gps_coordinates,
        'time': time
    }
    metadata_path = output_image_path + '.metadata'
    with open(metadata_path, 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)
    
    print(f"Combined image saved to {output_image_path} with metadata.")

def combine(image1_path, image2_path, gps_coordinates, timestamp, output_file):
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

# The callback for when the client receives a CONNACK response from the server.
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

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
# combine_images_and_text("./thermal/thermal_0.jpg", "origin/origin_0.jpg", "77.124345, 11.234567", "3:44:23", "combined/comb_0.jpg")
combine("./thermal/thermal_0.jpg", "origin/origin_0.jpg", "77.124345, 11.234567", "3:44:23", "combined/comb_0.jpg")

f=open("combined/comb_0.jpg", "rb") #3.7kiB in same folder
fileContent = f.read()
byteArr = bytearray(fileContent)

client.publish("Image", byteArr, qos=1)
client.loop_forever()
