import time
from pymavlink import mavutil

connection = mavutil.mavlink_connection('tcp:10.41.1.1:5790')
connection.wait_heartbeat()

while True:
    msg = connection.recv_match(blocking=False)
    if not msg:
        continue
    if msg.msgname == "ATTITUDE":
        print(msg)
    