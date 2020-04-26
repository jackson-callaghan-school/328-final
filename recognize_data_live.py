import socket, sys

import threading
import numpy as np
import pickle
from features import extract_features
from util import reorient, reset_vars

host = ''
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

class_names = ["falling", "jumping", "sitting", "standing", "turning", "walking"]

with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()


def onActivityDetected(activity):
    """
    Notifies the user of the current activity
    """
    
    # if detected_activity == "falling":
        # print("Detected ")
    
    print("Detected activity:" + activity)


def predict(window):
    """
    Given a window of accelerometer data, predict the activity label.
    """

    # TODO: extract features over the window of data

    feature_names, feature_vector = extract_features(window)

    # TODO: use classifier.predict(feature_vector) to predict the class label.

    # Make sure your feature vector is passed in the expected format
    class_label = classifier.predict(feature_vector.reshape(1, -1))

    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # pass the activity name to onActivityDetected()

    activity_name = class_names[class_label.astype(int)[0]]

    onActivityDetected(activity_name)

    return


sensor_data = []
window_size = 100  # ~1 sec assuming 100 Hz sampling rate
step_size = 100  # no overlap
index = 0  # to keep track of how many samples we have buffered so far
reset_vars()  # resets orientation variables

while 1:
    try:
        message, address = s.recvfrom(8192)
        info = message.decode()

        # print (info)
        data = info.split(",")
        if len(data) < 13:
            continue

        # print(data)
        for i in range(9):
            data[i] = data[i].strip()
            # print(str(i) + " " + data[i])

        timestamp = data[0]
        accel_x = data[2]
        accel_y = data[3]
        accel_z = data[4]
        gyro_x = data[6]
        gyro_y = data[7]
        gyro_z = data[8]

        temp_data = reorient(accel_x, accel_y, accel_z)
        temp_data.append(gyro_x)
        temp_data.append(gyro_y)
        temp_data.append(gyro_z)

        sensor_data.append(temp_data)
        index += 1

        while len(sensor_data) > window_size:
            sensor_data.pop(0)

        if index >= step_size and len(sensor_data) == window_size:
            t = threading.Thread(target=predict, args=(
                np.asarray(sensor_data[:]),))
            t.start()
            index = 0

    except KeyboardInterrupt:
        print("interrupted")
        file.close()