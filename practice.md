```python
### drive.py ###

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
```

```python
### model.py ###

import cv2
import csv
import numpy as np
from scipy import ndimage

lines = []

'''
학습 데이터를 sample 데이터를 쓴 이유는 직접 시뮬레이션을 돌렸더니 너무 느려서 도저히 할수가 없었음
대충 샘플 데이터를 보니 라인 중간에서 잘 주행을 한거같아서 좋은 데이터이긴 한데
라인을 완전히 벗어났을때 되돌아오는 방법을 학습시킬 수 없다는게 한계임
'''
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del lines[0]

images = []
measurements = []


''' <This is only for center camera image>
for line in lines:
    source_path = line[0]
    file_name = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/data/IMG/' + file_name
    image = ndimage.imread(current_path)

    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
'''


# To augment image data of left,right camera
# For left camera image, steering angle needs to turn right
# For right camera image, steering angle needs to turn left
# This assumption is that every image is pictured when camera is in center of car



for line in lines:
    for i in range(3):
        correction_factor = 0.2
        source_path = line[i]
        file_name = source_path.split('/')[-1]
        current_path = '/opt/carnd_p3/data/IMG/' + file_name
        image = ndimage.imread(current_path)

        images.append(image)
        
        if i==0:
            measurement = float(line[3])
            measurements.append(measurement)
        elif i==1:
            measurement = float(line[3]) + correction_factor
            measurements.append(measurement)
        elif i==2:
            measurement = float(line[3]) - correction_factor
            measurements.append(measurement)
            


X_train = np.array(images)
Y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D

# <This is third model using NVIDIA CNN architecture>
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(36,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(48,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



# <This is for cropping non essential part of image like background, hood>
'''
cropping 의 효과
이 전까지는 다리를 못건넜는데 cropping 하고나서 다리를 매우 잘 건넜음, 부가적으로 학습시키는데 걸리는 시간도 줄어듦
이유를 생각해봤는데 시간 줄어드는거야 데이터 크기가 줄어드니까 당연한거고
배경을 제거하면 차선만 남으니까 얘가 어떤 상황인지 판단하기가 더 유용한듯
즉 배경까지 있으면 이 상황이 학습한 이미지중 어떤거랑 같은건지 판단하기 위한 조건들이 많은데
배경을 제거하니까 같은 상황임을 판단하기 용이해진듯
'''

'''
# <This is second model using LeNet>


model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(10,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

'''
   <This is first model just using 1 perceptron>
model.add(Flatten())
model.add(Dense(1))
'''

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
```
