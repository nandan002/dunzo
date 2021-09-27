from flask import Flask, render_template, request, Response
import cv2
from keras.models import load_model
import numpy as np
import base64
import re
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('face.xml')
model = load_model('best_model.h5')

class_labels = ['DUNZO', 'NOLOGO']


def video_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    labels = []
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # print(faces)
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            image = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            image = np.expand_dims(image, axis=0)
            preds = model.predict_generator(image)
            labels = class_labels[preds.argmax()]
            if labels == 'DUNZO':
                result = "HUMAN DETECTED AND DUNZO GUY IS HERE"
            else:
                result = "HUMAN DETECTED AND DUNZO GUY IS NOT HERE"
            # cv2.putText(frame, labels, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        result = "NO HUMAN DETECTED"
    return result


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST', 'GET'])
def res():
    if request.method == 'POST':
        global result
        img = request.files['file1']
        img.save("img.jpg")
        image = cv2.imread("img.jpg")
        result = video_process(image)
        return render_template("after.html", result=result)

# def decode_base64(data, altchars=b'+/'):
#     """Decode base64, padding being optional.
#
#     :param data: Base64 data as an ASCII byte string
#     :returns: The decoded byte string.
#
#     """
#     data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
#     missing_padding = len(data) % 4
#     if missing_padding:
#         print("True")
#         data += b'='* (4 - missing_padding)
#     return base64.b64decode(data, altchars)

@app.route('/video_p', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        global result
        img = request.get_data()
        print(img)
        # imgdata = decode_base64(img)
        # with open("video.png", 'wb') as f:
        #     f.write(imgdata)

        image = cv2.imread("video.png")
        result = video_process(image)
        return render_template("after.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, threaded=True,host='0.0.0.0',port=5050)
