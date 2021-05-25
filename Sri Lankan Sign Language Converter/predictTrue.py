import numpy as np
from keras.models import model_from_json
import operator
import cv2
from PIL import ImageFont, ImageDraw, Image
import sys, os

# Load the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

pre = []
s = ''
n = ''
p = ''
word=''
num_frames = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 25
    x2 = frame.shape[1] - 40
    y2 = int(0.5 * frame.shape[1])
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'ම්': result[0][0],
                  'එ': result[0][1],
                  'ක්': result[0][2],
                  'ව්': result[0][3],
                  'ආ': result[0][4],
                  'අ': result[0][5],
                  'ඇ': result[0][6],
                  'ඔ': result[0][7],
                  'උ': result[0][8]}
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    fontpath = "iskpota.ttf"
    font = ImageFont.truetype(fontpath, 16)
    img_pill = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pill)
    draw.text((330, 420), (str("Make sure your hand is inside the marked area ")), font=font)
    frame = np.array(img_pill)

    fontpath = "iskpota.ttf"
    font = ImageFont.truetype(fontpath, 16)
    img_pill = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pill)
    draw.text((330, 450), (str("Press C to Clear")), font=font)
    draw.text((10, 190), (str(word)), font=font)
    frame = np.array(img_pill)

    # Display the predictions
    fontpath = "iskpota.ttf"
    font = ImageFont.truetype(fontpath, 32)
    img_pill = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pill)
    draw.text((10, 60), (prediction[0][0]), font=font)
    s = prediction[0][0]
    frame = np.array(img_pill)
    # cv2.putText(frame, 'ය', (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    if s == "අ":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("අ")), font=font)
        frame = np.array(img_pill)
    elif s == "ම්":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("ම්")), font=font)
        frame = np.array(img_pill)
    elif s == "ආ":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("ආ")), font=font)
        frame = np.array(img_pill)
    elif s == "ඇ":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("ඇ")), font=font)
        frame = np.array(img_pill)
    elif s == "ව්":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("ව්")), font=font)
        frame = np.array(img_pill)
    elif s == "උ":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("උ")), font=font)
        frame = np.array(img_pill)
    elif s == "ක්":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 120), (str("ක්")), font=font)
        frame = np.array(img_pill)

    if w == "අම්ම්ආ":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 220), (str("අම්මා")), font=font)
        frame = np.array(img_pill)

    if w == "ක්ඇව්උම්":
        fontpath = "iskpota.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pill = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pill)
        draw.text((10, 220), (str("කැවුම්")), font=font)
        frame = np.array(img_pill)
    if n == "o":
            fontpath = "iskpota.ttf"
            font = ImageFont.truetype(fontpath, 32)
            img_pill = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pill)
            draw.text((10, 220), (str(" ")), font=font)
            frame = np.array(img_pill)

        # cv2.putText(frame, str("AMMA"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cv2.imshow("SL Sign Language Converter", frame)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        w = s
        print(s)

    if keypress == ord("c"):

        word = ""
        n = "o"
        # cv2.putText(frame, str("AMMA"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        print(str(s))


    elif keypress == 27:
        cap.release()
        cv2.destroyAllWindows()

        break


