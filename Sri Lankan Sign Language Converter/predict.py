# import numpy as np
# from keras.models import model_from_json
# import operator
# import cv2
# from PIL import ImageFont,ImageDraw,Image
# import sys, os
#
# # Load the model
# json_file = open("model-bw.json", "r")
# model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(model_json)
# # load weights into new model
# loaded_model.load_weights("model-bw.h5")
# print("Loaded model from disk")
#
# cap = cv2.VideoCapture(0)
#
#
# pre = []
# s = ''
# num_frames = 0
#
#
# while True:
#     _, frame = cap.read()
#
#
#     x1 = int(0.5*frame.shape[1])
#     y1 = 10
#     x2 = frame.shape[1]-10
#     y2 = int(0.5*frame.shape[1])
#
#     cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
#
#     roi = frame[y1:y2, x1:x2]
#
#
#     roi = cv2.resize(roi, (64, 64))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     _, test_image = cv2.threshold(roi, 125, 255, cv2.THRESH_BINARY)
#     cv2.imshow("test", test_image)
#
#     result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
#     prediction = {'ම්': result[0][0],
#                   'එ': result[0][1],
#                   'ක්': result[0][2],
#                   'ව්': result[0][3],
#                   'ආ': result[0][4],
#                   'අ': result[0][5],
#                   'ඇ': result[0][6],
#                   'ඔ': result[0][7],
#                   'උ': result[0][8]}
#
#     prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
#
#     # Display the predictions
#     fontpath = "iskpota.ttf"
#     font = ImageFont.truetype(fontpath, 32)
#     img_pill = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img_pill)
#     draw.text((10, 120), (prediction[0][0]), font=font)
#     frame = np.array(img_pill)
#     # cv2.putText(frame, 'ය', (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
#     if s == "THREETWO":
#         fontpath = "iskpota.ttf"
#         font = ImageFont.truetype(fontpath, 32)
#         img_pill = Image.fromarray(frame)
#         draw = ImageDraw.Draw(img_pill)
#         draw.text((10, 220), (str(s)), font=font)
#
#         # cv2.putText(frame, str("AMMA"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
#
#     cv2.imshow("Frame", frame)
#     keypress = cv2.waitKey(1) & 0xFF
#
#
#     if keypress == ord("q"):
#
#         s = s + prediction[0][0]
#         print(s)
#
#
#     elif keypress == ord("c"):
#         s = ""
#
#         print(str(s))
#
#
#     elif keypress == 27:
#         cap.release()
#         cv2.destroyAllWindows()
#
#         break
#
# cv2.imshow("Frame", frame)
