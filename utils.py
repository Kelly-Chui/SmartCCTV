import cv2
import sys
from avg_detect import main
import cv2
import numpy as np
import serial

Port = 'COM4'
Baudrate = 9600
arduino = serial.Serial(Port, Baudrate)

def camera_init(num=0, width=640, height=480):
    cap = cv2.VideoCapture(num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()
    ret, back = cap.read()
    if not ret:
        print('Background image registration failed!')
        sys.exit()
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    fback = back.astype(np.float32)
    return cap, fback

# def camera_read(cap, fback, saved_model_loaded, threshold=100):
#     img_li = []
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
#     cv2.accumulateWeighted(gray, fback, 0.01)
#     back = fback.astype(np.uint8)
#     diff = cv2.absdiff(gray, back)
#     _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)  # threshold를 어떻게 줄건지
#     if diff.any() > 0:
#         main(frame, saved_model_loaded)
#         img_li.append(frame)
#         cv2.imshow('frame', frame)
#     else:
#         grayf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         img_li.append(grayf)
#         cv2.imshow('frame', grayf)
#     return img_li

def camera_read(cap, fback, saved_model_loaded, name="frame", threshold=100):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    cv2.accumulateWeighted(gray, fback, 0.01)
    back = fback.astype(np.uint8)
    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)  # threshold를 어떻게 줄건지
    if diff.any() > 0:
        aa = main(frame, saved_model_loaded)
        cv2.imshow(name, frame)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(name, frame)
    return frame, aa

def video_writing(fps, img_li, cap, width, height, name=""):
    out = cv2.VideoWriter(f'./result/{name}.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    for i in range(0, len(img_li)):
        out.write(img_li[i])
    print("저장 완료")
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def trans_person(class_name=""):
    if class_name == "person":
        print(class_name)
        Trans = 'p'
        Trans = Trans.encode('utf-8')
        arduino.write(Trans)
    else:
        subTrans = 'n'
        subTrans = subTrans.encode('utf-8')
        arduino.write(subTrans)
