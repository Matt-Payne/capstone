import numpy as np
import cv2
import tkinter as tk
import PIL.Image, PIL.ImageTk
from pyardrone import ARDrone
import keyboard
import time
from imutils.object_detection import non_max_suppression
from main import ImageRec
from imutils.video import VideoStream
import imutils

drone = ARDrone()
drone.video_ready.wait()
# print(dir(drone.state))
# print(drone.navdata['demo']['battery'])

#Set up GUI
window = tk.Tk()
window.wm_title("Drone Cam")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

find = ""

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=3, column=2)
def show_frame():
    global find
    # frame = drone.frame
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # img = PIL.Image.fromarray(cv2image)
    # imgtk = PIL.ImageTk.PhotoImage(image=img)
    # lmain.imgtk = imgtk
    # lmain.configure(image=imgtk)
    # lmain.after(10, show_frame)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    frame = imutils.resize(drone.frame, width=720)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            if find and (CLASSES[idx] == find):
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = PIL.ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


# adapted from Adrian Rosebrock's article
def cropText(img_path):
    global find
    image = cv2.imread(img_path)
    orig = image.copy()
    (H, W) = image.shape[:2]
    rW = W / 320
    rH = H / 320
    image = cv2.resize(image, (320, 320))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

    # show the output image
    cropped = orig[(startY-20):(endY+10), (startX+10):(endX+10)]
    filename = "cropped-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
    cv2.imwrite(filename, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    i = ImageRec()
    # send cropped image to nn
    (word, prob) = i.main(filename)
    print(word)
    find = word
    # cv2.imshow("Text Detection", cropped)
    # cv2.waitKey(0)

def screenshot():
    filename = "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
    cv2.imwrite(filename, cv2.cvtColor(drone.frame, cv2.COLOR_RGB2BGR))
    cropText(filename)

# does not work
def movement():
    if not drone.state.fly_mask:
        # print("test")
        # drone.takeoff()
        # drone.hover()
        pass
    else:
        if keyboard.is_pressed('q'):
          drone.land()
        elif keyboard.is_pressed('w'):
          drone.move(forward=0.1)
        elif keyboard.is_pressed('s'):
          drone.move(backward=0.1)
        elif keyboard.is_pressed('a'):
          drone.move(left=0.1)
        elif keyboard.is_pressed('d'):
          drone.move(right=0.1)
        elif keyboard.is_pressed('e'):
          drone.move(up=0.1)
        elif keyboard.is_pressed('c'):
          drone.move(down=0.1)
        elif keyboard.is_pressed('z'):
          drone.move(ccw=0.1)
        elif keyboard.is_pressed('x'):
          drone.move(cw=0.1)
    window.after(1, movement)

btn_snapshot=tk.Button(window, text="Takeoff", width=25, command=movement)
btn_snapshot.grid(row = 2, column=0)
btn_snapshot=tk.Button(window, text="Snapshot", width=25, command=screenshot)
btn_snapshot.grid(row = 3, column=0)
show_frame()
window.after(1, movement)
window.mainloop()
drone.close()
