from PIL import Image
from PIL import ImageTk
from pyardrone import ARDrone
import tkinter as tk
import threading
import datetime
import imutils
import cv2
import os
import keyboard
import numpy as np
from imutils.object_detection import non_max_suppression
from main import ImageRec
import imutils
import time
from difflib import SequenceMatcher

class DroneCam:
    def __init__(self):
        self.drone = ARDrone()
        self.drone.video_ready.wait()
        self.outputPath = "../output"
        self.frame = None
        self.video_thread = None
        self.control_thread = None
        self.text_rec_thread = None
        self.stopEvent = None
        self.find = "bicycle"       # object current being looked for
        self.img_path = ""
        self.focal_length = 548     # focal length of drone forward camera
        self.distance = 0           # current distance from object in inches
        self.land_distance = 24     # inches away from object to land
        self.ratio = 0.5            # ratio to match words

        self.root = tk.Tk()
        self.panel = None

        btn = tk.Button(self.root, text="screenshot", command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.SIZES = [0, 0, 66, 0, 0,
            2.5, 0, 0, 0, 18, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            45, 0, 0] # corresponding sizes of objects in array CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

        self.stopEvent = threading.Event()
        self.video_thread = threading.Thread(target=self.videoLoop, args=())
        self.video_thread.start()
        self.control_thread = threading.Thread(target=self.inputLoop, args=())
        # self.control_thread = threading.Thread(target=self.navigate, args=())
        self.control_thread.start()

        self.root.wm_title("Drone Cam")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def getPixels(self):
        return 0

    def initFocalLength(self, pixels, distance, width):
        focal = (pixels*distance)/width
        return focal

    def findDistance(self, focal, width, pixels):
        distance = (width * focal)/pixels
        return distance

    def drawBoxes(self):
        #draw circle in center of object
        screenWidth = 640
        screenHeight = 360
        testBoxX = int(screenWidth / 3)
        testBoxY = int(screenHeight / 3)
        #top left and bottom right of box
        TL_test = (0,0)
        BR_test = (testBoxX, testBoxY)
        #draw box
        cv2.rectangle(self.frame,TL_test,BR_test,(20,20,255),3)

        TL_M_Box=(testBoxX,0)
        BR_M_Box=(int(screenWidth*(2/3)),testBoxY)
        #top middle box
        cv2.rectangle(self.frame,TL_M_Box,BR_M_Box,(255,20,20),3)
        #top right box
        TL_R_Box = (int(640*(2/3)),0)
        BR_R_Box = (640,int(screenHeight/3))
        cv2.rectangle(self.frame,TL_R_Box,BR_R_Box,(0,255,255),3)

        #left middle box
        TL_L_M_Box = (0,int(screenHeight/3))
        BR_L_M_Box = (int(640/3),int(screenHeight*(2/3)))
        cv2.rectangle(self.frame,TL_L_M_Box,BR_L_M_Box,(0,0,0),3)

        #middle middle box
        TL_M_M_Box = (int(640/3),360/3)
        BR_M_M_Box = (int(640*(2/3)),360*(2/3))
        cv2.rectangle(self.frame,TL_L_M_Box,BR_L_M_Box,(51,204,51),3)

        #middle right box
        TL_R_M_Box = (int(640*(2/3)),360/3)
        BR_R_M_Box = (640,360*(2/3))
        cv2.rectangle(self.frame,TL_L_M_Box,BR_L_M_Box,(255, 153, 51),3)


        #bottom left box
        TL_L_L_Box = (int(360*(2/3)),0)
        BR_L_L_Box = (360,int(screenWidth*(2/3)))
        cv2.rectangle(self.frame,TL_L_L_Box,BR_L_L_Box,(102, 0, 51),3)
    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                self.frame = self.drone.frame
                self.blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(self.blob)
                (h, w) = self.frame.shape[:2]
                detections = self.net.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.2:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                        if self.find and SequenceMatcher(self.CLASSES[idx],self.find).ratio > self.ratio:
                            cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                            self.distance = self.findDistance(self.focal_length, self.SIZES[idx], abs(startX-endX))

                            #draw circle in center of object
                            screenWidth = 640
                            screenHeight = 360
                            #get middle coords of object box
                            x = int((startX+endX)/2)
                            y = int((startY+endY)/2)

                            self.drawBoxes()
                            #draw circle in the center of object
                            cv2.circle(self.frame, (x,y), 5, (75,13,180), -1)

                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(self.frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")


    def navigate(self):
        self.drone.takeoff()
        self.drone.hover()
        while not self.stopEvent.is_set():
            if self.distance == 0:
                self.drone.move(cw=0.1)
            if self.distance > 24:
                self.drone.move(forward=0.1)
            if self.distance < 24 and self.distance != 0:
                self.drone.land()


    def inputLoop(self):
        try:
            while not self.stopEvent.is_set():
                if keyboard.is_pressed('q'):
                    self.drone.land()
                elif keyboard.is_pressed('p'):
                    self.drone.takeoff()
                    self.drone.hover()
                elif keyboard.is_pressed('w'):
                    self.drone.move(forward=0.1)
                elif keyboard.is_pressed('s'):
                    self.drone.move(backward=0.1)
                elif keyboard.is_pressed('a'):
                    self.drone.move(left=0.1)
                elif keyboard.is_pressed('d'):
                    self.drone.move(right=0.1)
                elif keyboard.is_pressed('e'):
                    self.drone.move(up=0.1)
                elif keyboard.is_pressed('c'):
                    self.drone.move(down=0.1)
                elif keyboard.is_pressed('z'):
                    self.drone.move(ccw=0.1)
                elif keyboard.is_pressed('x'):
                    self.drone.move(cw=0.1)
        except RuntimeError:
            print("[INFO] caught a runtime error")

    # adapted from Adrian Rosebrock's article
    def cropText(self):
        image = cv2.imread(self.img_path)
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

        self.startY = 0
        self.endY   = 0
        self.startX = 0
        self.endX   = 0

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
                self.endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                self.endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                self.startX = int(self.endX - w)
                self.startY = int(self.endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((self.startX, self.startY, self.endX, self.endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            self.startX = int(startX * rW)
            self.startY = int(startY * rH)
            self.endX = int(endX * rW)
            self.endY = int(endY * rH)

        # show the output image
        cropped = orig[(self.startY):(self.endY), (self.startX):(self.endX)]
        filename = "cropped-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
        cv2.imwrite(filename, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        i = ImageRec()
        # send cropped image to nn
        (word, prob) = i.main(filename)
        self.find = word
        print(self.find)
        # cv2.imshow("Text Detection", cropped)
        # cv2.waitKey(0)

    def takeSnapshot(self):
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(p))
        self.img_path = p
        self.text_rec_thread = threading.Thread(target=self.cropText, args=())
        self.text_rec_thread.start()

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.drone._close()
        self.root.quit()



dc = DroneCam()
dc.root.mainloop()
