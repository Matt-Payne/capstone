import numpy as np
import cv2
import tkinter as tk
import PIL.Image, PIL.ImageTk
from pyardrone import ARDrone
import keyboard

drone = ARDrone()
drone.video_ready.wait()

#Set up GUI
window = tk.Tk()
window.wm_title("Drone Cam")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
def show_frame():
    frame = drone.frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = PIL.ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI
# try:
#   while True:
#       print('hello')
#       window.update()
#       if keyboard.is_pressed('return'):
#           drone.takeoff()
#           drone.hover()
#       elif keyboard.is_pressed('q'):
#           drone.land()
#       elif keyboard.is_pressed('w'):
#           drone.move(forward=0.1)
#       elif keyboard.is_pressed('s'):
#           drone.move(backward=0.1)
#       elif keyboard.is_pressed('a'):
#           drone.move(left=0.1)
#       elif keyboard.is_pressed('d'):
#           drone.move(right=0.1)
#       elif keyboard.is_pressed('e'):
#           drone.move(up=0.1)
#       elif keyboard.is_pressed('c'):
#           drone.move(down=0.1)
#       elif keyboard.is_pressed('z'):
#           drone.move(ccw=0.1)
#       elif keyboard.is_pressed('x'):
#           drone.move(cw=0.1)
# finally:
#     drone.close()
