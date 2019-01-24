import cv2
from pyardrone import ARDrone
import keyboard

drone = ARDrone()
drone.video_ready.wait()
try:
  while True:
      cv2.imshow('im', drone.frame)
      if cv2.waitKey(10) == ord(' '):
          break
      if keyboard.is_pressed('return'):
          drone.takeoff()
          drone.hover()
      elif keyboard.is_pressed('q'):
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
finally:
    drone.close()
