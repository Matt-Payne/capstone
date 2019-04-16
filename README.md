# capstone
## list of main files
`ocr/SimpleHTR/src/dronecam.py` - script that creates gui to interface with drone, contains methods to:
  - `findDistance`  = takes parameters focal, width, pixels
  - `videoLoop`     = updates frames from drone and processes image
  - `navigate`      = algorithm to navigate drone towards object
  - `cropText`      = crops text from screenshot to send to neural net for HTR
`ocr/SimpleHTR/src/main.py` - hand written text recognition
### controls
  - `q`           = emergency land
  - `screenshot`  = button starts process of reading text and finding object

# requirements
`pyardrone 0.6.1` [docs](https://media.readthedocs.org/pdf/pyardrone/latest/pyardrone.pdf) [pip](https://pypi.org/project/pyardrone/)

`keyboard 0.13.2` [docs](https://github.com/boppreh/keyboard#api) [pip](https://pypi.org/project/keyboard/)

`opencv 3.4.3.18` [docs](https://docs.opencv.org/master/) [pip](https://pypi.org/project/opencv-python/)

`Tkinter` [docs](https://docs.python.org/3/library/tkinter.html) [pip](https://wiki.python.org/moin/TkInter)

`TensorFlow 1.12` [docs](https://www.tensorflow.org/api_docs) [pip](https://pypi.org/project/tensorflow/)
