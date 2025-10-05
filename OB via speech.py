import imutils
import cv2
import pyttsx3
import numpy as np
import urllib.request

net=cv2.dnn.readNetFromCaffe("mobilenet.prototxt","mobilenet.caffemodel")

classes=["background",      # ID 0
    "aeroplane",       # ID 1
    "bicycle",         # ID 2
    "bird",            # ID 3
    "boat",            # ID 4
    "bottle",          # ID 5
    "bus",             # ID 6
    "car",             # ID 7
    "cat",             # ID 8
    "chair",           # ID 9
    "cow",             # ID 10
    "diningtable",     # ID 11
    "dog",             # ID 12
    "horse",           # ID 13
    "motorbike",       # ID 14
    "person",          # ID 15
    "pottedplant",     # ID 16
    "sheep",           # ID 17
    "sofa",            # ID 18
    "train",           # ID 19
    "tvmonitor"        # ID 20 
    ]
engine=pyttsx3.init()
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

url='http://192.0.0.4:8080/shot.jpg'

while True:
    ret,Frame=cam.read()
    imgPath=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    img=imutils.resize(img,width=450)
    if not ret:
        break

    h,w=Frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(Frame,(300,300)),0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()

    Detected=set()

    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]

        if confidence > 0.5:
            idx=int(detections[0,0,i,1])
            if idx< len(classes):
                label=classes[idx]
                print(f"Detected:{label} ({confidence:.2f})")
                Detected.add(label)

                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype("int")
                label=f"{classes[idx]}:{confidence*100:.1f}%"
                cv2.rectangle(Frame,(startX,startY),(endX,endY),(0,255,0),2)
                cv2.putText(Frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    if Detected:
        text="I See "+" , " .join(Detected)
        print(text)
        engine.say(text)
        engine.runAndWait()

    cv2.imshow("Video",Frame)
    if cv2.waitKey(10)==27:
        break

cam.release()
cv2.destroyAllWindows()

