
import cv2 as cv2
import autopy as autopy
class takepicture:
    def __init__(self):
        print("take class initialized")
    def tookpicture(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        
        while True:
            success,img = cap.read()
            cv2.imshow('web cam',img)
            if cv2.waitKey(1)==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


