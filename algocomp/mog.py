from template import *

cap = camera
fgbg = cv2.BackgroundSubtractorMOG()
help(cv2.BackgroundSubtractorMOG)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()