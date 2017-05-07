import numpy as np
import cv2,datetime

breakflag=0

while 1:
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    try:
        import requests,json
        try:
            data=requests.get("http://api.geonames.org/timezoneJSON?lat=20&lng=78&username=demo")
            fname=json.loads(data.text)
            fname="store/"+fname['time'] + ".avi"
        except Exception as e:
            print "Exception in requests",e
    except:
        fname="store/" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M")+'local.avi'

    print "New video name...",fname
    out = cv2.VideoWriter(fname,fourcc, 20.0, (640,480))

    try:
        past = datetime.datetime.now()
        present = datetime.datetime.now()

        while(cap.isOpened() and present-past < datetime.timedelta(seconds=10)):
            present = datetime.datetime.now()
            ret, frame = cap.read()
            if ret==True:
                #frame = cv2.flip(frame,0)

                # write the flipped frame
                out.write(frame)

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    breakflag=1
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if breakflag:break
    except:
        pass