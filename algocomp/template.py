import cv2
import sys

cam_source = sys.argv[1]


try:
	cam_source=int(cam_source)
except:
	pass

camera = cv2.VideoCapture(cam_source)

for x in range(20):
	camera.read()


if __name__ == "__main__":
	while 1:
		status,frame=camera.read()
		cv2.imshow("window name",frame)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

	camera.release()
	cv2.destroyAllWindows()
