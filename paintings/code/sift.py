import cv2

img = cv2.imread('../data/Paintings/two_class/realism/butterfly.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.SURF()
kp = surf.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('surf_keypoints.jpg',img)


