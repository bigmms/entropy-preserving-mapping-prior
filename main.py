import cv2
import os
import utils


rootdir = "./imgs/"
theta0 = 1.3945
theta1 = 9.1377
theta2 = 0.5
extension = ".jpg"

if __name__ == '__main__':
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            f,ext = os.path.splitext(rootdir+filename) # Split filename and type
            if ext == extension:
                print "process image... %s" % rootdir+filename
                img = cv2.imread(rootdir+filename)
                imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                imgH, imgS, imgV = cv2.split(imgHSV)
                T = utils.getMappingCurve(imgV, theta0, theta1, theta2)
                imgHSV[:, :, 2] = utils.getTransform(imgV, T) * 255
                imgRes = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
                cv2.imwrite('./results/' + filename[:-4] + '_EPMP.jpg', imgRes)
