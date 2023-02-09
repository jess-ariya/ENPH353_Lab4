#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self): #initiate the object
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self) #initiates timer object
        self._timer.timeout.connect(self.SLOT_query_camera) #connects to a callback func
        self._timer.setInterval(1000 / self._cam_fps) #sets the frequency


    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)


    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                        bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        _, frame = self._camera_device.read()
        #TODO run SIFT on the captured frame

        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

        #code here:
        img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)  # queryiamge

        # Features  
        sift = cv2.SIFT_create()
        #detect sift keypts and descriptors in the img
        kp_img, desc_img = sift.detectAndCompute(img, None)
        #img = cv2.drawKeypoints(img, kp_img, img)

        #Feature mapping
        index_params = dict(algorithm = 0, trees = 5) #make dictionary of the key value types
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params) 

        #read frame from camera
        _, frame = self._camera_device.read() #reads from the camera image
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #trainimage to gray scale
        #detect sift keypts and descriptors in the frame
        kp_grayframe, desc_grayframe = sift.detectAndCompute(gray_frame, None) #find matches in grayscale frame

        #find matches
        matches = flann.knnMatch(desc_img, desc_grayframe, k = 2)

        arr = [] #number of good matches it found, going through all the matches and finding which is valid
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                arr.append(m)

        
        img3 = cv2.drawMatches(img, kp_img, gray_frame, kp_grayframe, arr, gray_frame)

        #show
        cv2.imshow("img3", img3)

        #homography
        if len(arr) > 28:
            query_pts = np.float32([kp_img[m.queryIdx].pt for m in arr]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in arr]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            #perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            #show results
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)




        #code from chag gpt:
        # #match the keypts & decriptors between the image and frame 
        # bf = cv2.BFMatcher(cv2.NORM_L2, crosscheck = True)
        # matches = bf.match(desc_img, desc_grayframe)

        # #draw the matches on the frame
        # img_matches = cv2.drawMatches(img, kp_img, gray_frame, kp_grayframe, matches[:10], None, flags=2)
        # # Display the matches
        # cv2.imshow("Matches", img_matches)


        #gray_frame = cv2.drawKeypoints(gray_frame, kp_grayframe, gray_frame)


    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())