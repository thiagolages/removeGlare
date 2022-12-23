from imutils.video import WebcamVideoStream
import logging
import threading
import math
import cv2
import cv2.aruco as aruco
import numpy as np


'''
Class PrecisionLanding.

'''
class PrecisionLanding:

    # class contructor
    def __init__(self, altitude, device):
        self.kill = False
        self.device = device
        self.altitude = altitude
        #self.vehicle = vehicle

        # aruco original dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()

        # camera resolution and fov
        self.horizontal_res = 800
        self.vertical_res = 600

        # calculate hfov and vfov
        #https://stackoverflow.com/questions/25088543/estimate-visible-bounds-of-webcam-using-diagonal-fov
        #logitech c270 specs only show 55 degrees of diagonal fov

        # self.horizontal_fov = 48.81 * (math.pi / 180)  # Pi cam V1: 53.5  V2: 62.2
        # self.vertical_fov = 28.63 * (math.pi / 180)  # Pi cam V1: 41.41 V2: 48.8
        self.horizontal_fov = 62.2 * (math.pi / 180)  # Pi cam V1: 53.5  V2: 62.2
        self.vertical_fov = 48.8 * (math.pi / 180)  # Pi cam V1: 41.41 V2: 48.8

        # REQUIRED: Calibration files for camera
        # Look up https://github.com/dronedojo/video2calibration for info
        self.calib_path = "/home/thiago/projects/cloud-control-vision/matrix/logitech_800_600/"
        self.cameraMatrix = np.loadtxt(self.calib_path + 'cameraMatrix.txt', delimiter = ',')
        self.cameraDistortion = np.loadtxt(self.calib_path + 'cameraDistortion.txt', delimiter = ',')

        self.counter = 0


    # image processing for tracking aruco markers
    def trackArucos(self, frame):
        frame_np = None

        try:
            counter = 0
            ids = None
            marker_to_find = 0

            # convert image to grayscale
            f = cv2.resize(frame, (self.horizontal_res, self.vertical_res))
            frame_np = np.array(f)
            gray_img = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)        

            # aruco basic marker detection
            corners, ids, rejected = aruco.detectMarkers(image=gray_img, dictionary=self.aruco_dict, parameters=self.aruco_params)
            #altitude = self.vehicle.rangefinder
            #altitude = 12

            # if altitude between 18 and 6 meters, then look for marker 72
            if self.altitude > 6.0 and self.altitude <= 18.0:
                marker_to_find = 72
                marker_size = 40

            # if altitude <= 6 meters, then look for marker 62
            elif self.altitude <= 6.0:
                marker_to_find = 0
                if ids is not None:
                    for id in ids:
                        if id == 72:
                            pass
                        elif id > marker_to_find:
                            marker_to_find = id
                marker_size = 15            
            
            # loop for found ids
            if ids is not None:
                for id in ids:
                    if id == marker_to_find:
                        corners_single = [corners[counter]]
                        corners_single_np = np.asarray(corners_single)

                        # process pose estimation at this time
                        ret = aruco.estimatePoseSingleMarkers(corners_single, marker_size, cameraMatrix=self.cameraMatrix, distCoeffs=self.cameraDistortion)
                        rvec,tvec = ret[0][0,0,:], ret[1][0,0,:]

                        y_sum = 0
                        x_sum = 0

                        x_sum = corners_single_np[0][0][0][0] + corners_single_np[0][0][1][0] + corners_single_np[0][0][2][0] + corners_single_np[0][0][3][0]
                        y_sum = corners_single_np[0][0][0][1] + corners_single_np[0][0][1][1] + corners_single_np[0][0][2][1] + corners_single_np[0][0][3][1]

                        x_avg = x_sum * .25
                        y_avg = y_sum * .25

                        x_ang = (x_avg - self.horizontal_res * .5) * (self.horizontal_fov / self.horizontal_res)
                        y_ang = (y_avg - self.vertical_res * .5) * (self.vertical_fov / self.vertical_res)

                        # draw detected markers and display axes
                        aruco.drawDetectedMarkers(frame_np, corners_single)
                        aruco.drawAxis(frame_np, self.cameraMatrix, self.cameraDistortion, rvec, tvec, 10)

                        # send landing target message for precision landing
                        #self.vehicle.send_landing_target_message(x_ang, y_ang)

                        #print("PRECISION_LANDING_ACQUIRED id [%s] x [%.6f] y [%.6f]" % (id, x_ang, y_ang))
                        #logging.info("PRECISION_LANDING_ACQUIRED id [%s] x [%.6f] y [%.6f]" % (id, x_ang, y_ang))                        

                    counter = counter + 1

        except Exception as e:
            print("[ARUCO] Error: " + str(e))
            logging.error("[ARUCO] Error: %s", str(e))

        # return frame
        if frame_np is not None:
            return frame_np
        else:
            return frame


    # this is the target precision landing thread function
    def processLanding(self):
        cap = None
        while True:
            if self.kill:
                break
            try:
                # start capturing video stream
                if cap is None:
                    cap = WebcamVideoStream(self.device, cv2.CAP_GSTREAMER).start()                

                # detect and track aruco markers
                frame = self.trackArucos( cap.read() )

                # show window with video for screen sharing
                resizedFrame = cv2.resize(frame, (320, 240))
                cv2.imshow('frame', resizedFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            except Exception as e:
                print(e)    
                logging.info(e)
                self.kill = True


    # if true the thread loop will be finished
    def stopThread(self):
        self.kill = True


    # start tracking arucos on a separate thread
    def startTracking(self):
        print("[ARUCO] Thread start")
        logging.info("[ARUCO] Thread start")
        threading.Thread( target=self.processLanding ).start()

