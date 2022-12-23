from aruco_v2 import PrecisionLanding

import cv2, sys
import numpy as np
from skimage import measure

def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, 180, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label( thresh_img, neighbors=8, background=0 )
    mask = np.zeros( thresh_img.shape, dtype="uint8" )
    # loop over the unique components
    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask


def assert_exit(condition, err_message):
    try:
        assert condition
    except AssertionError:
        sys.exit(err_message)

if __name__ == '__main__':

    assert_exit(len(sys.argv) >= 2, "Wrong number of parameters. Usage: python main.py <image/video/cam> <image or video name>")
    
    input_option = sys.argv[1] # video, image or cam

    if (input_option == "video" or input_option == "image"):
        assert_exit(len(sys.argv) == 3, "You must specify image or video filename. Usage: python main.py <image/video/cam> <image or video name>")
        filename = sys.argv[2]
        cap = cv2.VideoCapture(filename)
        #cap = cv2.VideoCapture('videos/02_final.mkv')
    else:
        cap = cv2.VideoCapture(2)
    
    aruco = PrecisionLanding(12, 4) # altitude, cameraID (wont be used)

    loopFlag = True

    while loopFlag:

        if (input_option == "image"):
            frame = cv2.imread(filename)
        else:
            loopFlag = cap.isOpened()
            ret, frame = cap.read()
            
            # if frame is read correctly ret is True
            if not ret:
                #print("Can't receive frame (stream end?). Exiting ...")
                break

        #frame = cv2.imread("frames/frame2755.jpg")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = aruco.trackArucos(frame)
        
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #result1 = clahe.apply(gray)
        
        #result1 = aruco.trackArucos(result1)
        #result1 = cv2.cvtColor(result1, cv2.COLOR_GRAY2BGR)

        # inpaintMask = create_mask(src)

        # inpaintRadius = 1

        # flags = cv2.INPAINT_TELEA #cv2.INPAINT_NS

        # dst = cv2.inpaint( src, inpaintMask,inpaintRadius, flags)


        cv2.imshow('original', frame)
        #cv2.imshow('CLAHE', result1)

        if cv2.waitKey(1) == ord('q'):
            loopFlag = False
            break