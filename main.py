import cv2
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


if __name__ == '__main__':

    #img_filename = 'image.jpeg'
    
    #img_filename = 'video_print.png'

    #src = cv2.imread(img_filename)

    cap = cv2.videoCapture('videos/01_final/mkv')

    

    inpaintMask = create_mask(src)

    inpaintRadius = 1

    flags = cv2.INPAINT_TELEA #cv2.INPAINT_NS

    dst = cv2.inpaint( src, inpaintMask,inpaintRadius, flags)

    while True:
        cv2.imshow('original', src)
        cv2.imshow('result', dst)

        if cv2.waitKey(1) == ord('q'):
            break