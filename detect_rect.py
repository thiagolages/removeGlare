import cv2
import numpy as np
from aruco_v2 import PrecisionLanding

# medida retangulo aruco: 90x60 cm (ratio = )

cap = cv2.VideoCapture("videos/testeDLV2_trimmed.avi")
#cap = cv2.VideoCapture("videos/05_final.mkv")

write = False

altitude = 12
landing = PrecisionLanding(altitude=altitude, device=None)

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

new_size = (800,600)

#size = (frame_width, frame_height)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_detect_rect_and_aruco.avi', fourcc, 30.0, new_size)


def drawContours(img, contours):
    count = 0
    stopFlag = False
    x_center, y_center, center_count = 0, 0 ,0

    while not stopFlag:
        if (count > len(contours) - 1):
            stopFlag = True
            break

        cnt = contours[count]
        count = count + 1
        #for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        #print(cv2.contourArea(cnt), len(approx))
        cnt_area = cv2.contourArea(cnt)
        sides = len(approx)
        if ((sides >= 4 and sides <= 8) and (cnt_area > 100 and cnt_area < 5500)):
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = round(float(w)/h,3)
            
            if ratio >= 0.650 and ratio <= 1.15:
                approx = np.squeeze(approx)
                x_sum, y_sum = 0, 0
                for corners in approx:
                    x_sum += corners[0]
                    y_sum += corners[1]

                x_avg = int(x_sum/sides)
                y_avg = int(y_sum/sides)
                
                x_center = x_center + x_avg
                y_center = y_center + y_avg
                center_count = center_count + 1

                #print("sides = {}, ratio = {}, area = {}".format(sides, ratio, cnt_area))
                #cv2.putText(img, 'Target', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(img, str(ratio)+'-'+str(sides), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # draw contour
                img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)                    
                # draw center
                #cv2.circle(img, center, radius=1, color=(0, 0, 255), thickness=5)

    if (center_count != 0):
        x_center = int(x_center/center_count)
        y_center = int(y_center/center_count)
        center = (x_center, y_center)
        cv2.circle(img, center, radius=1, color=(0, 0, 255), thickness=5)


def pipeline(img, sufix, erode=False, show_thresh=False):
    # aruco
    img, id, x_ang, y_ang = landing.trackArucos(img)
    
    # threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)

    if erode:
        # erode
        kernel = np.ones((5, 5), np.uint8)
        thresh_erode = cv2.erode(thresh, kernel) 
    else:
        # Using cv2.erode() method 
        thresh_erode = thresh

    # show threshold
    if show_thresh:
        cv2.imshow("thresh_"+str(sufix), thresh_erode)

    # find contours
    contours,hierarchy = cv2.findContours(thresh_erode, 1, 2)

    #cv2.imshow("img_"+str(sufix), img)

    return img, contours

def numberOfEdges(contour):
    return len(cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True))

while cap.isOpened():

    ret,img = cap.read()
    if not ret:
        print("Not ret, closing..")
        break

    # resize
    # (int(img.shape[0]/2), int(img.shape[1]/2))
    img = cv2.resize(img, new_size)
    # cv2.imshow("original", img)

    # Smoothing image
    img_smooth = cv2.GaussianBlur(img,(5,5),0)
    # # Sharpening image
    # kernel = np.array([ [-1, -1, -1], 
    #                     [-1,  9, -1], 
    #                     [-1, -1, -1]])
    # img_sharp = cv2.filter2D(img_smooth,-1,kernel)

    # original img
    img, contours_ori  = pipeline(img, "original", erode=False)

    #img smooth
    #img_smooth, contours_smooth = pipeline(img_smooth, "smooth", erode=False)

    # show contours
    drawContours(img, contours_ori)
    #drawContours(img_smooth,contours_smooth)


    if write:
        out.write(img)
    
    
    cv2.imshow("contours ori", img)
    #cv2.imshow("contours smooth", img_smooth)
    
    #cv2.waitKey(0)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()