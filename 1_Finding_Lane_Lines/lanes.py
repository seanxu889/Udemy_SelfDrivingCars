#!/anaconda2/envs/test_py3/bin/python3.6
# coding=utf-8
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # when using 'matplotlib' in python3
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    # indicate the coordinates to plot
    slope, intercept = line_parameters
    #print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) # 左右两边两条车道线从最底部开始，向上显示到图片垂直高度3/5处
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1) # will fit a first degree polynomial, get the parameter of a linear function
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    #print(left_fit)
    #print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # the average slope and y-intercept
    #print(left_fit_average, 'left')
    #print(right_fit_average, 'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

def canny(image):
    # 1.convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2.reduce noise
    # using Gaussian blur to reduce noise in th egray scale image
    blur = cv2.GaussianBlur(gray, (5,5), 0) # apply this blur in our gray scale image
    # with 5*5 kernal, with the deviation = 0

    # 3.using Canny method to identify edges
    # 'cv2.Canny(image,low_threshold,high_threshold)'
    canny = cv2.Canny(blur, 50, 150)

    return canny

def display_lines(image, lines):
# 6.define a function to display in our real images
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) # display in a black image, and then blend to our original image

    return line_image

# 4.identify lane lines
# find the region of interest,
# create an image that is black - a mask with the same dimensions as our image.
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]]) # make sure only one polygons
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # computing the bitwise &
    # use the polygons image to mask our canny image, and only showing the ROI traced by the triangle。
    # use bitwise_and 'AND' operation: 白色三角形区域内的像素点二进制值为11111111，那么11111111与任何值的二进制值
    # 进行AND操作都不会改变原有的像素值；而白色三角形区域外的黑色区域为0000，此时0000余任何值AND操作都将变成0000.
    # ROI will traced by the polygonal contour.
    return masked_image

# 5.use the Hough transform technique to detect straight lines in ROI (霍夫变换)
# 笛卡尔坐标系中的一个点，在Hough Space中是一条线；一条线，一个点(m,b代表笛卡尔坐标系中的斜率和截距)
# 但是由于不能表示水平线和垂直线，便使用极坐标和Hough Space相对应。

'''
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) # important to make a 'np.copy', not just using '='
# otherwise, the chenges will reflected to the original image
canny_image = canny(lane_image)
#plt.imshow(canny)
#plt.show()
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # specify
# a precision of two pixels by a 1 degree precision, 100 is the min number of intersections in space for a bin
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # blend, take the sum of our color image with our lane image
cv2.imshow('result', combo_image)
cv2.waitKey(0)
'''

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read() # decode every video frame
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # blend, take the sum of our color image with our lane image
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
