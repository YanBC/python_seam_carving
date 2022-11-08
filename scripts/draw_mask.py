'''
Created on Apr 3, 2016

@author: Bill BEGUERADJ

copy-pasted from https://stackoverflow.com/a/36382158/14221921
with minor modification
'''
import cv2 as cv
import numpy as np
import argparse

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
current_former_x = -1
current_former_y = -1
image_width = -1
image_height = -1
image_channel = 3
image = None
canvas = None

# mouse callback function
def begueradj_draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode, image, canvas

    if event==cv.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event==cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.line(image,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
                cv.line(canvas,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv.line(image,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
            cv.line(canvas,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y


def user_interface():
    global image
    cv.namedWindow("Bill BEGUERADJ OpenCV")
    cv.setMouseCallback('Bill BEGUERADJ OpenCV', begueradj_draw)
    print("press ESC to exit ...")
    while(1):
        cv.imshow('Bill BEGUERADJ OpenCV', image)
        k=cv.waitKey(1)&0xFF
        if k==27:
            break
    cv.destroyAllWindows()


def convex_mask() -> np.ndarray:
    global canvas
    mask = canvas[:, :, -1]
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    local_canvas = np.zeros_like(canvas, dtype=np.uint8)
    for cnt in contours:
        hull = cv.convexHull(cnt)
        cv.drawContours(local_canvas, [hull], 0, (0,0,255), -1)
    return local_canvas[:, :, -1]


def show_contour(contour: np.ndarray):
    global image_height, image_width, image_channel
    local_canvas = np.zeros((image_height, image_width, image_channel), dtype=np.uint8)
    cv.drawContours(local_canvas, [contour], 0, (0,255,0), 3)
    cv.imshow("local_canvas", local_canvas)
    cv.waitKey()
    cv.destroyWindow("local_canvas")


parser = argparse.ArgumentParser()
parser.add_argument("image", help="path to image")
args = parser.parse_args()
image_path = args.image

image = cv.imread(image_path)
image_height, image_width, image_channel = image.shape
assert image_channel == 3
canvas = np.zeros((image_height, image_width, image_channel), dtype=np.uint8)
user_interface()

full_mask = convex_mask()
print("saved mask image to mask.jpg")
cv.imwrite("mask.jpg", full_mask)
