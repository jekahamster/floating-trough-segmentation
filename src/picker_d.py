import sys
import os
import pathlib
import numpy as np
import cv2 
import json

from typing import List

ROOT_DIR = pathlib.Path(__file__).parent.absolute()
COLOR_MODE = "BGR"

image = None
controls_win = "Controls"
hue_low = 0
hue_high = 179
saturation_low = 0
saturation_high = 255
value_low = 0
value_high = 255
erosion_kernel = 1
erosion_iterations = 0
dilation_kernel = 1
dilation_iterations = 0
opening_kernel = 1
closing_kernel = 1
opening_iterations = 0
closing_iterations = 0
min_contour_len = 0
min_contour_area = 0
delta_h = 5
delta_s = 5
delta_v = 5

def update_trackbars():
    global image
    global controls_win
    global hue_low, hue_high
    global saturation_low, saturation_high
    global value_low, value_high
    global erosion_kernel, erosion_iterations
    global dilation_kernel, dilation_iterations
    global opening_kernel
    global closing_kernel
    global opening_iterations
    global closing_iterations
    global min_contour_len
    global min_contour_area
    global delta_h, delta_s, delta_v

    cv2.setTrackbarPos("H low", controls_win, hue_low)
    cv2.setTrackbarPos("H high", controls_win, hue_high)
    cv2.setTrackbarPos("S low", controls_win, saturation_low)
    cv2.setTrackbarPos("S high", controls_win, saturation_high)
    cv2.setTrackbarPos("V low", controls_win, value_low)
    cv2.setTrackbarPos("V high", controls_win, value_high)
    cv2.setTrackbarPos("Erosion Kernel", controls_win, erosion_kernel)
    cv2.setTrackbarPos("Erosion Iterations", controls_win, erosion_iterations)
    cv2.setTrackbarPos("Dilation Kernel", controls_win, dilation_kernel)
    cv2.setTrackbarPos("Dilation Iterations", controls_win, dilation_iterations)
    cv2.setTrackbarPos("Opening Kernel", controls_win, opening_kernel)
    cv2.setTrackbarPos("Closing Kernel", controls_win, closing_kernel)
    cv2.setTrackbarPos("Opening Iterations", controls_win, opening_iterations)
    cv2.setTrackbarPos("Closing Iterations", controls_win, closing_iterations)
    cv2.setTrackbarPos("Delta h", controls_win, delta_h)
    cv2.setTrackbarPos("Delta s", controls_win, delta_s)
    cv2.setTrackbarPos("Delta v", controls_win, delta_v)
    cv2.setTrackbarPos("Min contour len", controls_win, min_contour_len)
    cv2.setTrackbarPos("Min contour area", controls_win, min_contour_area)
    

def orig_callback(event, x, y, flags, param):
    global image
    global controls_win
    global hue_low
    global hue_high
    global saturation_low
    global saturation_high
    global value_low
    global value_high
    global erosion_kernel
    global erosion_iterations
    global dilation_kernel
    global dilation_iterations
    global opening_kernel
    global closing_kernel
    global min_contour_len
    global min_contour_area
    global opening_iterations
    global closing_iterations
    global delta_h
    global delta_s
    global delta_v
    
    
    if event != cv2.EVENT_LBUTTONUP:
        return
    
    if COLOR_MODE == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h, s, v = hsv_image[y][x]
    
    hue_low = max(0, h - delta_h)
    hue_high = min(179, h + delta_h)
    saturation_low = max(0, s - delta_s)
    saturation_high = min(255, s + delta_s)
    value_low = max(0, v - delta_v)
    value_high = min(255, v + delta_v)
    opening_kernel = 2
    opening_iterations = 2
    min_contour_area = 45

    update_trackbars()



def nothing(x):
    pass


def proportional_resize(image, max_size=500):
    h, w, *_ = image.shape
    coeff = max_size / max(h, w)
    image = cv2.resize(image, (int(coeff*w), int(coeff*h)))
    return image


def draw_contours(image, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        # (x,y,w,h) = cv2.boundingRect(contour)
        # cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    return image


def prepare_mask(image):
    global erosion_kernel, erosion_iterations
    global dilation_kernel, dilation_iterations
    global opening_kernel
    global closing_kernel
    global min_contour_len
    global min_contour_area
    global closing_iterations, opening_iterations

    image = cv2.erode(
        image, 
        np.ones((erosion_kernel, erosion_kernel), dtype=np.uint8),
        iterations=erosion_iterations
    ) 

    image = cv2.dilate(
        image, 
        np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8),
        iterations=dilation_iterations
    ) 

    image = cv2.morphologyEx(
        image, 
        cv2.MORPH_OPEN, 
        np.ones((opening_kernel, opening_kernel), dtype=np.uint8),
        iterations=opening_iterations
    )

    image = cv2.morphologyEx(
        image, 
        cv2.MORPH_CLOSE,
        np.ones((closing_kernel, closing_kernel), dtype=np.uint8),
        iterations=closing_iterations
    )

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda c: cv2.arcLength(c, False) > min_contour_len, contours))
    contours = list(filter(lambda c: cv2.contourArea(c) > min_contour_area, contours))
    
    contour_mask = np.zeros((image.shape[:2]), dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, cv2.FILLED)  
    image = cv2.bitwise_and(image, image, mask=contour_mask)

    return image


def picker(images_paths_list: List[pathlib.Path]):
    global image
    global controls_win
    global hue_low, hue_high
    global saturation_low, saturation_high
    global value_low, value_high
    global erosion_kernel, erosion_iterations
    global dilation_kernel, dilation_iterations
    global opening_kernel
    global closing_kernel
    global opening_iterations
    global closing_iterations
    global min_contour_len
    global min_contour_area
    global delta_h, delta_s, delta_v
    
    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", orig_callback)

    cv2.namedWindow(controls_win)
    # cv2.resizeWindow(controls_win, 700, 900)
    cv2.resizeWindow(controls_win, 300, 500)

    cv2.createTrackbar("H low", controls_win, 0, 179, nothing)
    cv2.createTrackbar("H high", controls_win, 179, 179, nothing)
    cv2.createTrackbar("S low", controls_win, 0, 255, nothing)
    cv2.createTrackbar("S high", controls_win, 255, 255, nothing)
    cv2.createTrackbar("V low", controls_win, 0, 255, nothing)
    cv2.createTrackbar("V high", controls_win, 255, 255, nothing)

    cv2.createTrackbar("Erosion Kernel", controls_win, 1, 20, nothing)
    cv2.createTrackbar("Erosion Iterations", controls_win, 0, 100, nothing)
    cv2.createTrackbar("Dilation Kernel", controls_win, 1, 20, nothing)
    cv2.createTrackbar("Dilation Iterations", controls_win, 0, 100, nothing)
    cv2.createTrackbar("Opening Kernel", controls_win, 1, 20, nothing)
    cv2.createTrackbar("Closing Kernel", controls_win, 1, 20, nothing)
    cv2.createTrackbar("Opening Iterations", controls_win, 1, 100, nothing)
    cv2.createTrackbar("Closing Iterations", controls_win, 1, 100, nothing)
    cv2.createTrackbar("Delta h", controls_win, 0, 179, nothing)
    cv2.createTrackbar("Delta s", controls_win, 0, 255, nothing)
    cv2.createTrackbar("Delta v", controls_win, 0, 277, nothing)
    cv2.createTrackbar("Min contour len", controls_win, 0, 200, nothing)
    cv2.createTrackbar("Min contour area", controls_win, 0, 200, nothing)


    cv2.setTrackbarPos("Delta h", controls_win, delta_h)
    cv2.setTrackbarPos("Delta s", controls_win, delta_s)
    cv2.setTrackbarPos("Delta v", controls_win, delta_v)

    show_more_images = False


    image_index = 0
    while True:
        image = cv2.imread(images_paths_list[image_index])
        assert image is not None, f"Invalid image path: {images_paths_list[image_index]}"

        if COLOR_MODE == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hue_low = cv2.getTrackbarPos("H low", controls_win)
        hue_high = cv2.getTrackbarPos("H high", controls_win)
        saturation_low = cv2.getTrackbarPos("S low", controls_win)
        saturation_high = cv2.getTrackbarPos("S high", controls_win)
        value_low = cv2.getTrackbarPos("V low", controls_win)
        value_high = cv2.getTrackbarPos("V high", controls_win)

        erosion_kernel = cv2.getTrackbarPos("Erosion Kernel", controls_win)
        erosion_iterations = cv2.getTrackbarPos("Erosion Iterations", controls_win)
        dilation_kernel = cv2.getTrackbarPos("Dilation Kernel", controls_win)
        dilation_iterations = cv2.getTrackbarPos("Dilation Iterations", controls_win)
        opening_kernel = cv2.getTrackbarPos("Opening Kernel", controls_win)
        closing_kernel = cv2.getTrackbarPos("Closing Kernel", controls_win)
        opening_iterations = cv2.getTrackbarPos("Opening Iterations", controls_win)
        closing_iterations = cv2.getTrackbarPos("Closing Iterations", controls_win)
        min_contour_len = cv2.getTrackbarPos("Min contour len", controls_win)
        min_contour_area = cv2.getTrackbarPos("Min contour area", controls_win)
        delta_h = cv2.getTrackbarPos("Delta h", controls_win)
        delta_s = cv2.getTrackbarPos("Delta s", controls_win)
        delta_v = cv2.getTrackbarPos("Delta v", controls_win)

        hsv_low = np.array([hue_low, saturation_low, value_low], np.uint8)
        hsv_high = np.array([hue_high, saturation_high, value_high], np.uint8)

        mask = cv2.inRange(hsv_image, hsv_low, hsv_high)

        mask = prepare_mask(mask)        

        res = cv2.bitwise_and(image, image, mask=mask)
        mask_ch3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        stacked = np.hstack([
            proportional_resize(image), 
            proportional_resize(res), 
            proportional_resize(mask_ch3)
        ])
        
        cv2.imshow(controls_win, stacked)
        cv2.imshow("Original", image)
        additional_images = {
            "Result": res,
            "Mask": mask,
            "Contours": draw_contours(image, mask)
        }
        if show_more_images:
            for window_name, image in additional_images.items():
                image = cv2.resize(image, None, fx=0.85, fy=0.85)
                cv2.imshow(window_name, image)
        else:
            try:
                for window_name in additional_images.keys():
                    cv2.destroyWindow(window_name)
            except:
                pass

        key = cv2.waitKey(1)   
        
        # Key q - quit
        if key == ord("q"):
            break
        
        # left arrow - previous image
        if key == 49:
            image_index = max(image_index-1, 0)

        # right arrow - next image
        if key == 50:
            image_index = min(image_index+1, len(images_paths_list)-1)

        # w - write HSV params
        elif key == ord("w"):
            print(f"H: {hue_low} - {hue_high}")
            print(f"S: {saturation_low} - {saturation_high}")
            print(f"V: {value_low} - {value_high}")
            print()

        # m - show more images
        elif key == ord("m"):
            show_more_images = not show_more_images

        # c - print count of contours
        elif key == ord("c"):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(len(contours))
        
        # s - save parameters to json
        elif key == ord("s"):
            print("Saving...")
            with open(ROOT_DIR / "params.json", "w", encoding="utf-8") as file:
                json.dump({
                    "hue_low": hue_low, 
                    "hue_high": hue_high, 
                    "saturation_low": saturation_low, 
                    "saturation_high": saturation_high, 
                    "value_low": value_low, 
                    "value_high": value_high, 
                    "erosion_kernel": erosion_kernel, 
                    "erosion_iterations": erosion_iterations, 
                    "dilation_kernel": dilation_kernel, 
                    "dilation_iterations": dilation_iterations, 
                    "opening_kernel": opening_kernel, 
                    "closing_kernel": closing_kernel, 
                    "min_contour_len": min_contour_len, 
                    "min_contour_area": min_contour_area 
                }, file, indent=2)

        elif key == "r":
            pass


    cv2.destroyAllWindows()


def main(args):
    # first parameter - init image path
    if len(args) > 1:
        image_path = args[1]
    else:
        image_path = input("Image path: ").strip()
        
    images_list = [image_path]
    
    # second parameter - directory with images
    if len(args) > 2:
        images_list.extend([os.path.join(args[2], file) for file in os.listdir(args[2])])

    picker(images_list)


if __name__ == "__main__":
    main(sys.argv)
