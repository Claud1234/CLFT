#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to manual labeling out the contour of car and human for rgb image.
'''
import cv2
import numpy as np
import argparse

# FPS
fps = 60
# First diagonal corner
first_diag_corner = None
second_diag_corner = None
rectangles = []

# Contour corners
contour_corn = []
contour_trailing = None
contours = []

classes = {
   'car': 1,
   'human': 2,
   'bg': 0
}

selection_method = {
   'rectangle': 100,
   'contour': 101
}

buttons_classes = {
   ord('1'): classes['car'],
   ord('2'): classes['human'],
   ord('0'): classes['bg']
}

buttons_selections = {
   ord('r'): selection_method['rectangle'],
   ord('c'): selection_method['contour']
}

class_colors = {
    classes['car']: (0, 255, 0),
    classes['human']: (0, 0, 255),
}

selected_class = classes['car']
selected_method = selection_method['contour']


def draw_rect(imgc, p1, p2, color):
    x1, y1 = p1
    x2, y2 = p2
    # cv2.rectangle(img_render,(x1,y1),(x2,y2),color,thikness)
    # print('(%d,%d),(%d,%d)' % (x1,y1,x2,y2))
    img_gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    nz_mask = img_gray > 0
    imgc[y1:y2, x1:x2][nz_mask[y1:y2, x1:x2]] = color
    return imgc


def draw_lines(imgc, cont, color, thinkness):
    for i in range(1, len(cont)):
        x0, y0 = cont[i-1]
        x1, y1 = cont[i]
        imgc = cv2.line(imgc, (x0, y0), (x1, y1), color, 1)
    return imgc


def draw_cont(imgc, cont, color):
    img_gray = np.zeros(imgc.shape[:2], np.uint8)
    img_gray = cv2.drawContours(
            img_gray, [np.asarray(cont).astype(np.int32).reshape(1, -1, 2)],
            0, 255, -1)
    cont_mask = img_gray > 0
    nz_mask = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY) > 0
    imgc[cont_mask*nz_mask] = color
    return imgc


def show_image_wait_input_draw(w_name, img, color=(0, 255, 0), thikness=1):
    '''Show image, wait for key pressed, or close button clicked
    @param img_name: Window name string
    @param img: Image ndarray
    '''
    global selected_class
    global selected_method
    global contour_corn
    global first_diag_corner
    global second_diag_corner
    global contours
    global rectangles

    kb_input_wait = int(fps**-1*1000)  # milliseconds

    def mouse_click_callback(event, x, y, flags, params):
        global first_diag_corner
        global second_diag_corner
        global contour_corn
        global contour_trailing
        global contours
        global rectangles
        if selected_method == selection_method['rectangle']:
            if event == cv2.EVENT_LBUTTONDOWN:
                if first_diag_corner is None:
                    first_diag_corner = (x, y)
                else:
                    # Collect the rectangle
                    rectangles.append((selected_class, (first_diag_corner,
                                                        (x, y))))
                    print('Collected rectangle %d: %s' % (selected_class,
                                                          (first_diag_corner,
                                                           (x, y))))
                    first_diag_corner = None
            if event == cv2.EVENT_MOUSEMOVE:
                second_diag_corner = (x, y)
        if selected_method == selection_method['contour']:
            if event == cv2.EVENT_LBUTTONDOWN:
                contour_corn.append((x, y))
            if event == cv2.EVENT_RBUTTONDOWN:
                contour_corn.append(contour_corn[0])
                contours.append((selected_class, contour_corn))
                print('Collected contour %d: %s' % (selected_class,
                                                    contour_corn))
                contour_corn = []
            if event == cv2.EVENT_MOUSEMOVE:
                contour_trailing = (x, y)

    img_render = img.copy()
    cv2.imshow(w_name, img_render)
    cv2.setMouseCallback(w_name, mouse_click_callback)
    # Window closing routines
    # print('Press any key to terminate ...')
    while True:
        # Wait for input
        char_code = cv2.waitKey(kb_input_wait)
        # Check if drwaing rectangle is needed
        img_render = img.copy()

        # Draw collected figures
        for clazz, cont in contours:
            img_render = draw_cont(img_render, cont, class_colors[clazz])
        for clazz, (p1, p2) in rectangles:
            img_render = draw_rect(img_render, p1, p2, class_colors[clazz])

        # Draw figures being collected
        if selected_method == selection_method['rectangle']:
            # Rectangle
            if not ((first_diag_corner is None) or
                    (second_diag_corner is None)):
                img_render = draw_rect(img_render, first_diag_corner,
                                       second_diag_corner,
                                       class_colors[selected_class])
        elif selected_method == selection_method['contour']:
            # Contour
            if len(contour_corn) >= 2:
                img_render = draw_lines(img_render, contour_corn,
                                        class_colors[selected_class], 1)
                img_render = cv2.line(img_render, contour_corn[-1],
                                      contour_trailing,
                                      class_colors[selected_class], 1)
        else:
            pass

        # Check if key was pressed
        if char_code in buttons_classes.keys():
            selected_class = buttons_classes[char_code]
            print('Pressed button %s, selected class %d' % (chr(char_code),
                                                            selected_class))
        if char_code in buttons_selections.keys():
            selected_method = buttons_selections[char_code]
            print('Pressed button %s, selection method %d' % (chr(char_code),
                                                              selected_method))
            if selected_method == selection_method['rectangle']:
                contour_corn = []
            if selected_method == selection_method['contour']:
                first_diag_corner = None
                second_diag_corner = None
#        else:
#         char_code != kb_input_none:
#            print('Key code: %d,char value: %s' % (char_code, chr(char_code)))
#            break

        # Check if window was closed clicking [x] close button
        if cv2.getWindowProperty(w_name, cv2.WND_PROP_VISIBLE) <= 0:
            print(cv2.getWindowProperty(w_name, cv2.WND_PROP_VISIBLE))
            print('Closing window by mouse click ...')
            break
        # Check if mouse was cklicked
        cv2.imshow(w_name, img_render)
        cv2.setMouseCallback(w_name, mouse_click_callback)
    anno_path = w_name.replace('rgb', 'annotation_rgb')
    cv2.imwrite(anno_path, img_render)
    cv2.destroyWindow(w_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lidar labeling tool')
    parser.add_argument('image', type=str, help='Path to lidar image file')
    parser.add_argument('-g', '--geometry', type=str, required=False,
                        help='Desired resolution', default=None)
    args = parser.parse_args()

    image_rgb = cv2.imread(args.image)

    if not (args.geometry is None):
        width, height = [int(v) for v in args.geometry.split('x')]
        print('Scaling down to %dx%d' % (width, height))

        img_rgb_scaled = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_CUBIC)
        image_rgb = img_rgb_scaled

    # (currently we will just show the original image in grayscale)
    show_image_wait_input_draw(args.image, image_rgb)
