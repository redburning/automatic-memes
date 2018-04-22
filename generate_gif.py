#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:23:49 2018

@author: weijie
"""

import dlib
from PIL import Image
import argparse
from imutils import face_utils
import numpy as np
import moviepy.editor as mpy


parser = argparse.ArgumentParser()
parser.add_argument("-image", required=True, help="path to input image")
args = parser.parse_args()

# resize to a max_width to keep gif size small
max_width = 500

# open our image, convert to rgba
img = Image.open(args.image).convert('RGBA')

# two images we'll need, glasses and deal with it text
glasses = Image.open("glasses.png")
cigarette = Image.open('cigarette.png')
text = Image.open('text.png')

if img.size[0] > max_width:
    scaled_height = int(max_width * img.size[1] / img.size[0])
    img.thumbnail((max_width, scaled_height))

# need grayscale for dlib face detection
img_gray = np.array(img.convert('L')) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
rects = detector(img_gray, 0)

if len(rects) == 0:
    print("No faces found, exiting.")
    exit()
print("%i faces found in source image. processing into gif now." % len(rects))

faces = []

for rect in rects:
    face = {}
    shades_width = rect.right() - rect.left()

    # predictor used to detect orientation in place where current face is
    shape = predictor(img_gray, rect)
    shape = face_utils.shape_to_np(shape)

    # grab the outlines of each eye from the input image
    leftEye = shape[36:42]
    rightEye = shape[42:48]

    # compute the center of mass for each eye
    leftEyeCenter = leftEye.mean(axis = 0).astype("int")
    rightEyeCenter = rightEye.mean(axis = 0).astype("int")
    
    #grab the outline of the mouth
    mouth = shape[48:68]
    mouth_width = shape[54][0] - shape[48][0]
    mouthCenter = mouth.mean(axis = 0).astype(int)

	# compute the angle between the eye centroids
    dY = leftEyeCenter[1] - rightEyeCenter[1] 
    dX = leftEyeCenter[0] - rightEyeCenter[0]
    angle = np.rad2deg(np.arctan2(dY, dX)) 

    # resize glasses to fit face width
    current_glasses = glasses.resize((shades_width, int(shades_width * glasses.size[1] / glasses.size[0])),
                               resample = Image.LANCZOS)
    # rotate(counterclockwise) and flip to fit eye centers
    current_glasses = current_glasses.rotate(180 - angle, expand = True)
    
    #resize cigarette to fit the mouth
    current_cigarette = cigarette.resize((mouth_width * 2, int(mouth_width * 2 * cigarette.size[1] / cigarette.size[0])),
                                         resample = Image.LANCZOS)
    # rotate and flip to fit mouth centers
    #current_cigarette = current_cigarette.rotate(angle, expand = True)
    
    # add the scaled image to a list, shift the final position to the left of the leftmost eye
    face['glasses_image'] = current_glasses
    face['cigarette_image'] = current_cigarette
    left_eye_x = leftEye[0,0] - shades_width // 6
    left_eye_y = leftEye[0,1] - shades_width // 6
    face['glasses_final_pos'] = (left_eye_x, left_eye_y)
    face['cigarette_final_pos'] = (mouthCenter[0], mouthCenter[1])
    faces.append(face)
 
duration = 4;
def make_frame(t):
    # returns copy of original image
    draw_img = img.convert('RGBA')
    if t == 0: # no glasses first image
        return np.asarray(draw_img)

    for face in faces:
        if t <= duration - 2:
            glasses_current_x = int(face['glasses_final_pos'][0])
            glasses_current_y = int(face['glasses_final_pos'][1] * t / (duration - 2))
            draw_img.paste(face['glasses_image'], (glasses_current_x, glasses_current_y) , face['glasses_image'])
            
            cigarette_current_x = int(face['cigarette_final_pos'][0] * t / (duration - 2))
            cigarette_current_y = face['cigarette_final_pos'][1]
            draw_img.paste(face['cigarette_image'], (cigarette_current_x, cigarette_current_y), face['cigarette_image'])
        else:
            draw_img.paste(face['glasses_image'], face['glasses_final_pos'], face['glasses_image'])
            draw_img.paste(face['cigarette_image'], face['cigarette_final_pos'], face['cigarette_image'])
            # The position of the text: 1/4 height from the bottom
            draw_img.paste(text, (75, draw_img.height // 4 * 3), text)

    return np.asarray(draw_img)

animation = mpy.VideoClip(make_frame, duration = duration)
animation.write_gif("deal.gif", fps = 4)

    
    
    
    
    