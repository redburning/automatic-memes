# DEAL WITH IT in Python with Face Detection

This is the companion repository for the original post at [makeartwithpython.com](https://www.makeartwithpython.com/blog/deal-with-it-generator-face-recognition/).

It generates the DEAL WITH IT MEME by detecting faces in images.

![DEAL WITH IT](images/deal.gif?raw=true)

## Architecture

![MEME ARCHITECTURE](https://github.com/burningion/automatic-memes/blob/master/images/meme_generator_architecture.png?raw=true)

## Requirements 

Pillow, MoviePy, and NumPy for the Gif from still image generator, and OpenCV and Pillow for the real time DEAL generator. 

You'll need a webcam to get real time video in OpenCV to work.


## Usage

You'll need to download the [shape_predictor_68](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) from dlib-models and unzip it in this directory first.

After, you should be able to just pass in the location of that predictor to the Python3 program as a command line argument like so:

```bash
$ python generate_gif.py -image = 'beautiful_girl.png'
```

Make sure your image has front facing faces, otherwise the program will exit immediately.
