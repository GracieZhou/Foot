# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv
import footdetector as fd
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to the input image")
ap.add_argument("-o", "--output", required=True,
	help="path to the output image")

args = vars(ap.parse_args())

newpath = args["output"]
if not os.path.exists(newpath):
    os.makedirs(newpath)
fd.setodir(newpath)
indir = args["input"]
files = os.listdir(indir)
for f in files :
    print(f)
    if f.startswith('.') :
        continue
    img = cv.imread(os.path.join(indir,f))
    name = f.split('.',1)[0]
    #ret = fd.prehandle(img, name + "_prehandled.png")
    ret = fd.simpleprehandle(img, name + "_prehandled.png")
    if ret != 0 :
        continue
    img = fd.rimg(name + "_prehandled.png")
    ll, lw, rl, rw = fd.getFootWidthLength(img, oname = name + "_foot-width-length.png")
    print(ll, lw, rl, rw)
    if ll < lw or rl < rw :
        print("abnormal~" + f)
        continue
    img = fd.rimg(name+'_leftarea.png')
    ret = fd.footRorate(img, oname = name + "_leftarea-rotate.png")
    # x, y, w, h = fd.getRoiRect(img, oname = name + "_leftfoot-rect.png")
    # print(x, y, w, h)
    # x, y = fd.balanceCenter(img)
    # print(x,y)
    ret = fd.getFootArchIndex(img, oname=name+ "_leftfoot-archindex.png")
    print(ret)
    img = fd.rimg(name+'_rightarea.png')
    x, y, w, h = fd.getRoiRect(img, oname = name + "_rightfoot-rect.png")
    print(x, y, w, h)
    ret = fd.getFootArchIndex(img, oname=name+ "_rightfoot-archindex.png")
    print(ret)

