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
    if f.find(".txt") < 0:
        continue

    name = f.split('.',1)[0]

    hmname = name + ".png"

    hmImg = cv.imread(os.path.join(indir,hmname))
    # ret = fd.Sprehandle(hmImg, os.path.join(indir, f), oname = name+"_org.png")
    # if ret < 0:
    #     continue

    LArch, RArch = fd.getArch(hmImg, os.path.join(indir, f), oname = name+"_org.png")
    print LArch,RArch

    fd.getInnerPressLine(hmImg, os.path.join(indir, f), oname = name+"_org.png")

    data = np.array([[200,150],[205,160],[240,180],[230,190]])
    ret = fd.drawBalanceImg(hmImg, data, oname = name+"_balance.png")
    print ret

'''
    img = fd.rimg(name + "_org.png")
    fd.drawBalanceImg(hmImg, img, oname = name+"_ba
    
    
    lance.png")

    ## Left foot
    hmlImg = fd.rimg(name + "_left.png")
    img = fd.rimg(name+"_leftgray.png")
    La = fd.getRorate(img, oname = name + "_leftgray-rotate.png")
    print 'Angle:', La

    img = fd.rimg(name + "_leftgray-rotate.png")
    ret = fd.getLinesArch(hmlImg, img, 'L', La, oname=name+ "_leftfoot-arch.png")
    print 'Left Arch:', ret

    fd.PressLine(hmlImg, img, La, oname=name+ "_leftfoot-pressline.png")

    ## Right foot
    hmrImg = fd.rimg(name + "_right.png")
    img = fd.rimg(name + "_rightgray.png")
    Ra = fd.getRorate(img, oname=name + "_rightgray-rotate.png")
    print 'Angle:', Ra

    img = fd.rimg(name + "_rightgray-rotate.png")
    ret = fd.getLinesArch(hmrImg, img, 'R', Ra, oname=name+ "_rightfoot-arch.png")
    print 'Right Arch:', ret

    fd.PressLine(hmrImg, img, Ra, oname=name + "_rightfoot-pressline.png")
'''
