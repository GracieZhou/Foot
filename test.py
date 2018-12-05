# import the necessary packages
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

    # hmImg = cv.imread(os.path.join(indir,hmname))
    # hmpath = os.path.join(indir,hmname)
    # print hmpath
    # ret = fd.Sprehandle(hmImg, os.path.join(indir, f), oname = name+"_org.png")
    # if ret < 0:
    #     continue

    data = '20,23; 18,20; 20,20; 24,30; 21,26; 25,30'
    # ref = '1,0.78,200,100,0,0,0,0'
    ref = 'refp.txt'
    LArch, RArch, LQ, RQ = fd.getfootReportInfo(os.path.join(indir,hmname), os.path.join(indir, f), data, ref, oname = name+"_org.png")
    print LArch,RArch
    print (LQ), (RQ)

    Ref = fd.setRefPoints('line1.txt', 'line.png')
    print Ref

    # fd.drawrefimg('std.png')



