# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv
import sys
import math
import numpy_indexed as npi
import os
from PIL import Image
import matplotlib.pyplot as plt

pixelsPerMetric = None
odir = '.'
pixelMinValue = 1
def oimg(name,img):
    global odir
    if not os.path.exists(odir):
        os.makedirs(odir)
    #os.chdir(odir)
    #cv.imwrite(name,img)
    cv.imwrite(os.path.join(odir,name),img)

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def rimg(name):
    #pwd = os.getpwd()
    #global odir
    #if pwd != odir :
    #    os.chdir(odir)
    fname=os.path.join(odir,name)
    img = cv.imread(fname)
    #os.chdir()
    return img

def setodir(od):
    global odir
    odir= od

def dshow(img):
    cv.namedWindow('debug',cv.WINDOW_NORMAL)
    cv.imshow('debug',img)
    cv.waitKey()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def drawLines(lines,oimg):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(oimg, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

def drawLinesP(linesP,oimg):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(oimg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 5, cv.LINE_AA)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    y,u,v = cv.split(yuv)
    #img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    #for gray in cv.split(img):
    _retval, bin = cv.threshold(y, 0, 255, cv.THRESH_OTSU)
    binn = cv.bitwise_not(bin)
    c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opened = cv.morphologyEx(binn, cv.MORPH_OPEN, c)
    bin, contours, _hierarchy = cv.findContours(opened, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea,reverse=True)
    topN = 3
    maxLoop = topN if len(contours) >= topN else len(contours)
    for i in range(0,maxLoop):
        cnt = contours[i]
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in list(range(4))])
            if max_cos < 0.1:
                squares.append(cnt)
    squares = npi.unique(squares)
    return squares

def rectArea(square):
    p0 = (square[0][0],square[0][1])
    p1 = (square[1][0],square[1][1])
    p2 = (square[2][0],square[2][1])
    return dist.euclidean(p0,p1) * dist.euclidean(p1,p2)

def getCircledRoi(origfile,ofile):
    img = rimg(origfile)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, c)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 10, 35)
    if len(circles) < 1 :
        print('abnormal:ROI area error!')
        return -1
    cs = sorted(circles[0], key=lambda x: (x[1], x[0]))
    k = float(cs[3][1] - cs[0][1])/(cs[3][0] - cs[0][0])
    #use PIL
    pil_im = Image.open(os.path.join(odir,origfile))
    #chang to angle
    pil_im = pil_im.rotate(math.atan(k)*180/math.pi)
    rotedfileName = origfile.split('_',-1)[0] +  '_cicleRotated.png'
    pil_im.save(os.path.join(odir, rotedfileName))
    img = rimg(rotedfileName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, c)
    circles2 = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 10, 35)
    rads = [x[2] for x in circles2[0]]
    radmean = np.mean(rads)
    xs = [x[0] for x in circles2[0]]
    minX = int(np.min(xs) - radmean)
    maxX = int(np.max(xs) + radmean)
    ys = [x[1] for x in circles2[0]]
    minY = int(np.min(ys) - radmean)
    maxY = int(np.max(ys) + radmean)
    roi = img[minY:maxY,minX:maxX]
    w,h = roi.shape[0:2]
    roi = cv.resize(roi,(900,int(w*900/h)))
    oimg(ofile,roi)
    return 0


def prehandle(img,oname):
    orig = img.copy()
    ss=find_squares(img)
    if len(ss) < 1 :
        print("abnormal:no square found!")
        return -1
    else :
        for i in range(0,1) :
            orig = cv.drawContours(orig,ss,i,(0,0,255),5)
        oimg('square.png',orig)
        square = ss[0]
        minX = min([v[0] for v in square])
        maxX = max([v[0] for v in square])
        minY = min([v[1] for v in square])
        maxY = max([v[1] for v in square])
    areaRoi = img[minY:maxY,minX:maxX]
    w,h = areaRoi.shape[0:2]
    areaRoi = cv.resize(areaRoi,(1000,int(w*1000/h)))
    #cut off the device header
    areaRoi = areaRoi[130:1000,:]
    tmpname = oname.split('_',-1)[0] + '_cuted.png'
    oimg(tmpname,areaRoi)
    ret = getCircledRoi(tmpname, oname)
    return ret

def getPoints(start,end):
    ret=list()
    if end[0] != start[0] :
        k = float((float(end[1]) - start[1]) / (end[0] - start[0]))
        if abs(end[1] - start[1]) >= abs(end[0] - start[0]) :
            step = 1 if end[1] > start[1] else -1
            for y in range(start[1] , end[1] + step, step):
                x = int((float(y)-start[1])/k + start[0])
                ret.append((x,y))
        else :
            step = 1 if end[0] > start[0] else -1
            for x in range(start[0] , end[0] + step, step):
                y = int(k * (x-start[0]) + start[1])
                ret.append((x,y))
    else :
        step = 1 if end[0] > start[0] else -1
        for y in range(start[1] , end[1] + step, step):
            x = start[0]
            ret.append((x,y))
    return ret

def getPixsPerMetrics(img, metics):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 10, 30)
    #save image
    cimg = img.copy()
    if circles is not None :
        maxLen = len(circles[0]) if len(circles[0]) <= 8 else 8
        for i in range(maxLen) :
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 1, cv.LINE_AA)
    oimg('circles.png',cimg)
    total = 0
    for circle in circles[0] :
        total += circle[2]
    average = total / len(circles[0])

    return average * 2 / metics


def getCrossPoint(start,end,cnt):
    k = (float(end[1]) - start[1]) / (end[0] - start[0])
    if abs(k) <= 1 :
        if(end[0] >= start[0]) :
            x = end[0] + 1
            y = k * (x-end[0]) + end[1]
            flag = cv.pointPolygonTest(cnt,(x,y),False)
            x = x + 1
            y = k * (x-end[0]) + end[1]
            while(cv.pointPolygonTest(cnt,(x,y),False) == flag and x <= (max(cnt[:,:,0]) + 1)) :
                x = x + 1
                y = k * (x - end[0]) + end[1]
        else :
            x = end[0] - 1
            y = k * (x-end[0]) + end[1]
            flag = cv.pointPolygonTest(cnt,(x,y),False)
            x = x - 1
            y = k * (x-end[0]) + end[1]
            while(cv.pointPolygonTest(cnt,(x,y),False) == flag and x >= 0) :
                x = x - 1
                y = k * (x - end[0]) + end[1]
    else :
        if(end[1] >= start[1]) :
            y = end[1] + 1
            x = (y - end[1])/k + end[0]
            flag = cv.pointPolygonTest(cnt,(x,y),False)
            y = y + 1
            x = (y - end[1])/k + end[0]
            while(cv.pointPolygonTest(cnt,(x,y),False) == flag and y <= (max(cnt[:,:,1]) + 1 )) :
                y = y + 1
                x = (y - end[1])/k + end[0]
        else :
            y = end[1] - 1
            x = (y - end[1])/k + end[0]
            flag = cv.pointPolygonTest(cnt,(x,y),False)
            y = y - 1
            x = (y - end[1])/k + end[0]
            while(cv.pointPolygonTest(cnt,(x,y),False) == flag and y >= 0 ) :
                y = y - 1
                x = (y - end[1])/k + end[0]
    (x,y) = (int(x),int(y))
    return (x,y),cv.pointPolygonTest(cnt,(x,y),True)

def getYCbCr(img,oname='xxx.png'):
    yCrCb = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)
    yC,Cr,Cb = cv.split(yCrCb)
    ret, yCt = cv.threshold(yC, 0, 255, cv.THRESH_OTSU)
    ret, Crt = cv.threshold(Cr, 0, 255, cv.THRESH_OTSU)
    ret, Cbt = cv.threshold(Cb, 0, 255, cv.THRESH_OTSU)
    oimg('yct.png',yCt)
    oimg('crt.png',Crt)
    oimg('cbt.png',Cbt)
    Cbtm = np.mean(Cbt)/255
    if Cbtm < 0.4 :
        opp = Cbt
    elif Cbtm > 0.8 :
        opp = cv.bitwise_not(Crt)
    else :
        CbCrm = (Cb + Cr) / 2
        #the result is not ok for most green picture
        #CbCrmh = cv.equalizeHist(CbCrm)
        #CbCrmh = imadjust(CbCrm, CbCrm.min(), CbCrm.max(),0,1)
        CbCrm = CbCrm.astype(np.uint8)
        ret, opp = cv.threshold(CbCrm, 0, 255, cv.THRESH_OTSU)
        #opp = cv.bitwise_not(opp)
        #opp=imfill(opp)
    oimg(oname,opp)
    return opp

def drawWidthLength(cnt,orig):
    box = cv.minAreaRect(cnt)
    points = cv.boxPoints(box)
    ar = np.array(points, dtype="int")
    ar = perspective.order_points(ar)
    orig = cv.drawContours(orig, [ar.astype("int")], -1, (0, 255, 0), 2)
    for (x, y) in points:
        cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    (tl, tr, br, bl) = ar
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 2)
    cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 2)
    # length
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    # width
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    global pixelsPerMetric
    if pixelsPerMetric is None:
        pixelsPerMetric = getPixsPerMetrics(orig, 1)
    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
    cv.putText(orig, "{:.1f}cm".format(dimA),
               (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (255, 255, 255), 2)
    cv.putText(orig, "{:.1f}cm".format(dimB),
               (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (255, 255, 255), 2)
    return dimA,dimB


def getWidthLength(img,orig):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _retval, bin = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opened = cv.morphologyEx(bin, cv.MORPH_OPEN, c)
    nz = np.nonzero(opened)
    nzt = np.transpose(np.asarray(nz))
    # x, y  exchange
    nztx = [[x[1],x[0]] for x in nzt]
    h, w = img.shape[0:2]
    minX=minY=limitSize=40
    #left foot handle
    nztxv = [x for x in nztx if x[0] > minX and x[0] < (w/2 - limitSize) and x[1] >minY  and x[1] < h-limitSize]
    ll,lw=drawWidthLength(np.asarray(nztxv),orig)
    #right foot handle
    nztxv = [x for x in nztx if x[0] > (w/2 + limitSize) and x[0] < (w - limitSize) and x[1] >minY  and x[1] < h-limitSize]
    rl,rw=drawWidthLength(np.asarray(nztxv),orig)
    return ll,lw,rl,rw

# return width length
def getFootWidthLength(img,oname='foot-width-length.png'):
    orig = img.copy()
    h, w = img.shape[0:2]
    oimg(oname.split('.')[0].split('_',-1)[0] + '_leftarea.png',img[:,0:int(w/2)])
    oimg(oname.split('.')[0].split('_',-1)[0] + '_rightarea.png',img[:,w/2:])
    opp = getYCbCr(img,oname.split('.')[0] + "_opp.png")
    img2 = rimg(oname.split('.')[0] + "_opp.png")
    ll,lw,rl,rw = getWidthLength(img2,orig)
    oimg(oname,orig)
    return ll, lw, rl, rw


def imfill(img):
    # Copy the thresholded image.
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = img | im_floodfill_inv
    return im_out

def getFootArchIndex(img,odir='.',oname='archindex.png'):
    orig = img.copy()
    opp = getYCbCr(img,oname.split('.')[0] + "_opp.png")

    image, contours, hierarchy = cv.findContours(opp.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea,reverse=True)
    cnt=contours[0]

    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)
    if defects is None :
        print('abnormal: there is no defects!')
        return -1
    defects = sorted(defects, key=lambda x:x[0][3],reverse=True)
    s, e, f, d = defects[0][0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.circle(orig, start, 5, (0,0,255), -1)
    cv.circle(orig, end, 5, (0,0,255), -1)
    cv.circle(orig, far, 5, (0,0,255), -1)
    cv.line(orig, start, end, (0, 255, 0), 2)
    minDist = sys.maxsize
    minPoint = (0,0)
    points = getPoints(start,end)
    for p in points :
        dim = dist.euclidean(p,far)
        if dim <=  minDist:
            minDist = dim
            minPoint = p
    B=far
    A=minPoint
    C,_ = getCrossPoint(A,B,cnt)
    cv.circle(orig, A, 5, (0,0,255), -1)
    cv.circle(orig, C, 5, (0,0,255), -1)
    cv.line(orig,A,C,(255,255,0),2)
    cv.putText(orig, "A",
                A, cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv.putText(orig, "B",
                B, cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv.putText(orig, "C",
                C, cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    archIndex = dist.euclidean(A,B)/dist.euclidean(B,C)
    cv.putText(orig, "{:.2f}".format(archIndex),
                (int(B[0]), int(B[1]+25)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    oimg(oname,orig)

    return archIndex

def Sprehandle(hmimg,datafile,oname = 'org.png'):
    ret = SConvPressImg(datafile, oname)
    if ret < 0:
        return ret
    img = rimg(oname)
    if img is None:
        print 'the org image is None.', oname
        return -1

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if hmimg is None:
        print 'the heatmap image is None.'
        return -1

    hmh,hmw = hmimg.shape[0:2]
    h, w = img.shape[0:2]
    if abs(hmh-h)>5 or abs(hmw-w)>5:
        print("the heatmapImg size is error")
        return -1

    oimg(oname.split('_',-1)[0] + '_left.png', hmimg[:, 0:int(w/2)])
    oimg(oname.split('_',-1)[0] + '_right.png', hmimg[:, int(w/2):])

    oimg(oname.split('_',-1)[0] + '_leftgray.png', img[:, 0:int(w/2)])
    oimg(oname.split('_',-1)[0] + '_rightgray.png', img[:, int(w/2):])
    return 0

def getRorate(img, oname):
    name = oname[0 : oname.rfind('-')]
    rectName = name + '_Rect.png'
    x,y,w,h = getRoiRect(img, rectName)
    print(x,y,w,h)

    d = np.rint(h/6)

    # determine the up and down lines
    US = int(y+d)
    UE = int(y+3*d)
    DS = int(y+4*d)
    DE = int(y+h)

    Ux,Uy = getBalanceCenter(img[US:UE, :])
    Dx,Dy = getBalanceCenter(img[DS:DE, :])

    ## draw the up and down BC points
    # orig = img.copy()
    # cv.circle(orig, (int(Ux), int(Uy+US)), 5, (255, 0, 0), -1)
    # cv.circle(orig, (int(Dx), int(Dy+DS)), 5, (255, 0, 0), -1)
    # rectName = oname.split('-', -1)[0] + '_BC.png'
    # oimg(rectName, orig)
    # print('----up point----')
    # print(Ux, Uy+US)
    # print('----Down point----')
    # print(Dx, Dy+DS)

    # Height is X axis, and Width is Y axis for Img
    k = float(Dx-Ux)/(Dy-Uy+3*d)
    angle = math.atan(k)
    degree = angle*180/math.pi

    #use PIL
    igrayfile = name + '.png'
    pil_im = Image.open(os.path.join(odir, igrayfile))

    #rotate the angle
    if np.abs(degree) > 1:
        pil_im = pil_im.rotate(-degree)

    pil_im.save(os.path.join(odir, oname))

    return angle

def getRoiRect(img, oname):
    orig = img.copy()
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    nz = np.nonzero(img)
    nzt = np.transpose(np.asarray(nz))
    # x, y  exchange
    nztx = [[x[1],x[0]] for x in nzt]

    # get rect
    x,y,w,h = cv.boundingRect(np.asarray(nztx))
    imgrect = cv.rectangle(orig, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
    # oimg(oname, imgrect)
    return x,y,w,h

def getBalanceCenter(img):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[0:2]
    # print(h,w)
    SumP = np.sum(img)
    SumW = np.array(np.sum(img, axis=0))
    SumH = np.array(np.sum(img, axis=1))
    Hpress = WPress = 0
    # print(SumH.shape)
    # print(SumW.shape)

    for i in range(0,w):
        WPress = WPress + i*SumW[i]

    for j in range(0,h):
        Hpress = Hpress + j*SumH[j]

    x = y = 0
    if SumP > 0:
        x = int(WPress/SumP)
        y = int(Hpress/SumP)

    return x,y

def getPointAffinedPos(inpoint, center, angle):
    outpoint = np.zeros((2))
    x = inpoint[0] - center[0]
    y = inpoint[1] - center[1]

    outpoint[0] = int(x * math.cos(angle) + y * math.sin(angle) + center[0])
    outpoint[1] = int(-x * math.sin(angle) + y * math.cos(angle) + center[1])
    return outpoint

def getMaxLine(img, Hstart, Hend, Wstart, Wend, step):
    ## find the longest line of img
    global pixelMinValue
    LineNumMax = 0
    H = WL = WR = 0

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for i in range(Hstart, Hend, step):
        Linemin = max(Wstart, Wend)
        Linemax = min(Wstart, Wend)
        Linenum = 0
        for j in range(Wstart, Wend):
            if img[i][j] > pixelMinValue:
                Linenum = Linenum + 1
                if j < Linemin:
                    Linemin = j

                if j > Linemax:
                    Linemax = j

        if Linenum > LineNumMax:
            LineNumMax = Linenum
            H = i
            WL = Linemin
            WR = Linemax

    return H,WL,WR


def getMinLine(img, Hstart, Hend, Wstart, Wend):
    ## find the shortest line of img
    global pixelMinValue
    LineNumMin = Wend - Wstart
    H = WL = WR = 0

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for i in range(Hstart, Hend):
        Linemin = max(Wstart, Wend)
        Linemax = min(Wstart, Wend)
        Linenum = 0
        for j in range(Wstart, Wend):
            if img[i][j] > pixelMinValue:
                Linenum = Linenum+1
                if j < Linemin:
                    Linemin = j

                if j > Linemax:
                    Linemax = j

        if Linenum > 0:
            if Linenum < LineNumMin:
                LineNumMin = Linenum
                H = i
                WL = Linemin
                WR = Linemax

    return H,WL,WR


def getLinesArch(hmimg,img,whichone,angle,oname):
    orig = hmimg.copy()

    ## Arch is [0,1,2,3,4], 2 is normal, >2 is high arch, <2 is low arch
    Arch = 0
    H,W = img.shape[0:2]
    center = np.array((int(W/2), int(H/2)))
    print "img(H,W):",H,W
    # set the line start and end H axis
    DiffHW = int(abs(H-W)/2)
    Ths = DiffHW + 1
    The = H - DiffHW - 1

    rectName = oname.split('_', -1)[0] + '_Rect.png'
    x,y,w,h = getRoiRect(img, rectName)
    print(x,y,w,h)
    Ws = x
    Hs = y
    We = x+w-5
    He = y+h-5
    Hdeep = int(h/3)
    Wc = int(x+w/2)
    print Hs,He,Ws,We,Hdeep,Wc

    Uh, Uwl, Uwr = getMaxLine(img, Hs, Hs+Hdeep, Ws, We, 1)
    Dh, Dwl, Dwr = getMaxLine(img, He, He-Hdeep, Ws, We, -1)
    # print "Up: ",Uh, Uwl, Uwr
    # print "Down: ", Dh, Dwl, Dwr
    Upoint = Dpoint = np.zeros((2))
    if 'L' == whichone:
        Upoint = np.array([Uwr, Uh])
        Dpoint = np.array([Dwr, Dh])
    else:
        Upoint = np.array([Uwl, Uh])
        Dpoint = np.array([Dwl, Dh])
    # print "Upoint: ", Upoint
    # print "Dpoint: ", Dpoint

    # The tangent slope, H is X axis, and W is Y axis
    k1 = float(Dpoint[0]-Upoint[0])/float(Dpoint[1]-Upoint[1])
    print "k1: ", k1

    # Transform the points to Heat map img
    TUpoint = getPointAffinedPos(Upoint, center, angle)
    TDpoint = getPointAffinedPos(Dpoint, center, angle)
    TWcs = getPointAffinedPos((Wc, Ths), center, angle)
    TWce = getPointAffinedPos((Wc, The), center, angle)

    ## the k1 line and Wc line is parallel
    if k1 == 0:
        cv.line(orig, (TUpoint[0], TUpoint[1]), (TDpoint[0], TDpoint[1]), (255, 0, 255), 1, cv.LINE_AA)
        cv.line(orig, (TWcs[0], TWcs[1]), (TWce[0], TWce[1]), (255, 0, 255), 1, cv.LINE_AA)
        oimg(oname, orig)
        return Arch

    # get the cross point of k1 and W center line
    CrossP = np.array([Wc, (Wc-Dpoint[0])/k1+Dpoint[1]])

    # get the lines slope
    rad = math.atan(k1)
    Qrad = float(rad)/4
    k2 = math.tan(2 * Qrad)
    k3 = math.tan(Qrad)
    print "k2: ", k2
    print "k3: ", k3

    ## Transform the lines start and end points to heat map img
    Tk1s = getPointAffinedPos((int(k1 * (Ths - Dpoint[1]) + Dpoint[0]), Ths), center, angle)
    Tk1e = getPointAffinedPos((int(k1 * (The - Dpoint[1]) + Dpoint[0]), The), center, angle)

    Tk2s = getPointAffinedPos((int(k2 * (Ths - CrossP[1]) + CrossP[0]), Ths), center, angle)
    Tk2e = getPointAffinedPos((int(k2 * (The - CrossP[1]) + CrossP[0]), The), center, angle)

    Tk3s = getPointAffinedPos((int(k3 * (Ths - CrossP[1]) + CrossP[0]), Ths), center, angle)
    Tk3e = getPointAffinedPos((int(k3 * (The - CrossP[1]) + CrossP[0]), The), center, angle)

    ## get lines start point and end point
    lines = list()
    lines.append((int(Tk1s[0]), int(Tk1s[1]), int(Tk1e[0]), int(Tk1e[1])))
    lines.append((int(Tk2s[0]), int(Tk2s[1]), int(Tk2e[0]), int(Tk2e[1])))
    lines.append((int(Tk3s[0]), int(Tk3s[1]), int(Tk3e[0]), int(Tk3e[1])))
    lines.append((int(TWcs[0]), int(TWcs[1]), int(TWce[0]), int(TWce[1])))
    print("========lines=======")
    print(lines)

    cv.circle(orig, (int(TUpoint[0]), int(TUpoint[1])), 2, (255, 0, 0), -1)
    cv.circle(orig, (int(TDpoint[0]), int(TDpoint[1])), 2, (255, 0, 0), -1)
    for k in range(len(lines)):
        cv.line(orig, (lines[k][0], lines[k][1]), (lines[k][2], lines[k][3]), (255, 0, 255), 1, cv.LINE_AA)

    oimg(oname, orig)

    ## get arch line
    Ah,Awl,Awr = getMinLine(img, Hs+Hdeep, He-Hdeep, Ws, We)
    if Awl == -1 and Awr == -1:
        Arch = 4
        return Arch

    Px = np.zeros((4))
    Px[0] = Wc
    Px[1] = int(Dpoint[0] + (Ah - Dpoint[1]) / k1)
    Px[2] = int(CrossP[0] + (Ah - CrossP[1]) / k2)
    Px[3] = int(CrossP[0] + (Ah - CrossP[1]) / k3)
    Px = np.sort(Px)

    if 'L' == whichone:
        if Awr < Px[0]:
            Arch = 3
        elif Awr >= Px[0] and Awr < Px[1]:
            Arch = 2
        elif Awr >= Px[1] and Awr < Px[2]:
            Arch = 1
        else:
            Arch = 0
    else:
        if Awl > Px[3]:
            Arch = 3
        elif Awl <= Px[3] and Awl > Px[2]:
            Arch = 2
        elif Awl <= Px[2] and Awl > Px[1]:
            Arch = 1
        else:
            Arch = 0

    return Arch


def PressLine(hmimg,img,angle,oname):
    orig = hmimg.copy()

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[0:2]
    SumH = np.array(np.sum(img, axis=1))

    points = list()
    step = 10
    for i in range(0, h, step):
        # filter the little value points line
        if SumH[i] < 10:
            continue

        Tval = 0
        for j in range(0, w):
            if img[i,j]>0:
                Tval = Tval + j * img[i,j]

        points.append((int(Tval/SumH[i]),i))

    Tpoints = list()
    center = np.array([int(w/2), int(h/2)])
    for k in range(len(points)):
        Tmp = getPointAffinedPos(points[k], center, angle)
        Tpoints.append((int(Tmp[0]), int(Tmp[1])))

    for k in range(len(Tpoints)):
        cv.circle(orig, Tpoints[k], 1, (255, 0, 255), -1)
        if k > 0:
            cv.line(orig, Tpoints[k-1], Tpoints[k], (255, 0, 255), 1, cv.LINE_AA)

    oimg(oname,orig)
    return 0

def SConvPressImg(filename, oname):
    if filename.find('.txt') >= 0:
        fo = open(filename, 'r')
        str = fo.read()
        fo.close()
    else:
        str = filename


    ## remove space and []
    if len(str)>0:
        str = str.strip('')
        str = str.strip('[')
        str = str.strip(']')

    Temp = str.split(',')
    B = [int(x) for x in Temp]

    # enlarge value
    scaling = int(255/max(B))
    print "Scaling:", scaling
    if scaling>1:
        B = [x*scaling for x in B]

    if len(B)!=2288:
        print "the press data length is valid:", len(B)
        return -1

    D = np.array(B, dtype=np.uint8).reshape(52,44)
    Dt = np.transpose(D)

    img = cv.cvtColor(Dt, cv.COLOR_GRAY2BGR)
    Bimg = cv.resize(img, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    oimg(oname, Bimg)
    return 0

def drawBalanceImg(hmimg, BCdata, oname):
    name = oname.split('.png')[0].split('_', 1)[0]
    orig = hmimg.copy()
    points = list()

    if BCdata.find('.txt') >= 0:
        fo = open(BCdata, 'r')
        str = fo.read()
        fo.close()
    else:
        str = BCdata

    print 'Balance position:', str
    if len(str)>0:
        str = str.strip('')
        str = str.strip('[')
        str = str.strip(']')

    Temp = str.split(';')
    for i in range(len(Temp)):
        Tp = Temp[i].split(',')
        Tpn = [int(x) for x in Tp]
        points.append(Tpn)

    if len(points) > 0:
        for i in range(len(points)):
            cv.circle(orig, (int(8*points[i][0]), int(8*points[i][1])), 2, (50, 50, 50), -1)

    oimg(name+'_balance.png', orig)
    return 0

def getfootReportInfo(hmImg, dataname, oname):
    name = oname.split('.png')[0].split('_', 1)[0]
    print 'oname', oname
    print name

    ret = Sprehandle(hmImg, dataname, name+'_org.png')
    if ret < 0:
        return ret

    ## Left foot
    hmlImg = rimg(name + "_left.png")
    img = rimg(name + "_leftgray.png")
    La = getRorate(img, name + "_leftgray-rotate.png")
    print 'Angle:', La

    img = rimg(name + "_leftgray-rotate.png")
    LArch = getLinesArch(hmlImg, img, 'L', La, oname=name+ "_leftfoot-arch.png")
    print 'Left Arch:', LArch

    PressLine(hmlImg, img, La, name+ "_leftfoot-pressline.png")

    ## Right foot
    hmrImg = rimg(name + "_right.png")
    img = rimg(name + "_rightgray.png")
    Ra = getRorate(img, name + "_rightgray-rotate.png")
    print 'Angle:', Ra

    img = rimg(name + "_rightgray-rotate.png")
    RArch = getLinesArch(hmrImg, img, 'R', Ra, name+ "_rightfoot-arch.png")
    print 'Right Arch:', RArch

    PressLine(hmrImg, img, Ra, name + "_rightfoot-pressline.png")

    return LArch,RArch
