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
# import matplotlib.pyplot as plt

pixelsPerMetric = None
odir = '.'
pixelMinValue = 1


def oimg(name,img):
    global odir
    if not os.path.exists(odir):
        os.makedirs(odir)
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


def drawLines(lines, oimg):
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


# for sensor press mat
def Sprehandle(hmimg,datafile,oname = 'org.png'):
    img = SConvPressImg(datafile)
    if img is None:
        print 'the org image is None.', oname
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


def getRorate(img, ref, oname):
    angle = 0
    name = oname[0 : oname.rfind('-')]
    rectName = name + '_Rect.png'
    if len(ref)==4 and ref[0]!=0:
        k = ref[1]/ref[0]
    else:
        x,y,w,h = getRoiRect(img, rectName)
        # print(x,y,w,h)
        if w == 0 or h == 0:
            oimg(oname, img)
            return angle

        d = np.rint(h/6)

        # determine the up and down lines
        US = int(y+d)
        UE = int(y+4*d)
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
        k = float(Dx-Ux)/(Dy-Uy+(DS-US))

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


def getRoiRect(img, oname = 'rect.png'):
    x = y = w = h = 0
    minValidPoints = 10
    if img is None:
        return x,y,w,h

    orig = img.copy()
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    nz = np.nonzero(img)
    nzt = np.transpose(np.asarray(nz))
    # x, y  exchange
    nztx = [[x[1],x[0]] for x in nzt]

    # the foot valid points is less than 10 points
    if len(nztx) < minValidPoints:
        return x,y,w,h

    # get rect
    x,y,w,h = cv.boundingRect(np.asarray(nztx))
    # imgrect = cv.rectangle(orig, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
    # oimg(oname, imgrect)
    return x,y,w,h


def getBalanceCenter(img):
    x = y = 0
    if img is None:
        return x,y

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[0:2]
    # print(h,w)
    SumP = np.sum(img)
    SumW = np.array(np.sum(img, axis=0))
    SumH = np.array(np.sum(img, axis=1))
    Hpress = WPress = 0

    for i in range(0,w):
        WPress = WPress + i*SumW[i]

    for j in range(0,h):
        Hpress = Hpress + j*SumH[j]


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


def getBoundaryPoint(img, Hstart, Hend, Wstart, Wend, step):
    ## find the left and right boundary points of img
    global pixelMinValue
    HL = WL = HR= WR = 0

    if img is None:
        return HL,WL,HR, WR

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    WL = max(Wstart, Wend)
    WR = min(Wstart, Wend)
    for i in range(Hstart, Hend, step):
        for j in range(Wstart, Wend):
            if img[i][j] > pixelMinValue:
                if j <= WL:
                    WL = j
                    HL = i

                if j >= WR:
                    WR = j
                    HR = i

    return HL,WL,HR,WR


def getMaxLine(img, Hstart, Hend, Wstart, Wend, step):
    ## find the longest line of img
    global pixelMinValue
    LineNumMax = 0
    H = WL = WR = 0

    if img is None:
        return H,WL,WR

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
    if img is None:
        return H,WL,WR

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


def getLinesArch(hmimg,img,whichone,angle,ref,oname):
    ## Arch is [0,1,2,3,4], 2 is normal, >2 is high arch, <2 is low arch
    # orig = hmimg.copy()

    Arch = 2
    H,W = img.shape[0:2]
    center = np.array((int(W/2), int(H/2)))
    # print "img(H,W):",H,W

    rectName = oname.split('_', -1)[0] + '_Rect.png'
    x,y,w,h = getRoiRect(img, rectName)
    # print(x, y, w, h)

    # set the line start and end H axis
    DiffHW = int(min(h,W)/2)
    Ths = int(y+h/2-DiffHW)+1
    The = int(y+h/2+DiffHW)-1

    # no foot on this img
    if w == 0 or h == 0:
        # oimg(oname, orig)
        return Arch

    Ws = x
    Hs = y
    We = x+w-5
    He = y+h-5
    Hdeep = int(h/3)
    # Wc = int(x+w/2)
    # print Hs,He,Ws,We,Hdeep,Wc

    # get Ref line is out Ref line or the heal gravity center
    if len(ref) == 4 and ref[0] != 0 and ref[1] != 0:
        Refpoint = getPointAffinedPos((ref[3],ref[2]), center, -angle)
        Refl = Dx = Refpoint[0]
        Dy = Refpoint[1]
    else:
        Dx, Dy = getBalanceCenter(img[He-Hdeep:He, :])
        Refl = Dx

    print 'angle:', angle
    print 'Dx,Dy:', Dx,Dy

    if 0:
        Uh, Uwl, Uwr = getMaxLine(img, Hs, Hs+Hdeep, Ws, We, 1)
        Dh, Dwl, Dwr = getMaxLine(img, He, He-Hdeep, Ws, We, -1)
    else:
        Uhl, Uwl, Uhr, Uwr =getBoundaryPoint(img, Hs, Hs+Hdeep, Ws, We, 1)
        Dhl, Dwl, Dhr, Dwr = getBoundaryPoint(img, He, He-Hdeep, Ws, We, -1)

    if abs(Uwl-Uwr) == 0 or abs(Dwl-Dwr) == 0:
        # oimg(oname, orig)
        return Arch

    Upoint = Dpoint = np.zeros((2))
    if 'L' == whichone:
        Upoint = np.array([Uwr, Uhr])
        Dpoint = np.array([Dwr, Dhr])
    else:
        Upoint = np.array([Uwl, Uhl])
        Dpoint = np.array([Dwl, Dhl])
    # print "Upoint: ", Upoint
    # print "Dpoint: ", Dpoint

    # The tangent slope, H is X axis, and W is Y axis
    k1 = float(Dpoint[0]-Upoint[0])/float(Dpoint[1]-Upoint[1])

    # Transform the points to Heat map img
    # TUpoint = getPointAffinedPos(Upoint, center, angle)
    # TDpoint = getPointAffinedPos(Dpoint, center, angle)
    # TRefls = getPointAffinedPos((Refl, Ths), center, angle)
    # TRefle = getPointAffinedPos((Refl, The), center, angle)

    ## the k1 line and Refl line is parallel
    if k1 == 0:
        # cv.line(orig, (int(TUpoint[0]), int(TUpoint[1])), (int(TDpoint[0]), int(TDpoint[1])), (255, 0, 255), 1, cv.LINE_AA)
        # cv.line(orig, (int(TRefls[0]), int(TRefls[1])), (int(TRefle[0]), int(TRefle[1])), (255, 0, 255), 1, cv.LINE_AA)
        return Arch

    # get the cross point of k1 and W center line
    CrossP = np.array([Refl, (Refl-Dpoint[0])/k1+Dpoint[1]])

    # get the lines slope
    rad = math.atan(k1)
    Qrad = float(rad)/4
    k2 = math.tan(2 * Qrad)
    k3 = math.tan(Qrad)
    print "(k1,k2,k3): ", k1,k2,k3

    ## draw line on img
    orig = img.copy()
    ## Transform the lines start and end points to heat map img
    # Tk1s = getPointAffinedPos((int(k1 * (Ths - Dpoint[1]) + Dpoint[0]), Ths), center, angle)
    # Tk1e = getPointAffinedPos((int(k1 * (The - Dpoint[1]) + Dpoint[0]), The), center, angle)
    #
    # Tk2s = getPointAffinedPos((int(k2 * (Ths - CrossP[1]) + CrossP[0]), Ths), center, angle)
    # Tk2e = getPointAffinedPos((int(k2 * (The - CrossP[1]) + CrossP[0]), The), center, angle)
    #
    # Tk3s = getPointAffinedPos((int(k3 * (Ths - CrossP[1]) + CrossP[0]), Ths), center, angle)
    # Tk3e = getPointAffinedPos((int(k3 * (The - CrossP[1]) + CrossP[0]), The), center, angle)

    Tk1s = np.array((int(k1 * (Ths - Dpoint[1]) + Dpoint[0]), Ths))
    Tk1e = np.array((int(k1 * (The - Dpoint[1]) + Dpoint[0]), The))

    Tk2s = np.array((int(k2 * (Ths - CrossP[1]) + CrossP[0]), Ths))
    Tk2e = np.array((int(k2 * (The - CrossP[1]) + CrossP[0]), The))

    Tk3s = np.array((int(k3 * (Ths - CrossP[1]) + CrossP[0]), Ths))
    Tk3e = np.array((int(k3 * (The - CrossP[1]) + CrossP[0]), The))

    ## get lines start point and end point
    lines = list()
    lines.append((int(Tk1s[0]), int(Tk1s[1]), int(Tk1e[0]), int(Tk1e[1])))
    lines.append((int(Tk2s[0]), int(Tk2s[1]), int(Tk2e[0]), int(Tk2e[1])))
    lines.append((int(Tk3s[0]), int(Tk3s[1]), int(Tk3e[0]), int(Tk3e[1])))
    lines.append((int(Refl), int(Ths), int(Refl), int(The)))
    # print("========lines=======")
    # print(lines)

    cv.circle(orig, (int(Dx), int(Dy + He - Hdeep)), 4, (255, 0, 0), -1)
    cv.rectangle(orig, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv.circle(orig, (int(Upoint[0]), int(Upoint[1])), 2, (255, 0, 0), -1)
    cv.circle(orig, (int(Dpoint[0]), int(Dpoint[1])), 2, (255, 0, 0), -1)
    for k in range(len(lines)):
        cv.line(orig, (lines[k][0], lines[k][1]), (lines[k][2], lines[k][3]), (255, 0, 255), 1, cv.LINE_AA)

    oimg(oname, orig)

    ## get arch line
    Ah,Awl,Awr = getMinLine(img, Hs+Hdeep, He-Hdeep, Ws, We)
    cv.line(orig, (int(x), int(Ah)), (int(x+w), int(Ah)), (0, 0, 255), 1, cv.LINE_AA)
    oimg(oname, orig)

    if Ah==0 or Awl==Awr:
        Arch = 4
        return Arch

    Px = np.zeros((4))
    Px[0] = Refl
    Px[1] = int(Dpoint[0] + float(Ah - Dpoint[1]) * k1)
    Px[2] = int(CrossP[0] + float(Ah - CrossP[1]) * k2)
    Px[3] = int(CrossP[0] + float(Ah - CrossP[1]) * k3)
    Px = np.sort(Px)
    print Ah, Awl, Awr
    print 'Px:', Px

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


def SConvPressImg(filename):
    if filename.find('.txt') >= 0:
        fo = open(filename, 'r')
        Sdata = fo.read()
        fo.close()
    else:
        Sdata = filename

    if len(Sdata) < 1:
        print "the data len is valid"
        return None

    ## remove space and []
    if len(Sdata)>0:
        Sdata = Sdata.strip(' ')
        Sdata = Sdata.strip('[')
        Sdata = Sdata.strip(']')

    Temp = Sdata.split(',')
    B = [float(x) for x in Temp]

    # enlarge value
    scaling = int(255/max(B))
    # print "Scaling:", scaling
    if scaling>1:
        B = [x*scaling for x in B]


    if len(B)!=2288:
        print "the press data length is valid:", len(B)
        return -1

    D = np.zeros((44,52), np.uint8)
    for j in range(51,-1,-1):
        for i in range(44):
            if j>25:
                D[i][j] = B[i + 1144 + 44 * (51-j)]
            else:
                D[i][j] = B[i + 44 * (25 - j)]

    Dt = np.array(D, dtype=np.uint8)

    img = cv.cvtColor(Dt, cv.COLOR_GRAY2BGR)
    Bimg = cv.resize(img, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)

    return Bimg


def drawBalanceImg(hmimg, BCdata, oname):
    name = oname.split('.png')[0].split('_', 1)[0]
    points = list()

    orig = hmimg.copy()

    ## draw center of x axis and y axis
    h,w = hmimg.shape[0:2]
    center = np.array([int(w/2), int(h/2)])
    deep = 40
    cv.line(orig, (int(center[0]-deep), int(center[1])), (int(center[0]+deep), int(center[1])), (255,0,255), 1, cv.LINE_AA)
    cv.line(orig, (int(center[0]), int(center[1]-deep)), (int(center[0]), int(center[1]+deep)), (255,0,255), 1, cv.LINE_AA)

    if BCdata.find('.txt') >= 0:
        fo = open(BCdata, 'r')
        Sdata = fo.read()
        fo.close()
    else:
        Sdata = BCdata

    print 'Balance position:', Sdata
    if len(Sdata)>0:
        Sdata = Sdata.strip(' ')
        Sdata = Sdata.strip('[')
        Sdata = Sdata.strip(']')

        Temp = Sdata.split(';')
        for i in range(len(Temp)):
            Tp = Temp[i].split(',')
            Tpn = [8*int(x) for x in Tp]
            points.append(Tpn)

        if len(points) > 0:
            for i in range(len(points)):
                cv.circle(orig, (int(points[i][0]), int(points[i][1])), 1, (255, 0, 0), 0)
                if i > 0:
                    cv.line(orig, (int(points[i-1][0]), int(points[i-1][1])), (int(points[i][0]), int(points[i][1])),
                            (50, 50, 50), 1, cv.LINE_AA)

    oimg(name+'_balance.png', orig)
    return 0


def Qpress(img):
    Q = np.zeros((4))

    if img is None:
        print 'the image is null'
        return Q

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    H,W = img.shape[0:2]
    x,y,w,h = getRoiRect(img, '')
    # print(x, y, w, h)

    Hcen = int(y+h/2)
    Wcen = int(x+w/2)

    Tnum = np.sum(img)
    Q1 = Q2 = Q3 = 0
    if Tnum > 0:
        for i in range(Hcen):
            for j in range(Wcen):
                Q1 = Q1+img[i][j]

        for i in range(Hcen):
            for j in range(Wcen, W):
                Q2 = Q2+img[i][j]

        for i in range(Hcen, H):
            for j in range(Wcen):
                Q3 = Q3+img[i][j]

        Q[0] = float(Q1) / Tnum
        Q[1] = float(Q2) / Tnum
        Q[2] = float(Q3) / Tnum
        Q[3] = 1-Q[0]-Q[1]-Q[2]

    for i in range(len(Q)):
        Q[i] = "%.2f" % Q[i]

    return Q


def getfootReportInfo(hmname, dataname, BCdata, Refline, oname):
    hmimg = cv.imread(hmname)
    if hmimg is None:
        print 'the heatmap image is None.'
        return -1

    name = oname.split('.png')[0].split('_', 1)[0]
    print 'oname', oname
    print name

    ret = Sprehandle(hmimg, dataname, name+'_org.png')
    if ret < 0:
        return ret

    lref,rref = getRefPoints(Refline)

    ## Left foot
    hmlImg = rimg(name + "_left.png")
    img = rimg(name + "_leftgray.png")
    La = getRorate(img, lref, name + "_leftgray-rotate.png")
    print 'Left Angle:', La

    img = rimg(name + "_leftgray-rotate.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    LArch = getLinesArch(hmlImg, img, 'L', La, lref, oname=name+ "_leftfoot-arch.png")
    print 'Left Arch:', LArch

    PressLine(hmlImg, gray, La, name+ "_leftfoot-pressline.png")
    Qp = Qpress(gray)
    print Qp
    LQ = [str(x) for x in Qp]

    ## Right foot
    hmrImg = rimg(name + "_right.png")
    img = rimg(name + "_rightgray.png")
    Ra = getRorate(img, rref, name + "_rightgray-rotate.png")
    print 'Right Angle:', Ra

    img = rimg(name + "_rightgray-rotate.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    RArch = getLinesArch(hmrImg, img, 'R', Ra, rref, name+ "_rightfoot-arch.png")
    print 'Right Arch:', RArch

    PressLine(hmrImg, gray, Ra, name + "_rightfoot-pressline.png")
    Qp = Qpress(gray)
    RQ = [str(Qp[1]),str(Qp[0]), str(Qp[3]), str(Qp[2])]

    drawBalanceImg(hmimg, BCdata, name + "_balance.png")

    return LArch, RArch, LQ, RQ


def fitline(data):
    nz = np.nonzero(data)
    nzt = np.transpose(np.asarray(nz))
    nzta = np.array(nzt)
    # have refline or not
    if len(nzta) < 2:
        return 0,0,0,0

    # Height is X axis, and Width is Y axis for Img
    [vh,vw,h,w] = cv.fitLine(nzta, cv.DIST_HUBER, 0, 0.01, 0.01)
    return vh,vw,h,w

def getRefPoints(filename):
    lref = rref = np.array([])
    if filename is None:
        return lref, rref

    if filename.find('.txt') >= 0:
        fo = open(filename, 'r')
        Sdata = fo.read()
        fo.close()
    else:
        Sdata = filename

    print 'Ref Points:', Sdata
    if len(Sdata)>0:
        Sdata = Sdata.strip(' ')
        Sdata = Sdata.strip('[')
        Sdata = Sdata.strip(']')
        Sdata = Sdata.strip('"')

        Temp = Sdata.split('","')
        B = [float(x) for x in Temp]
        if len(B) == 8:
            lref = np.array([B[0],B[1],B[2],B[3]])
            rref = np.array([B[4],B[5],B[6],B[7]])

    return lref,rref

def setRefPoints(filename, oname):
    print 'setRefPoints', oname
    img = SConvPressImg(filename)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[0:2]
    leftD = gray[:, 0:int(w/2)]
    rightD = gray[:, int(w/2):]

    lvh,lvw,lh,lw = fitline(leftD)
    rvh,rvw,rh,rw = fitline(rightD)

    Ref = ["%.4f"%lvh, "%.4f"%lvw, "%.4f"%lh, "%.4f"%lw, "%.4f"%rvh, "%.4f"%rvw, "%.4f"%rh, "%.4f"%rw]

    # draow ref lines to std img
    stdimg = cv.imread('std.png')
    if lvh!=0 and lvw!=0:
        lk = lvw/lvh
        cv.line(stdimg, (int(lw - lk * lh), 0), (int((340 - lh) * lk + lw), 340), (255, 0, 255), 1)

    if rvh!=0 and rvw!=0:
        rk = rvw/rvh
        cv.line(stdimg, (int(rw-rk*rh),0), (int((340-rh)*rk+rw),340), (255,0,255),1)

    oimg(oname, stdimg)

    # draw line on img
    # if lvh!=0:
    #     lk = lvw/lvh
    #     center = np.array((int(w / 2), int(h / 2)))
    #     angle = math.atan(lk)
    #     Refpoint = getPointAffinedPos((lw, lh), center, -angle)
    #     Refpoint2 = getPointAffinedPos((int((280-lh)*lk+lw), 280), center, -angle)
    #     print Refpoint, Refpoint2
    #     cv.line(img, (int(lw-lk*lh), 0), (int((340-lh)*lk+lw),340), (0,255,0),1)
    #     cv.circle(img, (int(lw), int(lh)), 3, (255,0,0), -1)
    #     cv.circle(img, (int(Refpoint[0]), int(Refpoint[1])), 3, (255, 0, 255), -1)
    #     cv.circle(img, (int(Refpoint2[0]), int(Refpoint2[1])), 3, (255, 0, 255), -1)
    #     cv.circle(img, (int(center[0]), int(center[1])), 3, (0, 255, 255), -1)
    # if rvh!=0:
    #     rk = rvw/rvh
    #     cv.line(img, (int(rw-rk*rh),0), (int((340-rh)*rk+rw),340), (0,255,0),1)
    #
    # oimg('line1.png', img)

    return Ref


def drawrefimg(oname = 'std.png'):
    refstd = cv.imread('ref.png')
    h,w = refstd.shape[0:2]
    center = np.array((int(w / 2), int(h / 2)))

    # degree is 10, and distance is 8cm
    k = math.tan(float(10)/180*math.pi)
    bpitch = float(8)/36.4*416/2
    tpitch = bpitch + h*k

    deep = 20

    cv.line(refstd, (center[0],int(center[1]-deep)), (center[0],int(center[1]+deep)), (150, 150, 150), 1)
    cv.line(refstd, (int(center[0]-deep),center[1]), (int(center[0]+deep),center[1]), (150, 150, 150), 1)
    cv.line(refstd, (int(center[0]-tpitch),0), (int(center[0]-bpitch),h-1), (150, 150, 150), 4)
    cv.line(refstd, (int(center[0]+tpitch),0), (int(center[0]+bpitch),h-1), (150, 150, 150), 4)
    oimg(oname, refstd)

    return 0
