import cv2 as cv
import os

import requests
import time
from flask import jsonify, Flask, request, send_from_directory

import footdetector as fd

app = Flask(__name__)


@app.route("/footReport/info", methods=["POST"])
def footReportInfo():
    patiendId = request.form["patiendId"]
    picUrl = request.form["picUrl"]
    DataUrl = request.form["DataUrl"]
    BCData = request.form["BCData"]
    RefData = request.form.get("RefData", None)


    # baseDir = "C:/foot-dectector/nginx-1.14.0/html"
    # baseDomain = "http://120.79.130.74:8090"

    baseDir = "D:/WORKSPACE/python/foot/html"
    baseDomain = "http://localhost:5000/html"

    # datePart = time.strftime('%Y%m%d', time.localtime(time.time()))
    # timePart = time.strftime("%H%M%S", time.localtime(time.time()))
    # fileDir = "{}/{}/{}/{}".format(baseDir, patiendId, datePart, timePart)
    fileDir = "{}/{}".format(baseDir, patiendId)

    code = 1
    msg = ""
    leftArch = 0
    rightArch = 0
    lQpress = ""
    rQpress = ""

    try:
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)

        #fileName = "{}/{}".format(fileDir, picUrl[picUrl.rfind("/") + 1:len(picUrl)])
        # DataUrl = "{}/{}".format(fileDir, DataUrl[DataUrl.rfind("/") + 1:len(DataUrl)])

        print "fileName: ", picUrl
        # print "DataUrl: ", DataUrl

        # r = requests.get(picUrl, stream=True)
        # f = open(fileName, "wb")
        # for chunk in r.iter_content(chunk_size=512):
        #     if chunk:
        #         f.write(chunk)
        # f.close()

    except Exception as e:
        code = 0
        msg = "create dir error"
        print("Exception {}", e)


    if code == 1:
        try:
            # hmImg = cv.imread(picUrl)
            # DataUrl = 'D:/WORKSPACE/python/foot/html/foot1.txt'
            leftArch, rightArch, lQpress, rQpress = fd.getfootReportInfo(picUrl, DataUrl, BCData, RefData, fileDir+"/foot.png")

        except Exception as e:
            code = 0
            msg = "get foot report info error"
            print("Exception {}", e)

    # baseURI = "{}/{}/{}/{}".format(baseDomain, patiendId, datePart, timePart)
    baseURI = "{}/{}".format(baseDomain, patiendId)
    suffix = picUrl[picUrl.rfind("."):len(picUrl)]
    # leftArchImg = "foot_leftfoot-arch" + suffix
    # rightArchImg = "foot_rightfoot-arch" + suffix
    leftPressImg = "foot_leftfoot-pressline" + suffix
    rightPressImg = "foot_rightfoot-pressline" + suffix
    balanceImg = "foot_balance" + suffix

    if code == 1:
        # leftArchImg = "{}/{}".format(baseURI, leftArchImg)
        # rightArchImg = "{}/{}".format(baseURI, rightArchImg)
        leftPressImg = "{}/{}".format(baseURI, leftPressImg)
        rightPressImg = "{}/{}".format(baseURI, rightPressImg)
        balanceImg = "{}/{}".format(baseURI, balanceImg)
    else:
        # leftArchImg = ""
        # rightArchImg = ""
        leftPressImg = ""
        rightPressImg = ""
        balanceImg = ""

    data = {
        "leftArch": leftArch,
        "rightArch": rightArch,
        "lQpress": lQpress,
        "rQpress": rQpress,
        # "leftArchImg": leftArchImg,
        # "rightArchImg": rightArchImg,
        "leftPressImg": leftPressImg,
        "rightPressImg": rightPressImg,
        "balanceImg": balanceImg
    }

    result = {
        "code": code,
        "msg": msg,
        "data": data
    }

    return jsonify(result=result)


@app.route("/footReport/fetchRef", methods=["POST"])
def fetchRef():
    DataUrl = request.form["DataUrl"]
    Mac = request.form["Mac"]

    # baseDir = "C:/foot-dectector/nginx-1.14.0/html"
    # baseDomain = "http://120.79.130.74:8090"

    baseDir = "D:/WORKSPACE/python/foot/html"
    baseDomain = "http://localhost:5000/html"

    fileDir = "{}/{}/{}".format(baseDir, 'HW', Mac)

    code = 1
    msg = ""
    refline = ""

    try:
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)

    except Exception as e:
        code = 0
        msg = "create dir error"
        print("Exception {}", e)

    if code == 1:
        try:
            refline = fd.setRefPoints(DataUrl, fileDir+'/calibration.png')
        except Exception as e:
            code = 0
            msg = "fetch Ref line error"
            print("Exception {}", e)

    baseURI = "{}/{}/{}".format(baseDomain, 'HW', Mac)
    caliImg = "calibration.png"

    if code == 1:
        caliImg = "{}/{}".format(baseURI, caliImg)

    else:
        caliImg = ""

    data = {
        "refline": refline,
        "caliImg": caliImg
    }

    result = {
        "code": code,
        "msg": msg,
        "data": data
    }

    return jsonify(result=result)

'''
@app.route("/footReport/balance", methods=["POST"])
def footReportBalance():
    patiendId = request.form["patiendId"]
    picUrl = request.form["picUrl"]
    DataUrl = request.form["DataUrl"]

    # baseDir = "C:/foot-dectector/nginx-1.14.0/html"
    # baseDomain = "http://119.23.128.174:8090"
    baseDir = "D:/WORKSPACE/python/foot/html"
    baseDomain = "http://192.168.1.158:5000/html"

    # datePart = time.strftime('%Y%m%d', time.localtime(time.time()))
    # timePart = time.strftime("%H%M%S", time.localtime(time.time()))
    # fileDir = "{}/{}/{}/{}".format(baseDir, patiendId, datePart, timePart)
    fileDir = "{}/{}".format(baseDir, patiendId)

    code = 1
    msg = ""

    try:
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)

        # fileName = "{}/{}".format(fileDir, picUrl[picUrl.rfind("/") + 1:len(picUrl)])
        #
        # r = requests.get(picUrl, stream=True)
        # f = open(fileName, "wb")
        # for chunk in r.iter_content(chunk_size=512):
        #     if chunk:
        #         f.write(chunk)
        # f.close()

    except Exception as e:
        code = 0
        msg = "create dir error"
        print("Exception {}", e)

    if code == 1:
        try:
            hmImg = cv.imread(picUrl)
            # DataUrl = 'D:/WORKSPACE/python/foot/html/b.txt'
            fd.drawBalanceImg(hmImg, DataUrl, fileDir + "/foot")

        except Exception as e:
            code = 0
            msg = "draw foot Balance img error"
            print("Exception {}", e)

    # baseURI = "{}/{}/{}/{}".format(baseDomain, patiendId, datePart, timePart)
    baseURI = "{}/{}".format(baseDomain, patiendId)
    suffix = picUrl[picUrl.rfind("."):len(picUrl)]
    footBalance = "foot_balance" + suffix

    if code == 1:
        footBalance = "{}/{}".format(baseURI, footBalance)
    else:
        footBalance = ""

    data = {
        "footBalance": footBalance
    }

    result = {
        "code": code,
        "msg": msg,
        "data": data
    }

    return jsonify(result=result)
'''

@app.route('/html/<path:path>')
def send_file(path):
    return send_from_directory('html', path)

#app.run(host='0.0.0.0')