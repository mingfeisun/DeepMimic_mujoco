import numpy as np
import cv2
import os

import time

class VideoSaver():
    gShowType = "dump" #play or dump video
    gDumpDir = "./render"
    gDumpFd = None
    gFps = 15

    def __init__(self, showType="dump", dumpDir="./render", width=640, height=480, fps=15):
        assert showType in ["dump", "play"]

        self.gShowType = showType
        self.gDumpDir = dumpDir
        self.gFps = fps
        if self.gShowType == "dump":
            if not os.path.isdir(self.gDumpDir):
                os.mkdir(self.gDumpDir)
            video_name = self.gDumpDir + "/" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
            if os.path.isfile(video_name):
                os.remove(video_name)

            (major, _, _) = cv2.__version__.split(".")
            if major == '2':
                fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            print("demoShow.init openning %s " %(video_name))
            self.gDumpFd = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        print("demoShow.init success")

    def close(self):
        assert self.gDumpFd != None
        self.gDumpFd.release()

    def addFrame(self, img):
        if self.gShowType == "dump":
            assert self.gDumpFd != None
            self.gDumpFd.write(img)
        else:
            sleepTime = 1
            cv2.imshow('image', img)
            cv2.waitKey(sleepTime)