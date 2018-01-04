# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

sys.path.append('/home/user/workspace/openpose-priv-dev/3rdparty/lib/priv_tools')

from common import draw_bbox, draw_fancybbox, draw_pose_kpts, draw_mask, clock, draw_str, draw_kpts
from label_info import COLORMAP21, COLORMAP19, COLORMAP20, WHITE19, LINK_PAIR

from PyQt4.QtCore import *
from PyQt4.QtGui import *

try:
    _fromUtf8 = QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)

import numpy as np
import cv2
import datetime
from PIL import Image
import time
from functools import partial
from multiprocessing import Process, Queue

from libs.config import cfg
from libs.lib import *
from libs.toolbar import *
from libs.canvas import *
from libs.zoomwidget import *
from libs.cameradevice import *

# from libs.objdetector import *
#from libs.persondetector import *
#from libs.trafficdetector import *
#from libs.poseestimator import *
#from libs.facedetector import *
#from libs.facealign import *
#from libs.facerecog import *
from modules.facedetector import *

from mainwindow_ui import Ui_MainWindow


class GetImage:
    def __init__(self):
        pass

    def __call__(self, img):
        return img


task1_in = Queue(2)
task1_out = Queue(2)
task2_in = Queue(2)
task2_out = Queue(2)
task3_in = Queue(2)
task3_out = Queue(2)
task4_in = Queue(2)
task4_out = Queue(2)


def Task1_process(msg_in, msg_out):
    Worker1 = PersonDetector(gpu_id=1, fastface=False)
    while True:
        if not msg_in.empty():
            frame = msg_in.get()
            result = Worker1(frame)
            msg_out.put(result)


def Task2_process(msg_in, msg_out):
    Worker2 = PersonEstimator()
    while True:
        if not msg_in.empty():
            frame = msg_in.get()
            result = Worker2(frame)
            msg_out.put(result)


def Task3_process(msg_in, msg_out):
    Worker3 = TrafficDetector(gpu_id=2)
    # Worker3 = FaceAlign(gpu_id=2)
    while True:
        if not msg_in.empty():
            frame = msg_in.get()
            result = Worker3(frame)
            msg_out.put(result)


def Task4_process(msg_in, msg_out):
    Worker4 = FaceRecog(gpu_id=3)
    while True:
        if not msg_in.empty():
            frame = msg_in.get()
            result = Worker4(frame)
            msg_out.put(result)


process_task1 = Process(target=Task1_process, args=(task1_in, task1_out))
process_task1.daemon = True
process_task2 = Process(target=Task2_process, args=(task2_in, task2_out))
process_task2.daemon = True
process_task3 = Process(target=Task3_process, args=(task3_in, task3_out))
process_task3.daemon = True
process_task4 = Process(target=Task4_process, args=(task4_in, task4_out))
process_task4.daemon = True

process_task1.start()
process_task2.start()
process_task3.start()
process_task4.start()


class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = range(3)
    resized = QtCore.pyqtSignal()

    def __init__(self, app_name):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # canvas main window
        self.setWindowIcon(QIcon('./files/pose_icon.png'))
        self.canvas = Canvas()
        self.canvas.image = QImage("./files/backl.jpg")
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.ui.main_video_layout.addWidget(scroll)

        self.zoomWidget = ZoomWidget()
        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self.zoomMode = self.FIT_WINDOW
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.resized.connect(self.adjustScale)
        # camera
        self.camera_device = CameraDevice(video_path='./files/none.avi')
        self.camera_device.newFrame.connect(self.onNewImage)
        self.camera_device.video_time_out.connect(self.clear)

        # top button functions
        self.ui.thread1_bn.clicked.connect(self.start_camera)
        self.ui.thread2_bn.clicked.connect(self.start_video1)
        self.ui.thread3_bn.clicked.connect(self.start_video2)
        self.ui.thread4_bn.clicked.connect(self.start_pic)
        self.ui.thread5_bn.clicked.connect(self.load_video)
        # left button functions
        self.ui.play_bn.clicked.connect(self.start_cap)
        self.ui.pause_bn.clicked.connect(self.stop_cap)
        self.ui.record_bn.clicked.connect(self.save_video)
        self.ui.exit_bn.clicked.connect(self.close)
        # right button functions
        self.ui.model1_bn.clicked.connect(self.apply_model1)
        self.ui.model2_bn.clicked.connect(self.apply_model2)
        self.ui.model3_bn.clicked.connect(self.apply_model3)
        self.ui.model4_bn.clicked.connect(self.apply_model4)
        self.ui.model5_bn.clicked.connect(self.apply_model5)

        # top button functions
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-bright.png"))
        self.ui.thread1_bn.setIconSize(QSize(110, 20))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-bright.png"))
        self.ui.thread2_bn.setIconSize(QSize(110, 20))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-bright.png"))
        self.ui.thread3_bn.setIconSize(QSize(110, 20))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-bright.png"))
        self.ui.thread4_bn.setIconSize(QSize(110, 20))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-bright.png"))
        self.ui.thread5_bn.setIconSize(QSize(110, 20))
        # left button functions
        self.ui.play_bn.setIcon(QIcon("./files/icons/icon-left/play-bright.png"))
        self.ui.play_bn.setIconSize(QSize(110, 70))
        self.ui.pause_bn.setIcon(QIcon("./files/icons/icon-left/pause-bright.png"))
        self.ui.pause_bn.setIconSize(QSize(110, 70))
        self.ui.record_bn.setIcon(QIcon("./files/icons/icon-left/record-bright.png"))
        self.ui.record_bn.setIconSize(QSize(110, 70))
        self.ui.empty_bn.setIcon(QIcon("./files/icons/icon-left/empty-bright.png"))
        self.ui.empty_bn.setIconSize(QSize(110, 70))
        self.ui.setting_bn.setIcon(QIcon("./files/icons/icon-left/setting-bright.png"))
        self.ui.setting_bn.setIconSize(QSize(110, 70))
        self.ui.exit_bn.setIcon(QIcon("./files/icons/icon-left/exit.png"))
        self.ui.exit_bn.setIconSize(QSize(110, 70))
        # right button icons
        self.ui.model1_bn.setIcon(QIcon("./files/icons/icon-right/human-detection-bright.png"))
        self.ui.model1_bn.setIconSize(QSize(110, 70))
        self.ui.model2_bn.setIcon(QIcon("./files/icons/icon-right/human-pose-bright.png"))
        self.ui.model2_bn.setIconSize(QSize(110, 70))
        self.ui.model3_bn.setIcon(QIcon("./files/icons/icon-right/face-align-bright.png"))
        self.ui.model3_bn.setIconSize(QSize(110, 70))
        self.ui.model4_bn.setIcon(QIcon("./files/icons/icon-right/face-recog-bright.png"))
        self.ui.model4_bn.setIconSize(QSize(110, 70))
        self.ui.model5_bn.setIcon(QIcon("./files/icons/icon-right/human-seg-bright.png"))
        self.ui.model5_bn.setIconSize(QSize(110, 70))

        # task special param
        self.image_render = GetImage()
        self.flag_savevideo = False
        self.info_header = u"您好! 欢迎您试用PriVision!\n  "
        self.shown_info(self.info_header)
        self.allframes = []
        self.video_writer = None
        self.savevideo_counting = 0
        self.savevideo_max = 1800 * 20

    def resizeEvent(self, event):
        self.resized.emit()
        return super(MainWindow, self).resizeEvent(event)

    def update_image(self):
        pass

    def zoomRequest(self, delta):
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def setFitWindow(self, value=True):
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.width() * 0.65 - e
        h1 = self.height() * 0.65 - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.image.width() - 0.0
        h2 = self.canvas.image.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() / 5 * 3 - 2.0
        return w / self.canvas.pixmap.width()

    def paintCanvas(self):
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        # self.canvas.scale = 0.5
        self.canvas.adjustSize()
        self.canvas.update()

    def start_cap(self):
        # self.ui.play_bn.setIcon(QIcon("./files/icons/icon-left/play-color.png"))
        self.ui.pause_bn.setIcon(QIcon("./files/icons/icon-left/pause-bright.png"))
        self.camera_device.paused = False

    def stop_cap(self):
        self.ui.pause_bn.setIcon(QIcon("./files/icons/icon-left/pause-color.png"))
        self.camera_device.paused = True

    def save_video(self):
        if not self.flag_savevideo:
            self.flag_savevideo = True
            self.savevideo_counting = 0
            video_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter('/home/user/Desktop/{}_save.avi'.format(video_name), fourcc, 20,
                                                (960, 540))
            self.ui.record_bn.setIcon(QIcon("./files/icons/icon-left/record-color.png"))
        else:
            self.video_writer.release()
            self.savevideo_counting = 0
            self.flag_savevideo = False
            self.video_writer = None
            self.ui.record_bn.setIcon(QIcon("./files/icons/icon-left/record-bright.png"))
        pass

    def clear(self):
        self.allframes = []

    def shown_info(self, info):
        self.ui.info_display.setPlainText(info)

    def apply_model1(self):
        if cfg.GLOBAL.F_MODEL1:
            cfg.GLOBAL.F_MODEL1 = False
            self.ui.model1_bn.setIcon(QIcon("./files/icons/icon-right/human-detection-bright.png"))
        else:
            cfg.GLOBAL.F_MODEL1 = True
            self.ui.model1_bn.setIcon(QIcon("./files/icons/icon-right/human-detection-color.png"))

    def apply_model2(self):
        if cfg.GLOBAL.F_MODEL2:
            cfg.GLOBAL.F_MODEL2 = False
            self.ui.model2_bn.setIcon(QIcon("./files/icons/icon-right/human-pose-bright.png"))
        else:
            cfg.GLOBAL.F_MODEL2 = True
            self.ui.model2_bn.setIcon(QIcon("./files/icons/icon-right/human-pose-color.png"))

    def apply_model3(self):
        if cfg.GLOBAL.F_MODEL3:
            cfg.GLOBAL.F_MODEL3 = False
            self.ui.model3_bn.setIcon(QIcon("./files/icons/icon-right/face-align-bright.png"))
        else:
            cfg.GLOBAL.F_MODEL3 = True
            self.ui.model3_bn.setIcon(QIcon("./files/icons/icon-right/face-align-color.png"))

    def apply_model4(self):
        if cfg.GLOBAL.F_MODEL4:
            cfg.GLOBAL.F_MODEL4 = False
            self.ui.model4_bn.setIcon(QIcon("./files/icons/icon-right/face-recog-bright.png"))
        else:
            cfg.GLOBAL.F_MODEL4 = True
            self.ui.model4_bn.setIcon(QIcon("./files/icons/icon-right/face-recog-color.png"))

    def apply_model5(self):
        pass

    def start_camera(self):
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-color.png"))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-bright.png"))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-bright.png"))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-bright.png"))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-bright.png"))
        self.camera_device = CameraDevice()
        self.camera_device.newFrame.connect(self.onNewImage)
        self.clear()

    def start_video1(self):
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-bright.png"))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-color.png"))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-bright.png"))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-bright.png"))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-bright.png"))
        self.camera_device.set_video_path('./files/news.avi')
        self.clear()

    def start_video2(self):
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-bright.png"))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-bright.png"))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-color.png"))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-bright.png"))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-bright.png"))
        self.camera_device.set_video_path('./files/shanxi_traffic/0500320.avi')
        self.clear()

    def start_pic(self):
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-bright.png"))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-bright.png"))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-bright.png"))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-color.png"))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-bright.png"))
        self.camera_device.set_video_path('./files/xx2.avi')
        self.clear()

    def load_video(self):
        self.ui.thread1_bn.setIcon(QIcon("./files/icons/icon-top/realtime-mode-bright.png"))
        self.ui.thread2_bn.setIcon(QIcon("./files/icons/icon-top/playback1-bright.png"))
        self.ui.thread3_bn.setIcon(QIcon("./files/icons/icon-top/playback2-bright.png"))
        self.ui.thread4_bn.setIcon(QIcon("./files/icons/icon-top/picture-mode-bright.png"))
        self.ui.thread5_bn.setIcon(QIcon("./files/icons/icon-top/loading-mode-color.png"))
        self.camera_device.set_video_path('./files/facedet.avi')
        self.clear()

    @QtCore.pyqtSlot(np.ndarray)
    def onNewImage(self, frame):
        self.adjustScale()

        frame = np.asarray(frame[:, :])
        frame = cv2.resize(frame, (960, 640))
        self.allframes.append(frame)
        vis = frame.copy()

        # vis = cv2.resize(vis, (960, 640))
        t = clock()
        result1, result2, result3, result4 = None, None, None, None
        if cfg.GLOBAL.F_MODEL1:
            task1_in.put(frame)
        if cfg.GLOBAL.F_MODEL2:
            task2_in.put(frame)
        if cfg.GLOBAL.F_MODEL3:
            task3_in.put(frame)
        if cfg.GLOBAL.F_MODEL4:
            task4_in.put(frame)
        while True:
            if cfg.GLOBAL.F_MODEL1:
                task1_empty = task1_out.empty()
            else:
                task1_empty = False
            if cfg.GLOBAL.F_MODEL2:
                task2_empty = task2_out.empty()
            else:
                task2_empty = False
            if cfg.GLOBAL.F_MODEL3:
                task3_empty = task3_out.empty()
            else:
                task3_empty = False
            if cfg.GLOBAL.F_MODEL4:
                task4_empty = task4_out.empty()
            else:
                task4_empty = False
            if not task1_empty and not task2_empty and not task3_empty and not task4_empty:
                if cfg.GLOBAL.F_MODEL1:
                    result1 = task1_out.get()
                else:
                    result1 = None
                if cfg.GLOBAL.F_MODEL2:
                    result2 = task2_out.get()
                else:
                    result2 = None
                if cfg.GLOBAL.F_MODEL3:
                    result3 = task3_out.get()
                else:
                    result3 = None
                if cfg.GLOBAL.F_MODEL4:
                    result4 = task4_out.get()
                else:
                    result4 = None
                break
        if result1 is not None:
            vis = draw_fancybbox(vis, result1)
        if result2 is not None:
            if cfg.GLOBAL.F_MODEL3:
                vis = draw_pose_kpts(vis, result2, WHITE19, LINK_PAIR)
            else:
                vis = draw_pose_kpts(vis, result2, COLORMAP19, LINK_PAIR)
        if result3 is not None:
            vis = draw_fancybbox(vis, result3)
            # vis = draw_kpts(vis, result3)
        if result4 is not None:
            vis = draw_fancybbox(vis, result4, attri=True)
        dt = clock() - t

        if self.flag_savevideo and self.savevideo_counting <= self.savevideo_max:
            save_im = cv2.resize(vis, (960, 540))
            self.video_writer.write(save_im)
            self.savevideo_counting += 1
        elif self.savevideo_counting > self.savevideo_max:
            self.savevideo_counting = 0
            self.flag_savevideo = False
            self.ui.record_bn.setIcon(QIcon("./files/icons/icon-left/record-bright.png"))
        draw_str(vis, (30, 30), 'speed: %.1f fps' % (min(1.0 / dt, 30)))

        if len(self.allframes) % 15 == 0:
            cur_info = self.info_header + u'--------------------\n  '
            if self.flag_savevideo:
                cur_info += u'Saving Video~~\n--------------------\n'
            cur_info += u'当前为第{}帧视频\n  当前视频频率为: {:.1f}fps\n  '. \
                format(len(self.allframes), min(1.0 / dt, 30))
            cur_info += u'--------------------\n  '
            self.shown_info(cur_info)

        vis = cv2.resize(vis, (960, 640))
        image = QImage(vis.tostring(), vis.shape[1], vis.shape[0], QImage.Format_RGB888).rgbSwapped()

        self.canvas.update_image(image)
