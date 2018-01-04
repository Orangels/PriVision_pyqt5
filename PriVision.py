from PyQt4.QtCore import *
from PyQt4.QtGui import *
import qdarkstyle
import sys
from mainwindow_privison import MainWindow

__appname__ = 'PriVison'


def main(argv):
    """Standard boilerplate Qt application code."""
    app = QApplication(argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
    app.setApplicationName(__appname__)
    # app.setWindowIcon(QIcon('./files/pose_icon.png'))
    # app.setWindowIcon(newIcon("app"))
    win = MainWindow(__appname__)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
