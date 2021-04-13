import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow  # , QDialog, QMessageBox
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigCanvas,
                                                NavigationToolbar2QT as FigNavToolbar)
from main_window_ui import Ui_MainWindow
import sys
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


class Handler(PatternMatchingEventHandler, QtCore.QThread):
    def __init__(self, watch_signal):
        super(Handler, self).__init__(patterns=['*.txt'], ignore_directories=True, case_sensitive=True)
        self.watch_signal = watch_signal

    def on_modified(self, event):
        """Depending on how Caris process create the file, this might not work.
        Workaround: Caris create a tmp file while writing, rename it to txt on completion, and use on_moved here"""
        self.watch_signal.emit(str(event.src_path))


class Watcher(QtCore.QThread):
    watch_signal = QtCore.pyqtSignal(str)

    def __init__(self, watchdir):
        super(Watcher, self).__init__()
        self.watchdir = watchdir
        self.observer = Observer()
        self.handler = Handler(self.watch_signal)
        self.observer.schedule(self.handler, self.watchdir, recursive=False)
        self.observer.start()


def createbin(xmin, xmax, ymin, ymax, edge):
    xmin = xmin - 0.0001
    xmax = xmax + edge - (xmax - xmin) % edge
    ymin = ymin - 0.0001
    ymax = ymax + edge - (ymax - ymin) % edge
    binedge1 = np.linspace(xmin, xmax, num=int(round((xmax - xmin) / edge)) + 1)
    binedge2 = np.linspace(ymin, ymax, num=int(round((ymax - ymin) / edge)) + 1)
    bin1 = binedge1[:-1] + edge / 2
    bin2 = binedge2[:-1] + edge / 2
    return binedge1, binedge2, bin1, bin2


class Reader(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(tuple)
    bin_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, feature):
        super().__init__()
        self.feature = feature
        self.line_no = 0
        self.bin = ()
        self.data = []

    @QtCore.pyqtSlot(str)
    def readline(self, sval):
        print('Reading', sval)
        newdata = pd.read_csv(sval, sep=' ', usecols=[0, 1, 2]).to_numpy()
        self.line_no += 1
        print(self.line_no, 'lines copied.')

        if self.line_no == 1:
            xmin, ymin = newdata[:, :2].min(axis=0) - 1000
            xmax, ymax = newdata[:, :2].max(axis=0) + 1000
            self.bin = createbin(xmin, xmax, ymin, ymax, self.feature)
            self.bin_signal.emit(self.bin)

        newdata[:, 0] = np.searchsorted(self.bin[0], newdata[:, 0], side='right')
        newdata[:, 1] = np.searchsorted(self.bin[1], newdata[:, 1], side='right')
        newdata = newdata[pd.DataFrame(newdata[:, :2]).drop_duplicates().index]
        self.data.append(newdata)
        np_data = np.vstack(self.data)
        hist = np.histogram2d(np_data[:, 0] - 0.5, np_data[:, 1] - 0.5, bins=(np.arange(self.bin[0].shape[0]),
                                                                              np.arange(self.bin[1].shape[0])))[0]
        self.data_signal.emit((np_data, hist.T))


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.feature = 1
        self.bin = ()
        watchdir = 'C:/Users/limhs/Desktop/Intern2021/Code/TEST_onboard'
        self.watcher = Watcher(watchdir)
        self.watcher.start()
        self.reader = Reader(self.feature)
        self.watcher.watch_signal.connect(self.reader.readline)
        self.reader.bin_signal.connect(self.setbin)
        self.reader.data_signal.connect(self.drawmap)

        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_axes((0, 0, 1, 1), xlabel='Easting [m]', ylabel='Northing [m]', title='Coverage Map',
                                      aspect='equal', xticks=[], yticks=[], rasterized=True)
        self.ax1.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"
        self.cmesh = None
        self.cbar = None
        self.canvas1 = FigCanvas(self.fig1)
        self.plotLayout1.addWidget(self.canvas1)
        self.plotLayout1.addWidget(FigNavToolbar(self.canvas1, self.plotBox1, coordinates=True))
        self.fig2 = plt.figure()
        self.canvas2 = FigCanvas(self.fig2)
        self.plotLayout2.addWidget(self.canvas2)
        self.plotLayout2.addWidget(FigNavToolbar(self.canvas2, self.plotBox2, coordinates=True))

    @QtCore.pyqtSlot(tuple)
    def setbin(self, tup):
        self.bin = tup

    @QtCore.pyqtSlot(tuple)
    def drawmap(self, tup):
        data = tup[0]
        hist = tup[1]
        self.ax1.set_xlim(self.bin[2][int(data[:, 0].min()) - 5], self.bin[2][int(data[:, 0].max()) + 5])
        self.ax1.set_ylim(self.bin[3][int(data[:, 1].min()) - 5], self.bin[3][int(data[:, 1].max()) + 5])
        if self.cmesh is None:
            cmax = int(np.max(hist))
            self.cmesh = self.ax1.imshow(hist, cmap=plt.get_cmap('viridis', cmax + 1), interpolation='nearest',
                                         origin='lower',
                                         extent=[self.bin[0][0], self.bin[0][-1], self.bin[1][0], self.bin[1][-1]])
            self.cbar = self.fig1.colorbar(self.cmesh, ticks=
                                           np.linspace(cmax / (cmax + 1) / 2, cmax - cmax / (cmax + 1) / 2, cmax + 1),
                                           aspect=50, location='bottom')
            self.cbar.ax.set_xticklabels(np.arange(cmax + 1))
            self.ax1.set_position((0.1, 0.1, 0.85, 0.85))
            self.cbar.ax.set_position((0.1, 0.03, 0.85, 0.017))
        else:
            cmax = int(np.max(hist))
            self.cmesh.set_data(hist)
            self.cmesh.set_cmap(plt.get_cmap('viridis', cmax + 1))
            self.cmesh.autoscale()
            self.cbar.set_ticks(np.linspace(cmax / (cmax + 1) / 2, cmax - cmax / (cmax + 1) / 2, cmax + 1))
            self.cbar.draw_all()
            self.cbar.ax.set_xticklabels(np.arange(cmax + 1))
        self.canvas1.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
