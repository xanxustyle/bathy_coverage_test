from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigCanvas,
                                                NavigationToolbar2QT as FigNavToolbar)
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from main_window_ui import Ui_MainWindow
from concavehull import ConcaveHull


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

    def __init__(self, feature, diag):
        super().__init__()
        self.feature = feature
        self.diag = diag
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
            xmin, ymin = newdata[:, :2].min(axis=0) - self.diag
            xmax, ymax = newdata[:, :2].max(axis=0) + self.diag
            self.bin = createbin(xmin, xmax, ymin, ymax, self.feature)
            self.bin_signal.emit(self.bin)

        newdata[:, 0] = np.searchsorted(self.bin[0], newdata[:, 0], side='right')
        newdata[:, 1] = np.searchsorted(self.bin[1], newdata[:, 1], side='right')
        newdata = newdata[pd.DataFrame(newdata[:, :2]).drop_duplicates().index]
        self.data.append(newdata)
        np_data = np.vstack(self.data)
        hist = np.histogram2d(np_data[:, 0] - 0.5, np_data[:, 1] - 0.5, bins=(np.arange(self.bin[0].shape[0]),
                                                                              np.arange(self.bin[1].shape[0])))[0]
        self.data_signal.emit((np_data, hist))


class Builder(QtCore.QThread):
    runbuild_signal = QtCore.pyqtSignal(tuple)
    boun_signal = QtCore.pyqtSignal(object)

    def __init__(self, _bin):
        super().__init__()
        self.bin = _bin

    @QtCore.pyqtSlot(tuple)
    def buildboun(self, tup):
        hist = tup[0]
        boun_cover = tup[1]
        hull = ConcaveHull()
        hull_east, hull_north = np.nonzero(hist >= boun_cover)
        hull.loadpoints(np.column_stack((self.bin[2][hull_east], self.bin[3][hull_north])))
        hull.calculatehull(tol=(boun_cover ** 4) + 3)
        boun_xy = np.column_stack(hull.boundary.exterior.coords.xy)
        self.boun_signal.emit(boun_xy)


class Checker(QtCore.QThread):
    runcheck_signal = QtCore.pyqtSignal(tuple)
    fail_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, _bin):
        super().__init__()
        self.bin = _bin

    @QtCore.pyqtSlot(tuple)
    def checkgrid(self, sval):
        self.fail_signal.emit(sval)


class Planner(QtCore.QThread):
    runplan_signal = QtCore.pyqtSignal(tuple)
    plan_signal = QtCore.pyqtSignal(tuple)

    def __init__(self):
        super().__init__()

    @QtCore.pyqtSlot(tuple)
    def planpath(self, sval):
        self.plan_signal.emit(sval)


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.watcher = None
        self.reader = None
        self.builder = None
        self.checker = None
        self.planner = None
        self.bin = ()
        self.hist = None

        self.inputDirBrowseButton.clicked.connect(self.selectwatchdir)
        self.inputName.setText('Job_' + str(datetime.now())[:-16])
        self.inputFeature.setText('1')
        self.inputCoverage.setText('2')
        self.inputDiag.setText('1000')
        self.runButton.setEnabled(False)
        self.inputDir.textChanged.connect(self.enablerun)
        self.runButton.clicked.connect(self.runprogram)
        self.execButton.setEnabled(False)
        self.bounFileRadio.toggled.connect(self.disablexecute)
        self.failOutCheckbox.clicked.connect(self.disablexecute)
        self.ppOutCheckbox.clicked.connect(self.disablexecute)
        self.bounFileInput.textChanged.connect(self.disablexecute)
        self.failDir.textChanged.connect(self.disablexecute)
        self.ppDir.textChanged.connect(self.disablexecute)
        self.bounGroup.clicked.connect(self.clickgroup)
        self.failGroup.clicked.connect(self.clickgroup)
        self.ppGroup.clicked.connect(self.clickgroup)
        self.execButton.clicked.connect(self.exectask)

        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_axes((0.07, 0.1, 0.9, 0.85), xlabel='Easting [m]', ylabel='Northing [m]',
                                      title='Coverage Map', aspect='equal', xticks=[], yticks=[], rasterized=True)
        self.ax1.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"
        self.cmesh = None
        self.cbar = None
        self.canvas1 = FigCanvas(self.fig1)
        self.plotLayout1.addWidget(self.canvas1)
        self.plotLayout1.addWidget(FigNavToolbar(self.canvas1, self.plotBox1, coordinates=True))
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_axes((0.07, 0.1, 0.9, 0.85), xlabel='Easting [m]', ylabel='Northing [m]',
                                      title='Grid Compliance', aspect='equal',
                                      xticks=[], yticks=[], rasterized=True)
        self.ax2.set_anchor('C')
        self.ax2.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"
        self.canvas2 = FigCanvas(self.fig2)
        self.plotLayout2.addWidget(self.canvas2)
        self.plotLayout2.addWidget(FigNavToolbar(self.canvas2, self.plotBox2, coordinates=True))

    def selectwatchdir(self):
        self.inputDir.setText(QFileDialog.getExistingDirectory(self, 'Select Line File Directory'))

    def enablerun(self):
        self.runButton.setEnabled(bool(self.inputDir.text()))

    def clickgroup(self):
        if self.bounGroup.isChecked() and self.ppGroup.isChecked() and not self.failGroup.isChecked():
            self.failGroup.setChecked(True)
        self.disablexecute()

    def disablexecute(self):
        self.execButton.setDisabled(
            (self.bounGroup.isChecked() and self.bounFileRadio.isChecked() and not bool(self.bounFileInput.text())) or
            (self.failGroup.isChecked() and self.failOutCheckbox.isChecked() and not bool(self.failDir.text())) or
            (self.ppGroup.isChecked() and self.ppOutCheckbox.isChecked() and not bool(self.ppDir.text())))

    def runprogram(self):
        self.inputDirBrowseButton.setEnabled(False)
        self.inputDir.setEnabled(False)
        self.inputName.setEnabled(False)
        self.inputFeature.setEnabled(False)
        self.inputCoverage.setEnabled(False)
        self.inputDiag.setEnabled(False)
        self.runButton.setEnabled(False)
        self.watcher = Watcher(self.inputDir.text())
        self.watcher.start()
        self.reader = Reader(float(self.inputFeature.text()), float(self.inputDiag.text()))
        self.watcher.watch_signal.connect(self.reader.readline)
        self.reader.bin_signal.connect(self.setbin)
        self.reader.data_signal.connect(self.drawmap)

    def exectask(self):
        self.execButton.setEnabled(False)
        if self.bounGroup.isChecked():
            if self.bounRadio.isChecked():
                self.builder.runbuild_signal.emit((self.hist, self.bounSpinbox.value()))
            elif self.ginputRadio.isChecked():
                boun_xy = np.asfarray(self.fig1.ginput(-1))
                self.builder.boun_signal.emit(boun_xy)
            else:
                boun_xy = pd.read_csv(self.bounFileInput.text(), sep=' ', usecols=[0, 1]).to_numpy()
                self.builder.boun_signal.emit(boun_xy)

        elif not self.bounGroup.isChecked() and self.failGroup.isChecked():
            self.runcheck()

        elif not self.bounGroup.isChecked() and not self.failGroup.isChecked() and self.ppGroup.isChecked():
            self.runplan()

        else:
            self.execButton.setEnabled(True)

    def runcheck(self):
        self.checker.runcheck_signal.emit(('Checking', 1))

    def runplan(self):
        self.planner.runplan_signal.emit(('Planning', 1))

    @QtCore.pyqtSlot(tuple)
    def setbin(self, tup):
        self.bin = tup
        self.builder = Builder(self.bin)
        self.checker = Checker(self.bin)
        self.planner = Planner()
        self.builder.runbuild_signal.connect(self.builder.buildboun)
        self.builder.boun_signal.connect(self.drawboun)
        self.checker.runcheck_signal.connect(self.checker.checkgrid)
        self.checker.fail_signal.connect(self.drawfail)
        self.planner.runplan_signal.connect(self.planner.planpath)
        self.planner.plan_signal.connect(self.drawpath)
        self.disablexecute()

    @QtCore.pyqtSlot(tuple)
    def drawmap(self, tup):
        data = tup[0]
        self.hist = tup[1]
        self.ax1.set_xlim(self.bin[2][int(data[:, 0].min()) - 5], self.bin[2][int(data[:, 0].max()) + 5])
        self.ax1.set_ylim(self.bin[3][int(data[:, 1].min()) - 5], self.bin[3][int(data[:, 1].max()) + 5])
        if self.cmesh is None:
            cmax = int(np.max(self.hist))
            self.cmesh = self.ax1.imshow(
                self.hist.T, cmap=plt.get_cmap('viridis', cmax + 1), interpolation='nearest', origin='lower',
                extent=[self.bin[0][0], self.bin[0][-1], self.bin[1][0], self.bin[1][-1]])
            self.cbar = self.fig1.colorbar(
                self.cmesh, ticks=np.linspace(cmax / (cmax + 1) / 2, cmax - cmax / (cmax + 1) / 2, cmax + 1),
                aspect=50, location='bottom')
            self.cbar.ax.set_xticklabels(np.arange(cmax + 1))
            self.ax1.set_position((0.07, 0.1, 0.9, 0.85))
            self.ax1.set_anchor('C')
            self.cbar.ax.set_position((0.07, 0.03, 0.9, 0.018))
        else:
            cmax = int(np.max(self.hist))
            self.cmesh.set_data(self.hist.T)
            self.cmesh.set_cmap(plt.get_cmap('viridis', cmax + 1))
            self.cmesh.autoscale()
            self.cbar.set_ticks(np.linspace(cmax / (cmax + 1) / 2, cmax - cmax / (cmax + 1) / 2, cmax + 1))
            self.cbar.draw_all()
            self.cbar.ax.set_xticklabels(np.arange(cmax + 1))
        self.canvas1.draw()

    @QtCore.pyqtSlot(object)
    def drawboun(self, boun_xy):
        self.ax2.plot(boun_xy[:, 0], boun_xy[:, 1], 'black', linewidth=0.5)
        self.canvas2.draw()
        if self.failGroup.isChecked():
            self.runcheck()
        else:
            self.execButton.setEnabled(True)

    @QtCore.pyqtSlot(tuple)
    def drawfail(self, tup):
        print(tup)
        if self.ppGroup.isChecked():
            self.runplan()
        else:
            self.execButton.setEnabled(True)

    @QtCore.pyqtSlot(tuple)
    def drawpath(self, tup):
        print(tup)
        self.execButton.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
