from datetime import datetime
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, distance_matrix
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as Polypatch
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigCanvas,
                                                NavigationToolbar2QT as FigNavToolbar)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from concavehull import ConcaveHull
from elkai import solve_float_matrix
from inpoly import inpoly2
from main_window_ui import Ui_MainWindow
from ufunclab import max_argmax


def createbin(xymin, xymax, edge):
    xymin -= 0.0001
    xymax += edge - (xymax - xymin) % edge
    binedge1 = np.linspace(xymin[0], xymax[0], num=int(round((xymax[0] - xymin[0]) / edge)) + 1)
    binedge2 = np.linspace(xymin[1], xymax[1], num=int(round((xymax[1] - xymin[1]) / edge)) + 1)
    bin1 = binedge1[:-1] + edge / 2
    bin2 = binedge2[:-1] + edge / 2
    return binedge1, binedge2, bin1, bin2


def getpolyarea(xy):
    xy_ = xy - xy.mean(axis=0)
    correction = xy_[-1, 0] * xy_[0, 1] - xy_[-1, 1] * xy_[0, 0]
    main_area = np.dot(xy_[:-1, 0], xy_[1:, 1]) - np.dot(xy_[:-1, 1], xy_[1:, 0])
    return 0.5 * np.abs(main_area + correction)


def nlist2array(nlist):
    lens = np.array([len(i) for i in nlist])
    mask = lens[:, None] > np.arange(lens.max())
    arr = np.full(mask.shape, -1, dtype=int)
    arr[mask] = np.concatenate(nlist)
    return arr


class Handler(PatternMatchingEventHandler):
    def __init__(self, watch_signal):
        super(Handler, self).__init__(patterns=['*.txt'], ignore_directories=True, case_sensitive=True)
        self.watch_signal = watch_signal

    def on_modified(self, event):
        """Depending on how Caris process creates the file, this might not work.
        Solution: Create tmp file when writing, rename to txt on completion, use on_moved and change to dest_path"""
        self.watch_signal.emit(str(event.src_path))


class Watcher(QThread):
    watch_signal = pyqtSignal(str)

    def __init__(self, watchdir):
        super(Watcher, self).__init__()
        self.watchdir = watchdir
        self.observer = Observer()
        self.handler = Handler(self.watch_signal)
        self.observer.schedule(self.handler, self.watchdir, recursive=False)
        self.observer.start()


class Reader(QThread):
    bin_signal = pyqtSignal(tuple)
    data_signal = pyqtSignal(tuple)

    def __init__(self, feature, density, diag):
        super().__init__()
        self.feature = feature
        self.density = density
        self.diag = diag
        self.xylim = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
        self.depth = [0, 0]
        self.bin = ()
        self.hist = None

    @pyqtSlot(str)
    def readline(self, sval):
        newdata = pd.read_csv(sval, sep=' ', usecols=[0, 1, 2]).to_numpy()
        self.xylim[0, :] = np.minimum(self.xylim[0, :], newdata[:, :2].min(axis=0))
        self.xylim[1, :] = np.maximum(self.xylim[1, :], newdata[:, :2].max(axis=0))

        if self.hist is None:
            self.bin = createbin(self.xylim[0, :] - self.diag, self.xylim[1, :] + self.diag, self.feature)
            self.bin_signal.emit(self.bin)
            self.hist = np.where(
                np.histogram2d(newdata[:, 0], newdata[:, 1], bins=(self.bin[0], self.bin[1]))[0] >= self.density, 1, 0)
        else:
            self.hist += np.where(
                np.histogram2d(newdata[:, 0], newdata[:, 1], bins=(self.bin[0], self.bin[1]))[0] >= self.density, 1, 0)

        self.depth[0] = (newdata[:, 2].sum() + self.depth[0] * self.depth[1]) / (newdata.shape[0] + self.depth[1])
        self.depth[1] += newdata.shape[0]
        self.data_signal.emit((self.hist, self.depth[0], self.xylim))


class Builder(QThread):
    runbuild_signal = pyqtSignal(tuple)
    boun_signal = pyqtSignal(object)

    def __init__(self, _bin):
        super().__init__()
        self.bin = _bin

    @pyqtSlot(tuple)
    def buildboun(self, tup):
        hist = tup[0]
        boun_cover = tup[1]
        hull = ConcaveHull()
        hull_east, hull_north = np.nonzero(hist >= boun_cover)
        hull.loadpoints(np.column_stack((self.bin[2][hull_east], self.bin[3][hull_north])))
        hull.calculatehull(tol=(boun_cover ** 4) + 3)
        boun_xy = np.column_stack(hull.boundary.exterior.coords.xy)
        self.boun_signal.emit(boun_xy)


class Checker(QThread):
    runcheck_signal = pyqtSignal(tuple)
    fail_signal = pyqtSignal(object)

    def __init__(self, _bin):
        super().__init__()
        self.bin = _bin

    @pyqtSlot(tuple)
    def checkgrid(self, tup):
        hist = tup[0]
        boun_xy = tup[1]
        coverage = tup[2]
        fail_east, fail_north = np.nonzero(hist < coverage)
        fail_xy = np.column_stack((self.bin[2][fail_east], self.bin[3][fail_north]))
        fail_xy = fail_xy[inpoly2(fail_xy, boun_xy)[0]]
        self.fail_signal.emit(fail_xy)


class Planner(QThread):
    runplan_signal = pyqtSignal(tuple)
    plan_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

    @pyqtSlot(tuple)
    def planpath(self, tup):
        fail_xy = tup[0]
        swath_radius = tup[1] * np.tan(np.deg2rad(tup[2] / 2))
        run_iter = tup[3]

        fbin = createbin(fail_xy.min(axis=0), fail_xy.max(axis=0), swath_radius / 2)
        fail_tree = KDTree(fail_xy)
        fail_grp = np.column_stack((np.searchsorted(fbin[0], fail_xy[:, 0], side='right'),
                                    np.searchsorted(fbin[1], fail_xy[:, 1], side='right')))
        fail_grp = fail_grp[pd.DataFrame(fail_grp).drop_duplicates().index].astype(float)
        fail_grp[:, 0] = fbin[2][fail_grp[:, 0].astype(int) - 1]
        fail_grp[:, 1] = fbin[3][fail_grp[:, 1].astype(int) - 1]
        fail_grp = fail_xy[fail_tree.query(fail_grp, workers=-1)[1]]

        fail_tree = KDTree(fail_grp)
        neighbor = fail_tree.query_ball_tree(fail_tree, swath_radius)
        count_max = fail_grp.shape[0]
        neighbor = nlist2array(neighbor) + 1
        neighbor_max, neighbor_imax = max_argmax(np.count_nonzero(neighbor, axis=1))
        waypt = [fail_grp[neighbor_imax]]
        count = neighbor_max
        while count < count_max:
            neighbor[np.isin(neighbor, neighbor[neighbor_imax, :])] = 0
            neighbor_max, neighbor_imax = max_argmax(np.count_nonzero(neighbor, axis=1))
            waypt.append(fail_grp[neighbor_imax])
            count += neighbor_max
        waypt = np.row_stack(waypt)

        dist_mat = distance_matrix(waypt, waypt, threshold=1e10)
        path = solve_float_matrix(dist_mat, runs=run_iter)
        path.append(0)
        self.plan_signal.emit(waypt[path])


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.watcher = None
        self.reader = None
        self.builder = None
        self.checker = None
        self.planner = None
        self.line_no = 0
        self.depth = 0
        self.bin = ()
        self.hist = None
        self.boundary = None
        self.failgrid = None

        self.inputDirBrowseButton.clicked.connect(self.selectwatchdir)
        self.inputName.setText('Job_' + str(datetime.now())[:-16])
        self.inputFeature.setText('1')
        self.inputCoverage.setText('1')
        self.inputDensity.setText('1')
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
        self.linepatch = Polypatch([[0, 0], [0, 0]], edgecolor='black', facecolor='None', lw=0.5, zorder=0)
        self.ax2.add_patch(self.linepatch)
        self.polypatch = Polypatch([[0, 0], [0, 0]], edgecolor='None', facecolor='lawngreen', alpha=0.9, zorder=0)
        self.ax2.add_patch(self.polypatch)
        self.failplot, = self.ax2.plot([], [], ls='None', marker='.', c='crimson', ms=4, mew=0.01, zorder=1,
                                       label='Fail grids')
        self.failtext = plt.figtext(0.52, 0.02, '', figure=self.fig2, ha='center', va='bottom')
        self.pathplot, = self.ax2.plot([], [], ls='--', lw=1.5, marker='o', c='black', ms=5, zorder=2,
                                       label='Path waypoints')
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
        if len(self.bin) > 0:
            self.execButton.setDisabled(
                (self.bounGroup.isChecked() and self.bounFileRadio.isChecked() and not bool(self.bounFileInput.text()))
                or (self.failGroup.isChecked() and self.failOutCheckbox.isChecked() and not bool(self.failDir.text()))
                or (self.ppGroup.isChecked() and self.ppOutCheckbox.isChecked() and not bool(self.ppDir.text())))

    def runprogram(self):
        self.inputDirBrowseButton.setEnabled(False)
        self.inputDir.setEnabled(False)
        self.inputName.setEnabled(False)
        self.inputFeature.setEnabled(False)
        self.inputCoverage.setEnabled(False)
        self.inputDensity.setEnabled(False)
        self.inputDiag.setEnabled(False)
        self.runButton.setEnabled(False)
        self.watcher = Watcher(self.inputDir.text())
        self.reader = Reader(float(self.inputFeature.text()), float(self.inputDensity.text()),
                             float(self.inputDiag.text()))
        self.watcher.watch_signal.connect(self.notifyread)
        self.watcher.watch_signal.connect(self.reader.readline)
        self.reader.bin_signal.connect(self.setbin)
        self.reader.data_signal.connect(self.drawmap)
        self.watcher.start()
        self.consoleBox.appendPlainText('Program started.\nWatching {}'.format(self.inputDir.text()))

    def exectask(self):
        self.execButton.setEnabled(False)
        if self.bounGroup.isChecked():
            self.consoleBox.appendPlainText('Building survey area boundary... ')
            if self.bounRadio.isChecked():
                self.builder.runbuild_signal.emit((self.hist, self.bounSpinbox.value()))
            elif self.ginputRadio.isChecked():
                boun_xy = np.asarray(self.fig1.ginput(-1))
                if boun_xy.shape[0] > 2:
                    self.builder.boun_signal.emit(boun_xy)
                else:
                    self.consoleBox.appendPlainText('Input boundary ERROR. A minimum of 3 points must be given.')
                    self.execButton.setEnabled(True)
            else:
                boun_xy = pd.read_csv(self.bounFileInput.text(), sep=' ', usecols=[0, 1]).to_numpy()
                if boun_xy.shape[0] > 2:
                    self.builder.boun_signal.emit(boun_xy)
                else:
                    self.consoleBox.appendPlainText('Input boundary ERROR. A minimum of 3 points must be given.')
                    self.execButton.setEnabled(True)

        elif not self.bounGroup.isChecked() and self.failGroup.isChecked():
            self.runcheck()

        elif not self.bounGroup.isChecked() and not self.failGroup.isChecked() and self.ppGroup.isChecked():
            self.runplan()

        else:
            self.execButton.setEnabled(True)

    def runcheck(self):
        self.consoleBox.appendPlainText('Checking grid compliance... ')
        self.checker.runcheck_signal.emit((self.hist, self.boundary, float(self.inputCoverage.text())))

    def runplan(self):
        self.consoleBox.appendPlainText('Planning path for repairing data... ')
        if self.failgrid.shape[0] > 0:
            self.planner.runplan_signal.emit(
                (self.failgrid, self.depth, self.swathSpinbox.value(), self.ppSpinBox.value()))
        else:
            self.consoleBox.appendPlainText('No path planning is required. All grids are compliant.')
            self.execButton.setEnabled(True)

    @pyqtSlot(str)
    def notifyread(self, sval):
        self.consoleBox.appendPlainText('Reading {}... '.format(sval))

    @pyqtSlot(tuple)
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

    @pyqtSlot(tuple)
    def drawmap(self, tup):
        self.consoleBox.insertPlainText('Done.')
        self.line_no += 1
        self.hist = tup[0]
        self.depth = tup[1]
        xylim = tup[2]
        self.ax1.set_xlim(xylim[0][0] - 5, xylim[1][0] + 5)
        self.ax1.set_ylim(xylim[0][1] - 5, xylim[1][1] + 5)
        self.ax2.set_xlim(xylim[0][0] - 5, xylim[1][0] + 5)
        self.ax2.set_ylim(xylim[0][1] - 5, xylim[1][1] + 5)
        self.ax1.set_title('Coverage Map of {} Lines '.format(self.line_no))
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

    @pyqtSlot(object)
    def drawboun(self, boun_xy):
        self.consoleBox.insertPlainText('Done.')
        self.boundary = boun_xy
        # noinspection PyArgumentList
        self.linepatch.set_xy(boun_xy)
        self.canvas2.draw()

        if self.failGroup.isChecked():
            self.runcheck()
        else:
            self.execButton.setEnabled(True)

    @pyqtSlot(object)
    def drawfail(self, fail_xy):
        self.consoleBox.insertPlainText('Done.')
        self.failgrid = fail_xy
        bounarea = getpolyarea(self.boundary)
        failarea = fail_xy.shape[0] * (float(self.inputFeature.text()) ** 2)
        failrate = min(failarea / bounarea, 1)
        self.failtext.set_text(
            'Compliant Grids (Green): {:.2%}\nNon-compliant Grids (Red): {:.2%}'.format(1 - failrate, failrate))
        # noinspection PyArgumentList
        self.polypatch.set_xy(self.boundary)
        self.failplot.set_data(fail_xy[:, 0], fail_xy[:, 1])
        self.canvas2.draw()

        if self.ppGroup.isChecked():
            self.runplan()
        else:
            self.execButton.setEnabled(True)

    @pyqtSlot(object)
    def drawpath(self, waypt):
        self.consoleBox.insertPlainText('Done.')
        self.pathplot.set_data(waypt[:, 0], waypt[:, 1])
        self.canvas2.draw()
        self.execButton.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
