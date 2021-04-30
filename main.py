import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThreadPool, QRunnable, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QDialogButtonBox
from elkai import solve_float_matrix
from inpoly import inpoly2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigCanvas,
                                                NavigationToolbar2QT as FigNavToolbar)
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as Polypatch
from osgeo import gdal, osr
from scipy.spatial import KDTree, distance_matrix
from ufunclab import max_argmax
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from concavehull import ConcaveHull
from epsg_dialog_ui import Ui_Dialog
from main_window_ui import Ui_MainWindow


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


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.fn(*self.args, **self.kwargs)


class Handler(PatternMatchingEventHandler):
    def __init__(self, watch_signal):
        super(Handler, self).__init__(patterns=['*.txt'], ignore_directories=True, case_sensitive=True)
        self.watch_signal = watch_signal

    def on_created(self, event):
        """Depending on how Caris process creates the file, this might not work.
        Solution: Create tmp file when writing, rename to txt on completion, use on_moved and change to dest_path"""
        time.sleep(30)
        self.watch_signal.emit(str(event.src_path))


class Watcher(QObject):
    watch_signal = pyqtSignal(str)

    def __init__(self, watchdir):
        super(Watcher, self).__init__()
        self.watchdir = watchdir
        self.observer = Observer()
        self.handler = Handler(self.watch_signal)

    def startwatch(self):
        self.observer.schedule(self.handler, self.watchdir, recursive=False)
        self.observer.start()

    def prewatch(self, files):
        for file in files:
            self.watch_signal.emit(os.path.join(self.watchdir, file))
            time.sleep(.5)


class Reader(QObject):
    bin_signal = pyqtSignal(tuple)
    data_signal = pyqtSignal(tuple)

    def __init__(self, feature, density, diag):
        super(Reader, self).__init__()
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


class Builder(QObject):
    boun_signal = pyqtSignal(object)

    def __init__(self, _bin):
        super(Builder, self).__init__()
        self.bin = _bin

    @pyqtSlot(tuple)
    def buildbound(self, tup):
        hist, boun_cover = tup
        hull = ConcaveHull()
        hull_east, hull_north = np.nonzero(hist >= boun_cover)
        hull.loadpoints(np.column_stack((self.bin[2][hull_east], self.bin[3][hull_north])))
        hull.calculatehull(tol=2)
        boun_xy = np.column_stack(hull.boundary.exterior.coords.xy)
        self.boun_signal.emit(boun_xy)


class Checker(QObject):
    fail_signal = pyqtSignal(object)

    def __init__(self, _bin):
        super(Checker, self).__init__()
        self.bin = _bin

    @pyqtSlot(tuple)
    def checkgrid(self, tup):
        hist, boun_xy, coverage = tup
        fail_east, fail_north = np.nonzero(hist < coverage)
        fail_xy = np.column_stack((self.bin[2][fail_east], self.bin[3][fail_north]))
        fail_xy = fail_xy[inpoly2(fail_xy, boun_xy)[0]]
        self.fail_signal.emit(fail_xy)


class Planner(QObject):
    plan_signal = pyqtSignal(object)

    def __init__(self):
        super(Planner, self).__init__()

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


class Reporter(QObject):
    end_signal = pyqtSignal()

    def __init__(self):
        super(Reporter, self).__init__()

    @pyqtSlot(tuple)
    def genreport(self, tup):
        info, hist, xybin, xylim, boun, fail, file = tup
        with PdfPages(file) as pdf:
            fig1 = Figure(figsize=(8.3, 11.7))
            ax = fig1.add_subplot(111)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(which='both', colors='white')
            fig1.text(0.12, 0.9, info, transform=fig1.transFigure, family='calibri', size=15, linespacing=2.5,
                      ha='left', va='top')
            pdf.savefig(fig1)
            plt.close()

            fig2 = Figure(figsize=(8.3, 11.7))
            gridspec = fig2.add_gridspec(2, 2, width_ratios=[1, 0.04])
            ax1 = fig2.add_subplot(gridspec[0, 0], xlabel='Easting [m]', ylabel='Northing [m]', title='Coverage Map',
                                   aspect='equal')
            ax1.set_xlim(xylim[0][0] - 5, xylim[1][0] + 5)
            ax1.set_ylim(xylim[0][1] - 5, xylim[1][1] + 5)
            ax1.get_xaxis().get_major_formatter().set_useOffset(False)
            ax1.get_xaxis().get_major_formatter().set_scientific(False)
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)
            ax1.get_yaxis().get_major_formatter().set_scientific(False)
            cmax = int(np.max(hist))
            cmesh = ax1.imshow(hist.T, cmap=plt.get_cmap('viridis', cmax + 1), interpolation='nearest', origin='lower',
                               extent=[xybin[0][0], xybin[0][-1], xybin[1][0], xybin[1][-1]])
            ax1.set_anchor((0.8, 1))
            ax2 = fig2.add_subplot(gridspec[0, 1])
            cbar = fig2.colorbar(cmesh, cax=ax2,
                                 ticks=np.linspace(cmax / (cmax + 1) / 2, cmax - cmax / (cmax + 1) / 2, cmax + 1))
            cbar.ax.set_yticklabels(np.arange(cmax + 1))
            ax2.set_anchor('W')
            ax3 = fig2.add_subplot(gridspec[1, 0], xlabel='Easting [m]', ylabel='Northing [m]', title='Grid Compliance',
                                   aspect='equal')
            ax3.set_xlim(xylim[0][0] - 5, xylim[1][0] + 5)
            ax3.set_ylim(xylim[0][1] - 5, xylim[1][1] + 5)
            ax3.get_xaxis().get_major_formatter().set_useOffset(False)
            ax3.get_xaxis().get_major_formatter().set_scientific(False)
            ax3.get_yaxis().get_major_formatter().set_useOffset(False)
            ax3.get_yaxis().get_major_formatter().set_scientific(False)
            patch = Polypatch(boun, edgecolor='black', facecolor='lawngreen', lw=0.5, alpha=0.9, zorder=0)
            ax3.add_patch(patch)
            ax3.scatter(fail[:, 0], fail[:, 1], marker="s", c='r', s=1, lw=0, zorder=1)
            ax3.set_anchor((0.8, 1))
            pdf.savefig(fig2)
            plt.close()
        self.end_signal.emit()


class EpsgDialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.epsg = osr.SpatialReference()
        self.epsg.ImportFromEPSG(self.spinBox.value())
        self.textEdit.setPlainText(self.epsg.GetName())
        self.spinBox.valueChanged.connect(self.check_epsg)

    def check_epsg(self):
        self.epsg.ImportFromEPSG(self.spinBox.value())
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(bool(self.epsg.GetName()))
        if bool(self.epsg.GetName()):
            self.textEdit.setPlainText(self.epsg.GetName())
        else:
            self.textEdit.setPlainText('EPSG not found.')


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.threadpool = QThreadPool()
        self.watcher = None
        self.reader = None
        self.builder = None
        self.checker = None
        self.planner = Planner()
        self.reporter = Reporter()
        self.dialog = EpsgDialog(self)

        self.line_no = 0
        self.depth = 0
        self.bin = ()
        self.xylim = None
        self.hist = None
        self.boundary = None
        self.failgrid = None

        self.inputDirBrowseButton.clicked.connect(self.selectwatchdir)
        self.inputName.setText('Job_' + str(datetime.now())[:-16])
        self.runButton.setEnabled(False)
        self.inputDir.textChanged.connect(self.enablerun)
        self.runButton.clicked.connect(self.runprogram)
        self.execButton.setEnabled(False)
        self.bounFileBrowseButton.clicked.connect(self.selectbounfile)
        self.failDirBrowseButton.clicked.connect(self.selectfailoutput)
        self.ppDirBrowseButton.clicked.connect(self.selectplanoutput)
        self.bounFileRadio.toggled.connect(self.enabletask)
        self.failOutCheckbox.clicked.connect(self.enabletask)
        self.ppOutCheckbox.clicked.connect(self.enabletask)
        self.bounFileInput.textChanged.connect(self.enabletask)
        self.failDir.textChanged.connect(self.enabletask)
        self.ppDir.textChanged.connect(self.enabletask)
        self.bounGroup.clicked.connect(self.clickgroup)
        self.failGroup.clicked.connect(self.clickgroup)
        self.ppGroup.clicked.connect(self.clickgroup)
        self.execButton.clicked.connect(self.exectask)
        self.reportButton.setEnabled(False)
        self.reportDirBrowseButton.clicked.connect(self.selectreportoutput)
        self.reportDir.textChanged.connect(self.enablereport)
        self.reportButton.clicked.connect(self.run_reporter)
        self.reporter.end_signal.connect(self.endreport)

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_axes((0.07, 0.1, 0.9, 0.85), xlabel='Easting [m]', ylabel='Northing [m]',
                                      title='Coverage Map', aspect='equal', xticks=[], yticks=[], rasterized=True)
        self.ax1.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"
        self.cmesh = None
        self.cbar = None
        self.canvas1 = FigCanvas(self.fig1)
        self.plotLayout1.addWidget(self.canvas1)
        self.plotLayout1.addWidget(FigNavToolbar(self.canvas1, self.plotBox1, coordinates=True))
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_axes((0.07, 0.1, 0.9, 0.85), xlabel='Easting [m]', ylabel='Northing [m]',
                                      title='Grid Compliance', aspect='equal',
                                      xticks=[], yticks=[], rasterized=True)
        self.ax2.set_anchor('C')
        self.linepatch = Polypatch([[0, 0], [0, 0]], edgecolor='black', facecolor='None', lw=0.5, zorder=0)
        self.ax2.add_patch(self.linepatch)
        self.polypatch = Polypatch([[0, 0], [0, 0]], edgecolor='None', facecolor='lawngreen', alpha=0.9, zorder=0)
        self.ax2.add_patch(self.polypatch)
        self.failplot, = self.ax2.plot([], [], ls='None', marker='s', c='red', ms=2, mew=0, zorder=1,
                                       label='Fail grids')
        self.failtext = self.fig2.text(0.52, 0.02, '', transform=self.fig2.transFigure, ha='center', va='bottom')
        self.pathplot, = self.ax2.plot([], [], ls='--', lw=1.5, marker='o', c='black', ms=5, zorder=2,
                                       label='Path waypoints')
        self.ax2.format_coord = lambda x, y: f"x={x:.2f}, y={y:.2f}"
        self.canvas2 = FigCanvas(self.fig2)
        self.plotLayout2.addWidget(self.canvas2)
        self.plotLayout2.addWidget(FigNavToolbar(self.canvas2, self.plotBox2, coordinates=True))

    def selectwatchdir(self):
        self.inputDir.setText(QFileDialog.getExistingDirectory(self, 'Select Line File Directory'))

    def selectbounfile(self):
        self.bounFileInput.setText(
            QFileDialog.getOpenFileName(self, 'Select Boundary Input File', '', 'ASCII text (*.txt *.csv)')[0])

    def selectfailoutput(self):
        self.failDir.setText(
            QFileDialog.getSaveFileName(self, 'Save Non-compliant Grids', '',
                                        'Geotiff image (*.tif)\n Caris waypoints (*.wpt)\n ASCII text (*.txt)')[0])

    def selectplanoutput(self):
        self.ppDir.setText(QFileDialog.getSaveFileName(self, 'Save Path Waypoints', '',
                                                       'Caris waypoints (*.wpt)\n ASCII text (*.txt)')[0])

    def selectreportoutput(self):
        self.reportDir.setText(QFileDialog.getSaveFileName(self, 'Save Job Report', '', '(*.pdf)')[0])

    def enablerun(self):
        self.runButton.setEnabled(bool(self.inputDir.text()))

    def enabletask(self):
        if len(self.bin) > 0:
            self.execButton.setDisabled(
                (self.bounGroup.isChecked() and self.bounFileRadio.isChecked() and not bool(self.bounFileInput.text()))
                or (self.failGroup.isChecked() and self.failOutCheckbox.isChecked() and not bool(self.failDir.text()))
                or (self.ppGroup.isChecked() and self.ppOutCheckbox.isChecked() and not bool(self.ppDir.text())))

    def enablereport(self):
        if self.failgrid is not None:
            self.reportButton.setEnabled(bool(self.reportDir.text()))

    def clickgroup(self):
        if self.bounGroup.isChecked() and self.ppGroup.isChecked() and not self.failGroup.isChecked():
            self.failGroup.setChecked(True)
        self.enabletask()

    def runprogram(self):
        self.inputDirBrowseButton.setEnabled(False)
        self.inputDir.setEnabled(False)
        self.inputName.setEnabled(False)
        self.inputFeature.setEnabled(False)
        self.inputDensity.setEnabled(False)
        self.inputDiag.setEnabled(False)
        self.runButton.setEnabled(False)
        self.watcher = Watcher(self.inputDir.text())
        self.reader = Reader(self.inputFeature.value(), self.inputDensity.value(), self.inputDiag.value())
        self.watcher.watch_signal.connect(self.run_reader)
        self.reader.bin_signal.connect(self.setbin)
        self.reader.data_signal.connect(self.drawmap)
        watchdog_worker = Worker(self.watcher.startwatch)
        self.threadpool.start(watchdog_worker)
        self.consoleBox.appendPlainText('Program started.\nWatching {}'.format(self.inputDir.text()))
        files = [f for f in os.listdir(self.inputDir.text()) if f.endswith('.txt')]
        worker = Worker(self.watcher.prewatch, files)
        self.threadpool.start(worker)

    def exectask(self):
        self.execButton.setEnabled(False)
        if self.failGroup.isChecked() and self.failOutCheckbox.isChecked() and self.failDir.text()[-3:] == 'tif':
            self.dialog.exec()
            if self.dialog.result() == 0:
                self.consoleBox.appendPlainText('Try again. No EPSG code was given.')
                self.enabletask()
                return
        if self.bounGroup.isChecked():
            self.run_builder()
        elif not self.bounGroup.isChecked() and self.failGroup.isChecked():
            self.run_checker()
        elif not self.bounGroup.isChecked() and not self.failGroup.isChecked() and self.ppGroup.isChecked():
            self.run_planner()
        else:
            self.enabletask()

    def run_builder(self):
        self.consoleBox.appendPlainText('Building survey area boundary... ')
        if self.bounRadio.isChecked():
            worker = Worker(self.builder.buildbound, (self.hist, self.bounSpinbox.value()))
            self.threadpool.start(worker)
        elif self.ginputRadio.isChecked():
            boun_xy = np.asarray(self.fig1.ginput(-1))
            if boun_xy.shape[0] > 2:
                self.builder.boun_signal.emit(boun_xy)
            else:
                self.consoleBox.appendPlainText('Input boundary ERROR. A minimum of 3 points must be given.')
            self.enabletask()
        else:
            boun_xy = pd.read_csv(self.bounFileInput.text(), sep=' ', usecols=[0, 1]).to_numpy()
            if boun_xy.shape[0] > 2:
                self.builder.boun_signal.emit(boun_xy)
            else:
                self.consoleBox.appendPlainText('Input boundary ERROR. A minimum of 3 points must be given.')
                self.enabletask()

    def run_checker(self):
        self.consoleBox.appendPlainText('Checking grid compliance... ')
        if self.boundary is not None:
            worker = Worker(self.checker.checkgrid, (self.hist, self.boundary, self.inputCoverage.value()))
            self.threadpool.start(worker)
        else:
            self.consoleBox.appendPlainText('Fail to check compliance. Boundary must be built first.')
            self.enabletask()

    def run_planner(self):
        self.consoleBox.appendPlainText('Planning path for repairing data... ')
        if self.failgrid is not None:
            if self.failgrid.shape[0] > 0:
                worker = Worker(self.planner.planpath,
                                (self.failgrid, self.depth, self.swathSpinbox.value(), self.ppSpinBox.value()))
                self.threadpool.start(worker)
            else:
                self.consoleBox.appendPlainText('No path planning is required. All grids are compliant.')
                self.pathplot.set_data([], [])
                self.canvas2.draw()
                self.enabletask()
        else:
            self.consoleBox.appendPlainText('Fail to plan path. Grid compliance must be checked first.')
            self.enabletask()

    def run_reporter(self):
        self.consoleBox.appendPlainText('Writing job report... ')
        info = 'Job name: {}\n'.format(self.inputName.text()) + \
               'Report generated on: {}\n'.format(str(datetime.now())[:-7]) + \
               'Survey standards: \n' \
               '      Feature detection: {} m\n'.format(self.inputFeature.value()) + \
               '      Bathymetric coverage: {}00%\n'.format(self.inputCoverage.value()) + \
               '      Minimum data densiy: {} points per feature\n'.format(self.inputDensity.value()) + \
               'Total lines processed: {}\n'.format(self.line_no) + \
               self.failtext.get_text()
        worker = Worker(self.reporter.genreport,
                        (info, self.hist, self.bin, self.xylim, self.boundary, self.failgrid, self.reportDir.text()))
        self.threadpool.start(worker)

    @pyqtSlot(str)
    def run_reader(self, sval):
        self.consoleBox.appendPlainText('Reading {}... '.format(sval))
        worker = Worker(self.reader.readline, sval)
        self.threadpool.start(worker)

    @pyqtSlot(tuple)
    def setbin(self, tup):
        self.bin = tup
        self.builder = Builder(self.bin)
        self.checker = Checker(self.bin)
        self.builder.boun_signal.connect(self.drawbound)
        self.checker.fail_signal.connect(self.drawfail)
        self.planner.plan_signal.connect(self.drawpath)
        self.enabletask()

    @pyqtSlot(tuple)
    def drawmap(self, tup):
        self.hist, self.depth, self.xylim = tup
        self.line_no += 1
        self.consoleBox.appendPlainText('Loaded {} lines to coverage map.'.format(self.line_no))
        self.ax1.set_xlim(self.xylim[0][0] - 5, self.xylim[1][0] + 5)
        self.ax1.set_ylim(self.xylim[0][1] - 5, self.xylim[1][1] + 5)
        self.ax2.set_xlim(self.xylim[0][0] - 5, self.xylim[1][0] + 5)
        self.ax2.set_ylim(self.xylim[0][1] - 5, self.xylim[1][1] + 5)
        self.ax1.set_title('Coverage Map of {} Lines'.format(self.line_no))
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
    def drawbound(self, boun_xy):
        self.consoleBox.insertPlainText('Done.')
        self.boundary = boun_xy
        # noinspection PyArgumentList
        self.linepatch.set_xy(boun_xy)
        self.canvas2.draw()

        if self.failGroup.isChecked():
            self.run_checker()
        else:
            self.enabletask()

    @pyqtSlot(object)
    def drawfail(self, fail_xy):
        self.consoleBox.insertPlainText('Done.')
        self.failgrid = fail_xy
        bounarea = getpolyarea(self.boundary)
        failarea = fail_xy.shape[0] * (self.inputFeature.value() ** 2)
        failrate = min(failarea / bounarea, 1)
        self.failtext.set_text(
            'Compliant Grids (Green): {:.2%}\nNon-compliant Grids (Red): {:.2%}'.format(1 - failrate, failrate))
        # noinspection PyArgumentList
        self.polypatch.set_xy(self.boundary)
        self.failplot.set_data(fail_xy[:, 0], fail_xy[:, 1])
        self.canvas2.draw()

        if self.failOutCheckbox.isChecked():
            if self.failDir.text()[-3:] == 'tif':
                fig = Figure(figsize=(self.xylim[1][0] - self.xylim[0][0], self.xylim[1][1] - self.xylim[0][1]),
                             frameon=False)
                ax = fig.add_axes((0, 0, 1, 1), aspect='equal')
                ax.set_axis_off()
                ax.set_xlim(self.xylim[0][0], self.xylim[1][0])
                ax.set_ylim(self.xylim[0][1], self.xylim[1][1])
                patch = Polypatch(self.boundary, edgecolor='None', facecolor='grey', alpha=0.5, zorder=0,
                                  antialiased=True)
                ax.add_patch(patch)
                ax.plot(self.failgrid[:, 0], self.failgrid[:, 1], c='yellow', marker="s", mew=0, lw=0,
                        ms=72 * self.inputFeature.value(), alpha=0.5, zorder=1, antialiased=True)
                fig.savefig(self.failDir.text(), dpi=6, transparent=True)
                failtif = gdal.Open(self.failDir.text(), gdal.GA_Update)
                failtif.SetProjection(self.dialog.epsg.ExportToWkt())
                failtif.SetGeoTransform([self.xylim[0][0], 1 / 6, 0, self.xylim[1][1], 0, -1 / 6])
                failtif.FlushCache()
                failtif = None
            elif self.failDir.text()[-3:] == 'wpt':
                txt = ''.join('[WAYPOINT({})]\nType = 0\nXwp = {}\nYwp = {}\nZwp = 0\nName = {}\nTol1 = 0\nTol2 = 0\n\n'
                              .format(i + 1, x[0], x[1], str(datetime.now())[:-7]) for i, x in enumerate(self.failgrid))
                wptfile = open(self.failDir.text(), 'w')
                wptfile.write(txt)
                wptfile.close()
            else:
                np.savetxt(self.failDir.text(), self.failgrid, fmt='%.3f', header='Easting Northing', comments='')

        if self.ppGroup.isChecked():
            self.run_planner()
        else:
            self.enabletask()
        self.enablereport()

    @pyqtSlot(object)
    def drawpath(self, waypt):
        self.consoleBox.insertPlainText('Done.')
        self.pathplot.set_data(waypt[:, 0], waypt[:, 1])
        self.canvas2.draw()

        if self.ppOutCheckbox.isChecked():
            if self.ppDir.text()[-3:] == 'wpt':
                txt = ''.join('[WAYPOINT({})]\nType = 0\nXwp = {}\nYwp = {}\nZwp = 0\nName = {}\nTol1 = 0\nTol2 = 0\n\n'
                              .format(i + 1, x[0], x[1], str(datetime.now())[:-7]) for i, x in enumerate(waypt))
                wptfile = open(self.ppDir.text(), 'w')
                wptfile.write(txt)
                wptfile.close()
            else:
                np.savetxt(self.ppDir.text(), waypt, fmt='%.3f', header='Easting Northing', comments='')

        self.enabletask()

    @pyqtSlot()
    def endreport(self):
        self.consoleBox.insertPlainText('Done.')
        self.enablereport()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(8)
    app.setFont(font)
    win = Window()
    win.show()
    sys.exit(app.exec_())
