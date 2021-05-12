# CoverTest Program
Coverage test of bathymetric data based on IHO S44 Edition 6.0

## Getting Started
To use this program on Windows without setting up Python, download the ```build/exe.win-amd64-3.9/``` directory and launch the ```CoverTest.exe``` executable file. The executable was packed by using [cx_freeze](https://cx-freeze.readthedocs.io/en/latest/).

### Setting Up Python Environment
Use **Python 3.9** and install all the dependencies with:
```
pip install -r requirements.txt
```
Next, run the script file as below:
```
python main.py
```

## Usage
The program accepts semi-processed bathymetric data (*ASCII* files with the ```.txt``` extension). Each input file should contain data collected from a **single** trackline. The input file must be in the following format:

* First row: Header row
* First column: Easting in *metres*
* Second column: Northing in *metres*
* Third column: Depth in *metres*

At startup, the CoverTest program will bring up a frontend UI that allows users to specify survey standards and view test results.

<p align="center">
<img src="screenshot.png" width="700px" title="Program Demo">
</p>

### Test Configurations
The ```Configurations``` section allows users to select an input file directory and specify the required survey standards. 

* ```Line File Directory``` is where the input files are stored. The program uses the [watchdog](https://pythonhosted.org/watchdog/) module to monitor file system events in the selected directory. The program will automatically check for existing and newly added ```.txt``` files in the selected directory.

* ```Job Name``` parameter allows users to set the name recorded in the test report. It does not affect the test results. By default, it is named after the current date.

* ```Feature Size``` is the required size of feature detection in *metres*. It will be used as the grid size for constructing the bathymetric coverage map. The default value is 1.0 metre.

* ```Minimum Data Density``` is the minimum number of datapoints required by every grid. Grids with datapoints less than this parameter will be rejected in the bathymetric coverage. The default value is 1.

* ```Approx. Survey Diagonal``` is the approximated (maximum) diagonal size of the survey area. It will be used for constructing the grids of the bathymetric coverage map. The given diagonal size must be larger than the actual size to construct the coverage map correctly. The default value is 1000 metres.

Clicking on the ```Run Program``` button will start running the CoverTest program and prevent further changes to the selected directory and configurations. A restart will be required to make any changes. When the program is running, existing and newly added files in the selected directory will be read to compute a bathymatric coverage map. The program uses **Binary Search** to bin the data into the grids of the coverage map. After reading the first input file, the left plot window will display the coverage map.

### Data Analysis
The ```Data Analysis``` section allows users to analyse and check input data against the required survey standards. Users have the option to select one or more of the following tasks (by checking the boxes).

* ```Build Boundary``` - Users can build the boundary of survey areas for analysis. After executing this task, the right plot window will display the resultant boundary. Boundary can be built by using one of the following methods:
  * ```Manual Graphical Input``` - Build boundary manually by clicking in sequence on the coverage map (left plot window). Left-click to select; right-click to deselect; mid-click to confirm selection.
  * ```Use Current Data``` - Build boundary automatically based on the current input data. [Delaunay Triangulation](https://gist.github.com/AndreLester/589ea1eddd3a28d00f3d7e47bd9f28fb) is used to build a concave hull that will enclose data of a desired bathymetric coverage. Users can specify the desired ```Coverage``` (default value is 1).
  * ```Input File``` - Build boundary from an *ASCII* file. The input file must have a header row, an Easting (first) column in *metres*, and a Northing (second) column in *metres*.

* ```Check Compliance``` - Users can check the compliance of data against the survey standards by specifying a desired ```Minimum Coverage``` (default value is 1). This task can only be executed after building a survey area boundary. The program uses a fast [point-in-polygon](https://github.com/dengwirda/inpoly-python) check to select the non-compliant grids. After executing this task, the right plot window will display the grid compliance map. Users have the option to output the grid compliance map

* ```Path Planning``` - 

### Test Results


### Test Reporting


## Built With
* [PyQt5](https://doc.qt.io/qtforpython/) - UI framework used.
* [QThreadPool](https://doc.qt.io/qt-5/qthreadpool.html) - Multi-threading framework used.

## Authors
* Hui Sheng Lim ([xanxustyle](https://github.com/xanxustyle)) - Developer.
* Nathan Green - Project advisor/supervisor.

## License
This project is released under the GNU General Public License v3.0 license - see the [LICENSE.md](LICENSE.md) file for details
