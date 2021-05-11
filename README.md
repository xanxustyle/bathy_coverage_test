# CoverTest Program
Coverage test of bathymetric data based on IHO S44 Edition 6.0

## Getting Started
For anyone wishes to use this program on Windows without setting up Python, the easiest way is to download the ```build/exe.win-amd64-3.9/``` directory and launch the ```CoverTest.exe``` executable file. The executable was packed by using [cx_freeze](https://pypi.org/project/cx-Freeze/).

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
At startup, the CoverTest program will bring up a frontend UI that allows a user to specify survey standards and view test results.

<img src="screenshot.png" width="700px" title="Program Demo"></img>

### Test Configurations


### Data Analysis


### Test Reporting


## Built With
* [PyQt5](https://doc.qt.io/qtforpython/) - UI framework used.
* [QThreadPool](https://doc.qt.io/qt-5/qthreadpool.html) - Multi-threading framework used.

## Authors
* Hui Sheng Lim ([xanxustyle](https://github.com/xanxustyle)) - Developer.
* Nathan Green - Project advisor/supervisor.

## License
This project is released under the GNU General Public License v3.0 license - see the [LICENSE.md](LICENSE.md) file for details
