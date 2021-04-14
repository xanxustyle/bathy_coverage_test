import gc
import os
from datetime import datetime
import numpy as np
from pandas import DataFrame as Df
from scipy.spatial import KDTree, distance_matrix
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as Polypatch
from matplotlib.backends.backend_pdf import PdfPages
from dask import dataframe as Ddf
from concavehull import ConcaveHull
from inpoly import inpoly2
from elkai import solve_float_matrix
from ufunclab import max_argmax


def createbin(xy_pts,edge):
    xmin, ymin = xy_pts.min(axis=0)
    xmax, ymax = xy_pts.max(axis=0)
    xmin = xmin - 0.0001
    xmax = xmax + edge - (xmax - xmin) % edge
    ymin = ymin - 0.0001
    ymax = ymax + edge - (ymax - ymin) % edge
    binedge1 = np.linspace(xmin, xmax, num=int(round((xmax - xmin) / edge)) + 1)
    binedge2 = np.linspace(ymin, ymax, num=int(round((ymax - ymin) / edge)) + 1)
    bin1 = binedge1[:-1] + edge / 2
    bin2 = binedge2[:-1] + edge / 2
    return binedge1, binedge2, bin1, bin2


def nlist2array(nlist):
    lens = np.array([len(i) for i in nlist])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, -1, dtype=int)
    out[mask] = np.concatenate(nlist)
    return out


start0 = datetime.now()
file_path = 'C:/Users/limhs/Desktop/Intern2021/Code/TEST'
jobname = 'testrun'
feature = 1         # Feature detection [m]
coverage = 2        # Bathymetric coverage [x100%]
boundary_cover = 1  # Minimal coverage from drawing boundary [x100%]
swath_angle = 110   # Nominal swath angle [deg]

# Read files
print('Reading line files...', end=" ")
data = np.empty((0, 4))
line_no = 0
tmp = []
for file in [f for f in os.listdir(file_path) if f.endswith('.txt')]:
    tmp.append(Ddf.read_csv(os.path.join(file_path, file), sep=' ', usecols=[0, 1, 2]))
    tmp[line_no]['Line'] = (tmp[line_no]["Depth"]*np.nan).fillna(line_no)
    line_no += 1
# tmp = ddf.read_csv('C:/Users/limhs/Desktop/Intern2021/Code/Hob_TasPort/*.txt', sep=' ', usecols=[0, 1, 2]).compute()
data = Ddf.concat(tmp).values.compute()
del tmp
print(str(line_no) + ' lines copied.')
print(datetime.now() - start0)

# Create bin
swath_width = 2 * data[:,2].mean() * np.tan(np.deg2rad(swath_angle / 2))
binEdge_east, binEdge_north, bin_east, bin_north = createbin(data[:,:2],feature)

# Grid data into bin
print('Gridding data...', end=" ")
start = datetime.now()
data[:,0] = np.searchsorted(binEdge_east, data[:,0], side='right')
data[:,1] = np.searchsorted(binEdge_north, data[:,1], side='right')
data = data[Df(data[:,[0,1,3]]).drop_duplicates().index]
print('Done.')
print(datetime.now() - start)

# Compute coverage map
print('Computing coverage map...', end=" ")
H = np.histogram2d(data[:,0] - 0.5, data[:,1] - 0.5,
                   bins=(np.arange(binEdge_east.shape[0]), np.arange(binEdge_north.shape[0])))[0]
xedge, yedge = np.meshgrid(binEdge_east, binEdge_north)
print('Done.')

# Build coverage boundary
print('Building coverage boundary...', end=" ")
hull = ConcaveHull()
hull_east, hull_north = np.nonzero(H >= boundary_cover)
hull.loadpoints(np.column_stack((bin_east[hull_east], bin_north[hull_north])))
hull.calculatehull(tol=(boundary_cover**4)+3)
boundary_pts = np.column_stack(hull.boundary.exterior.coords.xy)
print('Done.')

# Select non-compliant grids
print('Identifing non-compliant grids...', end=" ")
fail_east, fail_north = np.nonzero(H < coverage)
fail_pts = np.column_stack((bin_east[fail_east], bin_north[fail_north]))
fail_pts = fail_pts[inpoly2(fail_pts, boundary_pts)[0]]
print('Done.')

del fail_east,fail_north,hull_east,hull_north
gc.collect()

# Group non-compliant grids
print('Grouping non-compliant grids...', end=" ")
r = swath_width / 2

fbinEdge_east, fbinEdge_north, fbin_east, fbin_north = createbin(fail_pts, r/2)
fail_tree = KDTree(fail_pts)
fail_grp = np.column_stack((np.searchsorted(fbinEdge_east, fail_pts[:,0], side='right'),
                            np.searchsorted(fbinEdge_north, fail_pts[:,1], side='right')))
fail_grp = fail_grp[Df(fail_grp).drop_duplicates().index].astype(float)
fail_grp[:,0] = fbin_east[fail_grp[:,0].astype(int)-1]
fail_grp[:,1] = fbin_north[fail_grp[:,1].astype(int)-1]
fail_grp = fail_pts[fail_tree.query(fail_grp,workers=-1)[1]]

fail_tree = KDTree(fail_grp)
neighbor = fail_tree.query_ball_tree(fail_tree, r)
start = datetime.now()
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
print('Done')
print(datetime.now() - start)

# Find shortest path
print('Path planning...', end=" ")
start = datetime.now()
dist_mat = distance_matrix(waypt, waypt, threshold=1e10)
best_path = solve_float_matrix(dist_mat, runs=100)
waypt = waypt[best_path]
print('Done')
print(datetime.now() - start)

print(datetime.now() - start0)
fig = plt.figure(figsize=(8, 8))
axe = fig.add_subplot(111, xlabel='Easting [m]', ylabel='Northing [m]', aspect='equal')
axe.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'black', linewidth=0.5)
axe.scatter(fail_pts[:,0], fail_pts[:,1], marker=".", c='r', s=3, linewidths=0.01, zorder=1)
axe.scatter(waypt[:,0], waypt[:,1], s=10, c='lime', alpha=1)
axe.plot(waypt[:,0], waypt[:,1], c='lime', alpha=1)
# # axe.scatter(fail_grp[neighbor[200],0], fail_grp[neighbor[200],1], s=7, c='blue', alpha=1)
# # axe.scatter(fail_grp[200,0], fail_grp[200,1], s=20, c='lime', alpha=1)
# from matplotlib.patches import Circle
# for i in range(waypt.shape[0]):
#     cir = Circle((waypt[i,0],waypt[i,1]),r,color='blue',fill=False)
#     axe.add_patch(cir)
plt.show()
stop = 0

# Write job report
print('Writing job report...', end=' ')
report = 'Job name: ' + jobname + '\n' \
         'Report generated on: ' + str(datetime.now())[:-7] + '\n' \
         'Survey standards: \n' \
         '      Feature detection: ' + str(feature) + 'm\n' \
         '      Bathymetric coverage: ' + str(coverage*100) + '%\n' \
         'Total lines processed: ' + str(line_no) + '\n' \
         'Default swath angle: ' + str(swath_angle) + '\N{DEGREE SIGN}\n' \
         ''

with PdfPages('C:/Users/limhs/Desktop/job_report_' + jobname + '.pdf') as pdf:
    fig1 = plt.figure(figsize=(8.3, 11.7))
    ax = fig1.add_subplot(111)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(which='both', colors='white')
    plt.figtext(0.12, 0.9, report, family='calibri', size=15, linespacing=2.5, ha='left', va='top')
    pdf.savefig()
    plt.close()

    fig2 = plt.figure(figsize=(8.3, 11.7))
    gridspec = fig2.add_gridspec(2, 2, width_ratios=[1, 0.04])
    ax1 = fig2.add_subplot(gridspec[0, 0], xlabel='Easting [m]', ylabel='Northing [m]', aspect='equal', rasterized=True)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_scientific(False)
    cmax = int(np.max(H))
    cmap = plt.get_cmap('viridis', cmax + 1)
    cmesh = ax1.pcolormesh(xedge, yedge, H.T, cmap=cmap)
    ax1.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'black', linewidth=0.5)
    # ax1.scatter(fail_pts[:, 0], fail_pts[:, 1], s=0.05, c='red', alpha=0.2)
    ax1.set_anchor((0.8,1))
    plt.title('Coverage Map')
    ax2 = fig2.add_subplot(gridspec[0, 1], rasterized=True)
    cbar = fig2.colorbar(cmesh, cax=ax2,
                         ticks=np.linspace(cmax/(cmax+1)/2, cmax-cmax/(cmax+1)/2, cmax+1))
    cbar.ax.set_yticklabels(np.arange(cmax + 1))
    ax2.set_anchor('W')

    ax3 = fig2.add_subplot(gridspec[1, 0], xlabel='Easting [m]', ylabel='Northing [m]', aspect='equal', rasterized=True)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_scientific(False)
    ax3.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'black', linewidth=0.5)
    patch = Polypatch(boundary_pts, color='lime', alpha=0.5, zorder=0)
    ax3.add_patch(patch)
    ax3.scatter(fail_pts[:,0], fail_pts[:,1], marker=".", c='r', s=3, linewidths=0.01, zorder=1)
    ax3.set_anchor((0.8,1))
    plt.title('Compliant regions (green). Non-compliant regions (red)')
    pdf.savefig(dpi=600)
    plt.close()

    # PDF metadata
    d = pdf.infodict()
    d['Title'] = ''
    d['Author'] = ''
    d['Subject'] = ''
    d['Keywords'] = ''
    d['CreationDate'] = datetime.now()
    d['ModDate'] = datetime.today()
print('Done')
