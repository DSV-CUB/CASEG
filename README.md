# CASEG - cascaded segmentation

## Preparation
In order to use the scripts, please make sure:
- install Python 3.8 and the required site packages, please have a look for the installation of rasterio under Windows below (other OS are not tested)
- the data is given as <SOPInstanceUID>.dcm (dicom image) and <SOPInstanceUID>.pickle (reference segmentation) in the same directory

## Installation
- Download and install Python 3.8 from https://www.python.org/downloads/
- install rasterio (instructions work for Windows 10 and is adapted from the instructions at https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows):
    - go to http://www.gisinternals.com/release.php and click on release-1928-x64-gdal-3-4-1-mapserver-8-0-0
        - download and install gdal-304-1928-x64-core.msi
        - download and install GDAL-3.4.1.win-amd64-py3.8.msi
    - add system environment variables in Windows
        - in the PATH variable add the path to the GDAL installation folder (i.e. C:\Program Files\GDAL)
        - add a GDAL_DATA variable (i.e. pointing to C:\Program Files\GDAL\gdal-data)
        - add a GDAL_DRIVER_PATH variable (i.e. pointing to C:\Program Files\GDAL\gdalplugins)
    - check GDAL installation by opening a command prompt and type in gdalinfo --version
    - download and install rasterio-1.2.10-cp38-cp38-win_amd64.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/
    - download and install VC redist from https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170
    - restart the PC
- pip install the requirements.txt

## Reference segmentation
The binary pickle file contains the reference segmentation with the following structure:

segmentationdata = dict{saepicardialContour, saendocardialContour}

with

saepicardialContour = list{numpy.array[Nepi,2], shape_of_original_image_x, shape_of_original_image_y}

saendocardialContour = list{numpy.array[Nendo,2], shape_of_original_image_x, shape_of_original_image_y}

where Nepi and Nendo denotes the numberof points in order to define the epi- and endocardial contour respectively.

For training and testing, the dicom image and the corresponding pickle file with the ground truth segmentation must be
stored in the same directory and match names. We recommend both having the SOPInstanceUID as filename, for the pickle file it is mandatory.

## Scripts
The used scripts for training and testing are provided in CASEG\scripts.
The marissa dict is from another project but contains necessary code files that are used here.
For usage, the scripts must be adapted in the paths pointing to the data.
