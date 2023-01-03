# CASEG - cascaded segmentation

This sourcecode is part of the publication

Viezzer, D. et al. Introduction of a cascaded segmentation pipeline for parametric T1 mapping in cardiovascular magnetic 
resonance to enhance segmentation performance. tba. https://doi.org/tba

## Preparation
In order to use the scripts, please make sure:
- install Python 3.8 and the required site packages, please have a look for the installation of rasterio under Windows below (other OS are not tested)
- the data is given as <SOPInstanceUID>.dcm (dicom image) and <SOPInstanceUID>.pickle (reference segmentation) in the same directory

## Installation
- Download and install Python 3.8 from https://www.python.org/downloads/
- install rasterio (instructions work for Windows 10 and is adapted from the instructions at https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows):
    - Install Visual Studio from
        - https://visualstudio.microsoft.com/de/ 
    - Install Visual Studio Build Tools 2019 and check during installation the workload C++ BuildTools by running in the command prompt:
        - winget install --id=Microsoft.VisualStudio.2019.BuildTools -e
    - Install VC-Redist
        - https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170 
    - Install GDAL for windows
        - https://www.gisinternals.com/release.php 
        - choose the MSVC 2019 Compiler -> x64 for 64bit Systems -> release-1928-x64-gdal-3-5-3-mapserver-8-0-0 (could be different in future, just check that the year is according to the Visual Studio year and the number 1928 is equal or bigger than the MSVC version of your python)
        - Download and install the core msi: gdal-305-1928-x64-core.msi (could be different in the future)
        - Download and install the python libraries (be aware of your version, in this SOP its 3.8): GDAL-3.5.3.win-amd64-py3.8.msi
        - Add GDAL to system environment variables
            - Go to Start  seach for variables  click on the variables button  go in the lower box (system variables)
            - Search for the Path entry  double click  Add new
            - Enter the Location of GDAL, i.e.: C:\Program Files\GDAL
            - Click ok
            - Now under system variables click add new and insert the following two
            - Name: GDAL_DATA / Value: C:\Program Files\GDAL\gdal-data
            - Name: GDAL_DRIVER_PATH / Value: C:\Program Files\GDAL\gdalplugins
            - Ok and leave
    - Overwrite Python GDAL with other version
        - https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
        - Search for GDAL
        - Download the suitable GDAL wheel for your Python application, i.e. GDAL‑3.4.3‑cp38‑cp38‑win_amd64.whl
        - Install with command in command prompt:
            - pip install <path to wheel>
    - Install Rasterio
        - https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
        - Search for Rasterio
        - Download the suitable Rasterio wheel for your Python application, i.e. rasterio 1.2.10 cp38 cp38 win_amd64.whl
        - Install with command in command prompt:
            - pip install <path to wheel>
    - Restart PC
    - Usage in python:
        - Necessary first: from osgeo import gdal, then import rasterio
        - import rasterio.features
        - Ignore Warnings saying “ERROR 1: Can’t load requested DLL

- pip install the requirements.txt

## Reference segmentation
The binary pickle file contains the reference segmentation with the following structure:

segmentationdata = dict{saepicardialContour, saendocardialContour}

with

saepicardialContour = list{numpy.array[Nepi,2], shape_of_original_image_x, shape_of_original_image_y}

saendocardialContour = list{numpy.array[Nendo,2], shape_of_original_image_x, shape_of_original_image_y}

where Nepi and Nendo denotes the numberof points in order to define the epi- and endocardial contour respectively.
Beware that the x-y-direction in Python is row-column, but in fff column-row.

For training and testing, the DICOM image and the corresponding pickle file with the ground truth segmentation must be
stored in the same directory and match names. We recommend both having the SOPInstanceUID as filename, for the pickle file it is mandatory.

## Scripts
All necessary scripts for basic usage are in CASEG\scripts.
The marissa dict is from another project but contains necessary code files that are used here.
For usage, the paths in the scripts must be adapted.

## Implementation Details / Further Information

The output of the models need some post-processing that is already included in the models functions. The predict and
predict_generator function have multiple arguments:

x: mandatory data (predict) or generator (predict_generator)

show_raw: return the direct output of the model (default False, it is recommended to set resize to False if show_raw is True)

resize: resizes the output of the modele to the original size of the input image (default True)

lcc: run largest connected component on output (default False)

contour_level: resize of the output based on contours and not pixmap (default True, recommended True)

We recommend using data generators such that the user does not need to take care about any conversions between the object 
detection network and the segmentation network in CASEG. The generator can be fed into the model's train function. It is
necessary, to have the configuration of the model and the generator equal. The configuration has the following options:

data_rescale: convert pixel data according to the DICOM information (default True)

data_representation: convert pixel data according to DICOM information for pretty plot (default False)

data_segmentation_level: "CONTOUR" or "MASK" (default "CONTOUR")

data_plot: plot data of the generator (default False, not recommendet during model training, if True plots are inline, if 
a path is given, the plots are exported to that path)

fit_epochs: number of epochs to run the model (default 1000)

fit_epoch_length: length of epoch (default None)

fit_batch_size: batch size during training (default 10)

fit_save_last: save last state of teh model (default False)

fit_save_best_only: save only the best state (default True)

fit_monitor: what to monitor to define best state (default "loss")

fit_mode: what to check in fit_monitor (default "min")

fit_early_stopping: do early stopping if plateau is reached (default True)

fit_early_stopping_min_delta: minimum change that is assumed as improvement (default 0)

fit_early_stopping_patience: how many cycles to run with no improvement to stop early (default 50)

fit_early_stopping_stopped_epoch:

model_type: which type of CNN model (default "UNET", no other implementation existing currently)

The following configuration are U-Net specific:

model_predictor: "SEGM" or "BB" (default "SEGM", defines model mode)

model_name: default name of the model (default "MMS_unet")

model_input_size: size of the input image (default [256, 256, 1])

model_output_size: number of output channels (default 1)

model_optimizer: model training optimizer (default "adam")

model_loss: loss function of the model training (default "loss_binary_crossentropy")

model_lr: learning rate of the model (default 1e-3)

model_lr_method: method for learning rate adaption during training (default "constant")

model_metrics: list of metrics to consider (default ["accuracy"])

model_depth: number of convolutional steps (default 6)

model_dropout_rate: rate in the dropout layers (default 0.2)

model_convdepth_initial: starting convolutional depth (default 32)

model_convdepth_max: if in feature space the depth would exceed this number, then its cropped to this (default 2048)

model_crop: magnification factor, False if not used like in refU or ODA (default 1)

All non-listed attributes should not be touched attributes 

The conversion of the ODA output towards the cropU and crinU segmentation network input is automatically performed within 
the generator by providing the BB model as attribute. The attributes for the generators are:

input: mandatory, describing the path to the data

configuration: configuration (see above, default None -> setups default configuration)

generator_type: mode of the generator (default "TRAINING", otherwise "TEST")

model_bb: complete loaded BB model (default None, necessary for cropU and crinU generators)

ws_dir: directory to the ground truth segmentation (default None)

ws_type: type of segmentation (default "SAX" -> expects contours as described above)

mask_mode: mode to convert masks to contours (default "RASTERIZE", otherwise "SHAPELY" but not recommended)

If the user wants to run the conversion manually, the default output of the ODA prediction which represents a binary BB 
mask is necessary. Then, the indeces of the four edges of the BB must be evaluated and the sites enlargedaccording to the 
magnification factor. The edgepoint with the lowest x- and y- value must be stored. Then, the original image is cropped to
the section representing the enlarged BB. This image section is then put into the segmentation network
model. A novel zero filled array in the same size as the original image must be created and the output of the segmentation
network model is added to that zero valued array with shifting the values by the stored edgepoint in order to map the
final segmentation mask at the right place.
