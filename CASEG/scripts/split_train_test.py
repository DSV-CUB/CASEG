import pydicom
import os
import numpy as np
import shutil


# PATH TO DATA
path = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT FULL\EXPERT FULL SAX"
# OUTPUT PATH WITH THE DATA SPLIT
path_out = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT"

# PLEASE CONSIDER THAT THE PATH TO DATA MUST CONTAIN SUBDIRS WITH dcm ws SAX.
# Each subdir refers to a studyset. For each dicom a corresponding pickle file must exist with the ground truth contours
# and both need the SOPInstanceUID as name.

split_test = 15
split_val = 10
subdirs = np.array(os.listdir(path))

for subdir in subdirs:
    if subdir.endswith("dcm ws SAX"):
        subsubdirs = np.array(os.listdir(os.path.join(path, subdir)))

        num = len(subsubdirs)
        num_test = int((split_test/100) * num)
        num_val = int((split_val/100) * num)

        run = True
        while run:
            indeces_test = np.unique(np.random.randint(0, num, (num_test,)))
            if len(indeces_test) == num_test:
                run=False

        run = True
        while run:
            indeces_val = np.unique(np.random.randint(0, num, (num_val,)))
            if len(indeces_val) == num_val and not np.any(np.in1d(indeces_val, indeces_test)):
                run=False

        for i in range(num):
            if i in indeces_test:
                #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, subdir, subsubdirs[i]), os.path.join(path_out, "TEST", subsubdirs[i]), dirs_exist_ok=True)
            elif i in indeces_val:
                #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, subdir, subsubdirs[i]), os.path.join(path_out, "VALIDATION", subsubdirs[i]), dirs_exist_ok=True)
            else:
                #os.makedirs(os.path.join(path_out, "Expert Split Train", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, subdir, subsubdirs[i]), os.path.join(path_out, "TRAIN", subsubdirs[i]), dirs_exist_ok=True)
