import pydicom
import os
import numpy as np


path = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT FULL\EXPERT FULL SAX"
list_pixel_array = []
list_path = []
counter_duplicates = 0

for root, _, files in os.walk(path):
    for file in files:
        try:
            dcm = pydicom.dcmread(os.path.join(root, file))
            dcm.PatientName



            if np.array([np.array_equal(x, dcm.pixel_array) for x in list_pixel_array]).any():
                index = int(np.argwhere(np.array([np.array_equal(x, dcm.pixel_array) for x in list_pixel_array])))
                print(list_path[index] + "\n" + os.path.join(root, file) + "\n")
                counter_duplicates = counter_duplicates + 1
            else:
                list_pixel_array.append(dcm.pixel_array)
                list_path.append(os.path.join(root, file))
        except:
            pass

print("In total " + str(counter_duplicates) + " found.")
