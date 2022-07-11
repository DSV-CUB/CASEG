import pydicom
import os
import numpy as np

def get_demographics(path):
    ages = []
    gender = []
    for root, _, files in os.walk(path):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
                ages.append(int(str(dcm[0x0010, 0x1010].value)[:-1]))
                gender.append(str(dcm[0x0010, 0x0040].value))
            except:
                pass
    ages = np.array(ages)
    gender = np.array(gender)

    print(str(np.mean(ages)) + "+/-" + str(np.std(ages)) + " | " + str(np.count_nonzero(gender=="M")) + " M | " + str(np.count_nonzero(gender=="F")) + " F")
    print("age range:" + str(np.min(ages)) + " " + str(np.max(ages)))
    return



if __name__ == "__main__":
    path_train = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TRAIN"
    path_validation = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\VALIDATION"
    path_test = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"
    path_test_MV = r"D:\ECRC_AG_CMR\3 - Promotion\Project CASEG\3 - Measurements\FULL DATASETS\EXPERT SPLIT\EXPERT FULL SAX_SPLIT\TEST_MV"

    get_demographics(path_train)
    get_demographics(path_validation)
    get_demographics(path_test)
    get_demographics(path_test_MV)
