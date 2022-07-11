import numpy as np
import pydicom
import os
import cv2
import copy
from PIL import Image
import shutil

from marissa.toolbox.tools import tool_general, tool_pydicom, tool_cvi
from marissa.modules.datahandler.dicom import configuration


class Setup:
    def __init__(self, config=None, **kwargs):

        if config is None:
            self.configuration = configuration.Setup()
        else:
            self.configuration = config

        path = kwargs.get("path", None)

        self.configuration.set(measure=kwargs.get("measure", None),
                               filter=kwargs.get("filter", "RELEVANT").upper(),
                               tissue=kwargs.get("tissue", None))

        self.path = []
        self.pixel_array = []
        self.tags = []

        self.load(path)

        return

    def load(self, path):
        # dicom tags of interest
        tag_dict = {}
        for tag in self.configuration.toi_addresses:
            tag_dict["tag" + str(tag)] = None

        if path is not None:
            path_list = []

            # run through path and list all files found
            if os.path.isfile(path):
                if path.lower().endswith(".txt"):  # list of dcms from txt file
                    file_paths = tool_general.read_file_and_split(path)
                    for file_path in file_paths:
                        path_list.append(file_path[0])
                else:  # file was given
                    # if path.endswith(".dcm") or path.endswith(".IMA"):  # direct dicom file
                    path_list.append(path)
            else:
                for root, _, files in os.walk(path):
                    for file in files:
                        path_list.append(os.path.join(root, file))

            # run through found files and read if those are dicom files
            for path_element in path_list:
                try:
                    obj_dcm = pydicom.dcmread(path_element)
                    obj_dcm[0x0008, 0x0008].value
                except:
                    continue

                if self.configuration.measure is None:
                    pass
                elif self.configuration.measure.upper() == "T1MAP":
                    if ("MOCO" in obj_dcm[0x0008, 0x0008].value or "T1 MAP" in obj_dcm[0x0008, 0x0008].value) and "T1" in obj_dcm[0x0008, 0x0008].value:
                        pass
                    else:
                        continue
                elif self.configuration.measure.upper() == "T2MAP":
                    if "MOCO" in obj_dcm[0x0008, 0x0008].value and "T2" in obj_dcm[0x0008, 0x0008].value:
                        pass
                    else:
                        continue
                elif self.configuration.measure.upper() == "T1T2MAP":
                    if (("MOCO" in obj_dcm[0x0008, 0x0008].value or "T1 MAP" in obj_dcm[0x0008, 0x0008].value) and "T1" in obj_dcm[0x0008, 0x0008].value) or ("MOCO" in obj_dcm[0x0008, 0x0008].value and "T2" in obj_dcm[0x0008, 0x0008].value):
                        pass
                    else:
                        continue
                else:
                    raise ValueError("Quantification Type unknown. Use one of the following types: None, T1MAP, T2MAP, T1T2MAP")

                if self.configuration.filter == "RELEVANT":
                    obj_dcm_tag = copy.deepcopy(tag_dict)
                    for tag in self.configuration.toi_addresses:
                        try:
                            obj_dcm_tag["tag" + str(tag)] = obj_dcm[tag].value
                        except:
                            pass
                elif self.configuration.filter == "ALL":
                    obj_dcm_tag = {}
                    for i in range(len(list(obj_dcm.keys()))):
                        try:
                            obj_dcm_tag["tag" + str(int(list(obj_dcm.keys())[i]))] = obj_dcm[list(obj_dcm.keys())[i]].value
                        except:
                            pass
                else:
                    raise ValueError("Filter unknown, use RELEVANT or ALL")

                self.path.append(path_element)
                self.pixel_array.append(obj_dcm.pixel_array)
                self.tags.append(obj_dcm_tag)

            result = True
        else:
            result = False
        return result

    def get_pixel_data(self, indices=None, **kwargs):
        if indices is None:
            run_indices = range(len(self.path))
        elif isinstance(indices, int):
            run_indices = [indices]
        else:
            run_indices = indices

        result = []

        if len(self.path) > np.max(run_indices):
            for i in run_indices:
                pixel_data = tool_pydicom.get_pixel_data(self.pixel_array[i], self.tags[i], **kwargs)
                result.append(pixel_data)

        if len(result) == 1:
            result = result[0]

        return result

    def get_rgb(self, image_type, **kwargs):
        indices = kwargs.get("indices", range(len(self.path)))

        result = []

        if len(self.path) > 0:
            for i in indices:
                img = self.get_pixel_data(indices=[i])[0]
                img_2d = img.astype(float)
                img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
                img_2d_scaled = np.uint8(img_2d_scaled)

                img_2d_rgb = np.dstack((img_2d_scaled, img_2d_scaled, img_2d_scaled))

                if image_type.upper() == "PIL":
                    result.append(Image.fromarray(img_2d_rgb))
                elif image_type.upper() == "CV2":
                    result.append(cv2.cvtColor(np.array(Image.fromarray(img_2d_rgb)), cv2.COLOR_RGB2BGR))
                else:
                    raise ValueError("Type must be either PIL or CV2")

        return result

########################################################################################################################
########################################################################################################################
########################################################################################################################

def extract_midventricular(path, include_pca = False):
    path_out = os.path.dirname(path)
    path_out = os.path.join(path_out, path.replace(path_out, "")[1:])

    if include_pca:
        path_out = path_out + "_MIT_native_pca"
    else:
        path_out = path_out + "_MIT_native"

    for root, _, files in os.walk(path):
        data_list = []
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
                UID = dcm[0x0008, 0x0018].value

                data = []

                data.append(UID) # SOP UID
                data.append(dcm[0x0008, 0x103e].value) # Series Description

                if "4cv" in data[-1].lower() or "3cv" in data[-1].lower() or "2cv" in data[-1].lower():
                    continue

                if (0x0018, 0x0010) in dcm or "post" in dcm[0x0008, 0x103e].value.lower(): # Post Contrast
                    data.append(1)
                else:
                    data.append(0)

                data.append(dcm[0x0020, 0x0013].value) # Instance Number
                data.append(float(dcm[0x0020, 0x1041].value)) # Slice Location
                data.append(file) #file name

                data_list.append(data)
            except:
                pass

        indeces_n = np.array([])
        indeces_ca = np.array([])

        if len(data_list) == 1:
            if data_list[0][2] == 1:
                if include_pca:
                    indeces_ca = np.array([[0]])
            else:
                indeces_n = np.array([[0]])

        elif len(data_list) > 1:
            data_list = np.array(data_list)
            ca = data_list[:,2].astype(int)
            indeces_n = np.argwhere(ca==0)

            if len(indeces_n) > 1:
                check_name = ["mit" in x for x in data_list[indeces_n,1].flatten()]
                max_instance = np.max(data_list[indeces_n,3].flatten().astype(int))
                if len(np.argwhere(check_name)) > 0:
                    indeces_n = indeces_n[np.argwhere(check_name).flatten()]
                elif max_instance > 1:
                    mit_instance = int(np.ceil(max_instance/2))
                    check_instance = [x == mit_instance for x in data_list[indeces_n,3].flatten().astype(int)]
                    indeces_n = indeces_n[np.argwhere(check_instance).flatten()]
                else:
                    slice_location_max = np.max(data_list[indeces_n,4].flatten().astype(float))
                    slice_location_min = np.min(data_list[indeces_n,4].flatten().astype(float))
                    slice_location_mid = (slice_location_max - slice_location_min) / 2 + slice_location_min
                    indeces_n = indeces_n[np.argmin(np.abs(data_list[indeces_n,4].flatten().astype(float) - slice_location_mid)).flatten()]

            if include_pca:
                indeces_ca = np.argwhere(ca==1)
                if len(indeces_ca) > 1:
                    check_name = ["mit" in x for x in data_list[indeces_ca,1].flatten()]
                    if len(np.argwhere(check_name)) > 0:
                        indeces_ca = indeces_ca[np.argwhere(check_name).flatten()]
                    else:
                        max_instance = np.max(data_list[indeces_ca,3].flatten().astype(int))
                        mit_instance = int(np.ceil(max_instance/2))
                        check_instance = [x == mit_instance for x in data_list[indeces_ca,3].flatten()]
                        indeces_ca = indeces_ca[np.argwhere(check_instance).flatten()]

        path_data = root.replace(path, path_out)
        os.makedirs(path_data, exist_ok=True)

        for indece in [indeces_n, indeces_ca]:
            for i in indece.flatten():
                shutil.copyfile(os.path.join(root, data_list[i][-1]), os.path.join(path_data, data_list[i][-1]))
                try:
                    shutil.copyfile(os.path.join(root, data_list[i][0] + ".pickle"), os.path.join(path_data, data_list[i][0] + ".pickle"))
                except:
                    pass

    return path_out

def extract_sax(path, include_pca = False):
    path_out = os.path.dirname(path)
    path_out = os.path.join(path_out, path.replace(path_out, "")[1:])

    if include_pca:
        path_out = path_out + "_SAX_native_pca"
    else:
        path_out = path_out + "_SAX_native"

    for root, _, files in os.walk(path):
        data_list = []
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
                UID = dcm[0x0008, 0x0018].value

                data = []

                data.append(UID) # SOP UID
                data.append(dcm[0x0008, 0x103e].value) # Series Description

                if "4cv" in data[-1].lower() or "3cv" in data[-1].lower() or "2cv" in data[-1].lower():
                    continue

                if (0x0018, 0x0010) in dcm or "post" in dcm[0x0008, 0x103e].value.lower(): # Post Contrast
                    data.append(1)
                else:
                    data.append(0)

                data.append(dcm[0x0020, 0x0013].value) # Instance Number
                data.append(float(dcm[0x0020, 0x1041].value)) # Slice Location
                data.append(file) #file name

                data_list.append(data)
            except:
                pass

        indeces_n = np.array([])
        indeces_ca = np.array([])

        if len(data_list) == 1:
            if data_list[0][2] == 1:
                if include_pca:
                    indeces_ca = np.array([[0]])
            else:
                indeces_n = np.array([[0]])

        elif len(data_list) > 1:
            data_list = np.array(data_list)
            ca = data_list[:,2].astype(int)
            indeces_n = np.argwhere(ca==0)

            if include_pca:
                indeces_ca = np.argwhere(ca==1)

        path_data = root.replace(path, path_out)
        os.makedirs(path_data, exist_ok=True)

        for indece in [indeces_n, indeces_ca]:
            for i in indece.flatten():
                shutil.copyfile(os.path.join(root, data_list[i][-1]), os.path.join(path_data, data_list[i][-1]))
                try:
                    shutil.copyfile(os.path.join(root, data_list[i][0] + ".pickle"), os.path.join(path_data, data_list[i][0] + ".pickle"))
                except:
                    pass

    return path_out

def extract_data(path, type="T1MAP"):
    path_out = os.path.dirname(path)
    path_out = os.path.join(path_out, path.replace(path_out, "")[1:] + "_" + type.upper())

    for root, _, files in os.walk(path):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
                UID = str(dcm[0x0008, 0x0018].value)

                if type.upper() == "T1MAP" and ("MOCO" in dcm[0x0008, 0x0008].value or "T1 MAP" in dcm[0x0008, 0x0008].value) and "T1" in dcm[0x0008, 0x0008].value:
                    pass
                elif type.upper() == "T2MAP" and "MOCO" in dcm[0x0008, 0x0008].value and "T2" in dcm[0x0008, 0x0008].value:
                    pass
                else:
                    continue

                os.makedirs(root.replace(path, path_out), exist_ok=True)
                shutil.copyfile(os.path.join(root, file), os.path.join(root.replace(path, path_out), UID + ".dcm"))

            except:
                pass

    return path_out

def extract_contours(path):
    path_out = os.path.dirname(path)
    path_out = os.path.join(path_out, path.replace(path_out, "")[1:] + "_contours")

    tool_cvi.parse_cvi42ws(path, path_out)

    return path_out

def match_dicom_contours(path_data, path_contours):
    path_out = os.path.dirname(path_data)
    path_out = os.path.join(path_out, path_data.replace(path_out, "")[1:] + "_match")

    data_list = []
    for root, _, files in os.walk(path_data):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file))
                UID = str(dcm[0x0008, 0x0018].value)
                name = ''.join(char for char in str(dcm[0x0010, 0x0010].value) if char.isalnum())

                data_list.append([os.path.join(root, file), UID, name])
            except:
                pass

    data_list = np.array(data_list)
    path_list = data_list[:,0].flatten()
    UID_list = data_list[:,1].flatten()
    name_list = data_list[:,2].flatten()

    for root, _, files in os.walk(path_contours):
        for file in files:
            if file.endswith("pickle") and file != "case.pickle":
                UID = file.replace(".pickle", "")
                index = np.argwhere(UID_list == UID)
                if len(index) > 0:
                    file_data = path_list[index[0]][0]


                    os.makedirs(os.path.join(path_out, name_list[index[0]][0]), exist_ok=True)
                    try:
                        shutil.copyfile(file_data, os.path.join(path_out, name_list[index[0]][0], UID_list[index[0]][0] + ".dcm"))
                    except:
                        pass
                    try:
                        shutil.copyfile(os.path.join(root, file), os.path.join(path_out, name_list[index[0]][0], file))
                    except:
                        pass

    return path_out

def split_train_test_validation(path, split_test=0.15, split_validation=0.10):
    studies = os.listdir(path)

    for study in studies:
        cases = os.listdir(os.path.join(path, study))

        num_test = int(split_test * len(cases))
        num_val = int(split_validation * len(cases))

        run = True
        while run:
            indeces_test = np.unique(np.random.randint(0, len(cases), (num_test,)))
            if len(indeces_test) == num_test:
                run=False

        run = True
        while run:
            indeces_val = np.unique(np.random.randint(0, len(cases), (num_val,)))
            if len(indeces_val) == num_val and not np.any(np.in1d(indeces_val, indeces_test)):
                run=False

        for i in range(len(cases)):
            if i in indeces_test:
                #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "TEST", study, cases[i]), dirs_exist_ok=True)
            elif i in indeces_val:
                #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "VALIDATION", study, cases[i]), dirs_exist_ok=True)
            else:
                #os.makedirs(os.path.join(path_out, "Expert Split Train", subsubdirs[i]), exist_ok=True)
                shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "TRAIN", study, cases[i]), dirs_exist_ok=True)

    return

def split_train_test_validation_equivalent(paths, split_test=0.15, split_validation=0.10):
    studies = os.listdir(paths[0])

    for s in range(len(studies)):
        cases = os.listdir(os.path.join(paths[0], studies[s]))

        num_test = int(split_test * len(cases))
        num_val = int(split_validation * len(cases))

        run = True
        while run:
            indeces_test = np.unique(np.random.randint(0, len(cases), (num_test,)))
            if len(indeces_test) == num_test:
                run=False

        run = True
        while run:
            indeces_val = np.unique(np.random.randint(0, len(cases), (num_val,)))
            if len(indeces_val) == num_val and not np.any(np.in1d(indeces_val, indeces_test)):
                run=False

        for i in range(len(cases)):
            for p in range(len(paths)):
                path = paths[p]
                study = os.listdir(path)[s]

                try:
                    if i in indeces_test:
                        #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                        shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "TEST", study, cases[i]), dirs_exist_ok=True)
                    elif i in indeces_val:
                        #os.makedirs(os.path.join(path_out, "Expert Split Test", subsubdirs[i]), exist_ok=True)
                        shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "VALIDATION", study, cases[i]), dirs_exist_ok=True)
                    else:
                        #os.makedirs(os.path.join(path_out, "Expert Split Train", subsubdirs[i]), exist_ok=True)
                        shutil.copytree(os.path.join(path, study, cases[i]), os.path.join(path + "_SPLIT", "TRAIN", study, cases[i]), dirs_exist_ok=True)
                except:
                    raise RuntimeError("STOP")

    return

if __name__ == "__main__":

    #paths = r"D:\ECRC_AG_CMR\3 - Promotion\Paper - CASEG\Data\DATA\FULL DATASETS\Expert FULL SAX"
    #for sub in os.listdir(paths):
    #    p = os.path.join(paths, sub)
    #    extract_sax(p, False)
    #    extract_midventricular(p, False)
    #    extract_midventricular(p, True)

    #paths = [r"D:\ECRC_AG_CMR\3 - Promotion\Paper - CASEG\Data\DATA\FULL DATASETS\EXPERT FULL\EXPERT FULL SAX", r"D:\ECRC_AG_CMR\3 - Promotion\Paper - CASEG\Data\DATA\FULL DATASETS\EXPERT FULL\EXPERT FULL MV native", r"D:\ECRC_AG_CMR\3 - Promotion\Paper - CASEG\Data\DATA\FULL DATASETS\EXPERT FULL\EXPERT FULL MV", r"D:\ECRC_AG_CMR\3 - Promotion\Paper - CASEG\Data\DATA\FULL DATASETS\EXPERT FULL\EXPERT FULL SAX native"]
    #split_train_test_validation_equivalent(paths)

    #extract_sax(r"F:\LeonoraSWVergleich\CARETOX", True)
    #extract_contours(r"F:\LeonoraSWVergleich\Intraobserver")
    #extract_contours(r"F:\LeonoraSWVergleich\MRT 1_T1-Map")
    #extract_data(r"F:\LeonoraSWVergleich\CARETOX_SAX_native_pca")
    #match_dicom_contours(r"F:\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP", r"F:\LeonoraSWVergleich\MRT 1_T1-Map_contours")
    extract_midventricular(r"F:\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_Intaobserver")
    extract_midventricular(r"F:\LeonoraSWVergleich\CARETOX_SAX_native_pca_T1MAP_match_MRT 1")

    a = 0

    # def get_annotation(self, **kwargs):
    #     options = mock.Mock()
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     result = []
    #
    #     for i in options.indices:
    #         pass
    #
    #     return result
    #
    # def get_input_bb(self, **kwargs):
    #     options = mock.Mock()
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     list_img = self.get_rgb("cv2")
    #     list_bounding_box = []
    #     list_object = []
    #
    #     for i in options.indices:
    #         single_dcm = self.list[i]
    #
    #         x1 = int(single_dcm[0x0033, 0x3304].value)
    #         y1 = int(single_dcm[0x0033, 0x3305].value)
    #         x2 = int(single_dcm[0x0033, 0x3304].value) + int(single_dcm[0x0033, 0x3306].value)
    #         y2 = int(single_dcm[0x0033, 0x3305].value) + int(single_dcm[0x0033, 0x3307].value)
    #
    #         list_bounding_box.append([[x1, y1, x2, y2]])
    #         list_object.append(["myocard"])
    #
    #     return list_img, list_bounding_box, list_object
    #
    # def get_input_lh(self, **kwargs):
    #     options = mock.Mock()
    #     options.num_bins = kwargs.get("num_bins", 100)
    #     options.min = kwargs.get("min", None)
    #     options.max = kwargs.get("max", None)
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     data_hist = np.zeros((len(options.indices), options.num_bins))
    #     data_edges = np.zeros((len(options.indices), options.num_bins + 1))
    #     data_bound = np.zeros((len(options.indices), 2))
    #     data_max = np.zeros(len(options.indices))
    #
    #     for i in options.indices:
    #         pixel_data = np.copy(self.list[i].pixel_array)
    #
    #         if options.min is None:
    #             range_min = pixel_data.min()
    #         else:
    #             range_min = options.min
    #
    #         if options.max is None:
    #             range_max = pixel_data.max()
    #         else:
    #             range_max = options.max
    #
    #         data_hist[i, :], data_edges[i, :], data_max[i] = __old_functions.histogram(pixel_data.reshape(-1), options.num_bins, (range_min, range_max))
    #
    #         try:
    #             data_bound[i, 0] = float(self.list[i][0x0033, 0x3301].value)
    #             data_bound[i, 1] = float(self.list[i][0x0033, 0x3302].value)
    #         except:
    #             pass
    #
    #     return data_hist, data_edges, data_max, data_bound
    #
    # def get_mask(self, **kwargs):
    #     options = mock.Mock()
    #     options.expand_lcc = kwargs.get("expand_lcc", False)
    #     options.expand_bb = kwargs.get("expand_bb", 1.0)
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     if len(self.bycatch.lh) != len(self.list):
    #         self.bycatch.lh = ps_lh.PSLH().predict(self.get_input_lh()[0])
    #
    #     if len(self.bycatch.rgb) != len(self.list):
    #         self.get_rgb("cv2")
    #
    #     if len(self.bycatch.bb) != len(self.list):
    #         self.bycatch.bb = bb_frcnn.FRCNN(True, configuration=configuration.path_frcnn_config, weights=configuration.path_frcnn_weights).predict(self.bycatch.rgb)
    #
    #     result = []
    #
    #     for i in options.indices:
    #         array = np.copy(self.list[i].pixel_array)
    #         array = array / np.max(array)
    #         result_intermediate = []
    #         for j in range(len(self.bycatch.bb[i])):
    #
    #             poi_raw = np.copy(array)
    #             poi_raw[~((poi_raw >= min(self.bycatch.lh[i, :])) & (poi_raw <= max(self.bycatch.lh[i, :])))] = 0
    #             poi_raw[((poi_raw >= min(self.bycatch.lh[i, :])) & (poi_raw <= max(self.bycatch.lh[i, :])))] = 1
    #
    #             mask_lcc = np.zeros(np.shape(array))
    #             mask_lcc[int(self.bycatch.bb[i][j][3]):int(self.bycatch.bb[i][j][5]), int(self.bycatch.bb[i][j][2]):int(self.bycatch.bb[i][j][4])] = poi_raw[int(self.bycatch.bb[i][j][3]):int(self.bycatch.bb[i][j][5]), int(self.bycatch.bb[i][j][2]):int(self.bycatch.bb[i][j][4])]
    #
    #             components, idxl = __old_functions.connected_components(mask_lcc)
    #
    #             if options.expand_lcc: # expand over bb border
    #                 components, idxl = __old_functions.connected_components(poi_raw, seed=np.array(components[idxl])[0, :])
    #
    #             mask_lcc = np.zeros(np.shape(array))
    #             mask_lcc[np.array(components[idxl])[:, 0], np.array(components[idxl])[:, 1]] = 1
    #
    #             result_intermediate.append(mask_lcc)
    #         result.append(result_intermediate)
    #
    #     self.bycatch.mask = result
    #
    #     return self.bycatch.mask
    #
    # def get_poi(self, **kwargs):
    #     options = mock.Mock()
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     if len(self.bycatch.mask) != len(self.list):
    #         self.get_mask()
    #
    #     if len(self.bycatch.rgb) != len(self.list):
    #         self.get_rgb("cv2")
    #
    #     if len(self.bycatch.bb) != len(self.list):
    #         self.bycatch.bb = bb_frcnn.FRCNN(True, configuration=configuration.path_frcnn_config, weights=configuration.path_frcnn_weights).predict(self.bycatch.rgb)
    #
    #     result_values = []
    #     result_indeces = []
    #
    #     for i in options.indices:
    #         array = self.list[i].pixel_array
    #         result_intermediate_values = []
    #         result_intermediate_indeces = []
    #
    #         for j in range(len(self.bycatch.bb[i])):
    #             # np.take(self.list[0].pixel_array, np.argwhere(self.bycatch.mask[0][0] == 1))
    #             values = (array * self.bycatch.mask[i][j]).flatten()
    #             indeces = np.argwhere(self.bycatch.mask[i][j] == 1)
    #             values = values[values > 0]
    #             result_intermediate_values.append(values)
    #             result_intermediate_indeces.append(indeces)
    #
    #         result_values.append(result_intermediate_values)
    #         result_indeces.append(result_intermediate_indeces)
    #
    #     self.bycatch.poi = [result_values, result_indeces]
    #
    #     return result_values, result_indeces
    #
    # def get_distribution(self, **kwargs):
    #     options = mock.Mock()
    #     options.min = kwargs.get("min", None)
    #     options.max = kwargs.get("max", None)
    #     options.num_bins = kwargs.get("num_bins", 100)
    #     options.indices = kwargs.get("indices", range(len(self.list)))
    #
    #     if len(self.bycatch.poi) != len(self.list):
    #         self.get_poi()
    #
    #     data_hist = np.zeros((len(options.indices), options.num_bins))
    #     data_edges = np.zeros((len(options.indices), options.num_bins + 1))
    #     data_bound = np.zeros((len(options.indices), 2))
    #     data_max = np.zeros(len(options.indices))
    #
    #     for i in options.indices:
    #         pixel_data = np.copy(self.list[i].pixel_array)
    #
    #         for j in range(len(self.bycatch.poi[i])):
    #
    #             if options.min is None:
    #                 range_min = pixel_data.min()
    #             else:
    #                 range_min = options.min
    #
    #             if options.max is None:
    #                 range_max = pixel_data.max()
    #             else:
    #                 range_max = options.max
    #
    #             data_hist[i, :], data_edges[i, :], data_max[i] = __old_functions.histogram(self.bycatch.poi[i][j][0].reshape(-1), options.num_bins, (range_min, range_max))
    #
    #             try:
    #                 data_bound[i, 0] = float(self.list[i][0x0033, 0x3301].value)
    #                 data_bound[i, 1] = float(self.list[i][0x0033, 0x3302].value)
    #             except:
    #                 pass
    #
    #     self.bycatch.distribution = [data_hist, data_edges, data_max, data_bound]
    #
    #     return self.bycatch.distribution #data_hist, data_edges, data_max, data_bound
    #
    # def get_register(self, **kwargs):
    #     options = mock.Mock()
    #     options.bin_type = kwargs.get("bin_type", "avgstd")
    #     options.bin_num = kwargs.get("bin_num", 3)
    #     options.bin_size = kwargs.get("bin_size", 0.1)
    #
    #     values, indeces = self.get_poi(**kwargs)
    #     self.bycatch.register = register.Register()
    #     self.bycatch.register.load(values, indeces, bin_type=options.bin_type, bin_num=options.bin_num, bin_size=options.bin_size)
    #
    #     return self.bycatch.register
