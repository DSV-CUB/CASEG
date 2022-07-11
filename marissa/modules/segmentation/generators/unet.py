from __future__ import absolute_import
import numpy as np
import copy
import os
import pickle

from marissa.modules.segmentation import configurations

from marissa.toolbox.tools import tool_general, tool_pydicom, tool_hadler
from marissa.modules.datahandler import dicom
from marissa.toolbox.creators import creator_generator


class Setup(creator_generator.Inheritance):

    def __init__(self, input, configuration=None, generator_type="TRAINING", **kwargs):
        super().__init__()

        if configuration is None or configuration.model_settings.model_type.upper() != "UNET":
            self.configuration = configurations.Setup("UNET", **kwargs)
        else:
            self.configuration = configuration
            self.configuration.set(**kwargs)

        self.generator_type = generator_type.upper()

        model_bb = kwargs.get("model_bb", None)
        ws_dir = kwargs.get("ws_dir", False)
        self.mask_mode = kwargs.get("mask_mode", "RASTERIZE")

        self.x = []
        self.x_bb = []
        self.y = []
        self.y_contour = []
        self.information = []
        self.pixel_spacing = []
        print("Load Data")

        if type(input) == type(self): #copy data
            self.x = copy.deepcopy(input.x)
            self.x_bb = copy.deepcopy(input.x_bb)
            self.y = copy.deepcopy(input.y)
            self.information = copy.deepcopy(input.information)
        elif ws_dir is None:
            if isinstance(input, str):
                dhdcm = dicom.Setup(path=input, measure=self.configuration.measure, tissue=self.configuration.tissue, view=self.configuration.view, examination=self.configuration.examination, filter="ALL")
            else:
                dhdcm = copy.deepcopy(input)

            for i in range(len(dhdcm.path)):
                try:
                    annotations = tool_pydicom.get_marissa_annotation(dhdcm.path[i])
                    try:
                        roi_id = annotations.roi_tissue.index(self.configuration.tissue.upper())
                    except ValueError:
                        roi_id = annotations.roi_tissue.index(self.configuration.tissue.lower())
                except:
                    continue

                in_0 = dhdcm.get_pixel_data(indices=i, rescale=self.configuration.data_rescale, representation=self.configuration.data_representation)

                out_0 = tool_pydicom.get_marissa_anotation_map(in_0, annotations.bb, annotations.rmv, annotations.lh, roi_id, num_bb=None, stage=3)
                out_0 = np.logical_or(out_0, tool_pydicom.get_segmentation_holes(out_0, exclude=2)).astype("int")

                if model_bb is not None:
                    # bb predict
                    pred_bb = model_bb.predict(in_0 / np.max(in_0))
                    self.x_bb.append(pred_bb)

                self.x.append(in_0)
                self.y.append(out_0)
                self.information.append([dhdcm.path[i], str(dhdcm.tags[i]["tag" + str(int("0x00100010", 0))]) + "__" + str(dhdcm.tags[i]["tag" + str(int("0x00080018", 0))])])
                self.pixel_spacing.append(np.array(dhdcm.tags[0]["tag" + str(int("0x00280030", 0))]))

                print(str(i + 1) + " of " + str(len(dhdcm.path)) + " data loaded in the generator.", end="\r")
        else:
            if not ws_dir:
                dhdcm = dicom.Setup(path=input, measure=self.configuration.measure, tissue=self.configuration.tissue, view=self.configuration.view, examination=self.configuration.examination, filter="ALL")
                for j in range(len(dhdcm.path)):
                    in_0 = dhdcm.get_pixel_data(j, rescale=True, representation=False)
                    out_0 = np.zeros(np.shape(in_0))
                    contours = []
                    if model_bb is not None:
                        # bb predict
                        pred_bb = model_bb.predict(in_0 / np.max(in_0))
                        self.x_bb.append(pred_bb)

                    self.x.append(in_0)
                    self.y.append(out_0)
                    self.y_contour.append([[[0,0],[0,1],[1,1],[1,0]], [[0,0],[0,1],[1,1],[1,0]]]) # dummy
                    self.information.append([dhdcm.path[j], str(dhdcm.tags[j]["tag" + str(int("0x00100010", 0))]) + "__" + str(dhdcm.tags[j]["tag" + str(int("0x00080018", 0))])])
                    self.pixel_spacing.append(np.array(dhdcm.tags[j]["tag" + str(int("0x00280030", 0))]))
                    print(str(len(self.x)) + " data loaded in the generator.", end="\r")

            else:
                path_dcm = []
                path_ws = []

                if isinstance(input, str) and input == ws_dir:
                    for root, _, files in os.walk(ws_dir):
                        for file in files:
                            if file.endswith("pickle") and file != "case.pickle":
                                path_ws.append(os.path.join(root, file))
                                path_dcm.append(os.path.join(root, file.replace(".pickle", ".dcm")))
                elif isinstance(input, str):
                    dhdcm = dicom.Setup(path=input, measure=self.configuration.measure, tissue=self.configuration.tissue, view=self.configuration.view, examination=self.configuration.examination, filter="ALL")
                    for root, _, files in os.walk(ws_dir):
                        for file in files:
                            if file.endswith("pickle") and file != "case.pickle":
                                UID = file.replace(".pickle", "")
                                with open(os.path.join(root, file), "rb") as ws_file:
                                    ws = pickle.load(ws_file)
                                    ws_file.close()
                                if "saendocardialContour" in ws and "saepicardialContour" in ws:
                                    for j in range(len(dhdcm.path)):
                                        if str(dhdcm.tags[j]["tag" + str(int(0x00080018))]) == UID:
                                            path_ws.append(os.path.join(root, file))
                                            path_dcm.append(dhdcm.path[j])
                else:
                    raise ValueError("input must be a path if ws_dir is given")

                for j in range(len(path_dcm)):
                    dhdcm = dicom.Setup(path=path_dcm[j], measure=self.configuration.measure, tissue=self.configuration.tissue, view=self.configuration.view, examination=self.configuration.examination, filter="ALL")
                    try:
                        with open(path_ws[j], "rb") as ws_file:
                            ws = pickle.load(ws_file)
                            ws_file.close()

                            spr = 1
                            in_0 = dhdcm.get_pixel_data(0, rescale=True, representation=False)
                            #in_0 = tool_general.array_resolution(in_0, spr)
                            contours = []

                            if "saendocardialContour" in ws:
                                contours.append(spr * ws["saendocardialContour"][0][:, [1,0]])

                            if "saepicardialContour" in ws:
                                contours.append(spr * ws["saepicardialContour"][0][:, [1,0]])

                            if not "saendocardialContour" in ws and not "saepicardialContour" in ws:
                                for key in ws:
                                    if "contour" in key.lower():
                                        contours.append(spr * ws[key][0][:, [1,0]])


                            out_0 = tool_general.contour2mask(contours, in_0, cubic_spline=False, mode=self.mask_mode)
                            #out_0 = tool_general.contour2mask([spr * ws["saendocardialContour"][0][:, [1,0]], spr * ws["saepicardialContour"][0][:, [1,0]]], in_0, cubic_spline=False, mode=self.mask_mode)
                            #out_0 = np.logical_or(out_0, tool_pydicom.get_segmentation_holes(out_0, exclude=2)).astype("int")

                            if model_bb is not None:
                                # bb predict
                                pred_bb = model_bb.predict(in_0 / np.max(in_0))
                                self.x_bb.append(pred_bb)

                                #from marissa.toolbox.tools import tool_plot
                                #indeces = np.argwhere(out_0)
                                #index_x_min = np.min(indeces[:,0])
                                #index_x_max = np.max(indeces[:,0])
                                #index_y_min = np.min(indeces[:,1])
                                #index_y_max = np.max(indeces[:,1])
                                #true_bb = np.zeros(np.shape(out_0))
                                #true_bb[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                                #_, masks = tool_plot.masks2delta2rgba(true_bb, pred_bb)
                                #tool_plot.plot_masks((in_0/np.max(in_0)), [masks], ["TEST"], export=True)

                            self.x.append(in_0)
                            self.y.append(out_0)
                            self.y_contour.append(contours)
                            self.information.append([path_dcm[j], str(dhdcm.tags[0]["tag" + str(int("0x00100010", 0))]) + "__" + str(dhdcm.tags[0]["tag" + str(int("0x00080018", 0))])])
                            self.pixel_spacing.append(np.array(dhdcm.tags[0]["tag" + str(int("0x00280030", 0))]))

                            print(str(len(self.x)) + " data loaded in the generator.", end="\r")
                    except:
                        print("Problem with loading workspace: " + path_ws[j])

        self.indeces = np.arange(len(self.x))
        np.random.shuffle(self.indeces)
        self.gen = self.generator()
        return

    def get_data(self, i):
        if self.configuration.data_segmentation_level.upper() == "CONTOUR":
            # DATA PREPARATION AND AUGMENTATION
            contour_augmentation = copy.deepcopy(self.y_contour[i])
            if not self.x_bb == []:
                indeces = np.argwhere(np.squeeze(self.x_bb[i]) > 0.5)
                if len(indeces) == 0:
                    contour_augmentation.append(np.array([[-1, -1]]))
                else:
                    contour_augmentation.append(np.array([[np.min(indeces[:,0]), np.min(indeces[:,1])], [np.min(indeces[:,0]), np.max(indeces[:,1])], [np.max(indeces[:,0]), np.max(indeces[:,1])], [np.max(indeces[:,0]), np.min(indeces[:,1])]]))

            if self.generator_type == "TRAINING":
                if not self.configuration.model_settings.model_crop:
                    variation, variation_contours = tool_general.image_augmentation(self.x[i], contour=contour_augmentation)
                else:
                    variation, variation_contours = tool_general.image_augmentation(self.x[i], contour=contour_augmentation, exclude="crop")
            else:
                variation, variation_contours = self.x[i], contour_augmentation

            gen_x = variation / np.max(variation)
            if not self.x_bb == []:
                gen_y = variation_contours[:-1]
                gen_bb = variation_contours[-1]
            else:
                gen_y = variation_contours
                gen_bb = None

            # GENERATOR USE CASES
            if self.configuration.model_settings.model_predictor == "SEGM": # predict segmentation
                if gen_bb is None or (not self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 1): # no bounding box -> refU
                    x = tool_general.array_resize(gen_x, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    for i in range(len(gen_y)):
                        gen_y[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(gen_x))), 1, gen_y[i])
                    y = tool_general.contour2mask(gen_y, x, mode=self.mask_mode)

                elif self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 1: # crop image to bb, but no bb input -> cropU
                    prob = np.random.randint(0, 100)
                    if (prob < 5 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # 5 percent no crop
                        x_array = gen_x
                        y_array = gen_y
                    elif (prob < 10 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # 5 percent perfect crop
                        edges = np.array(np.vstack((gen_y[0], gen_y[1]))).astype("float32")
                        index_x_min = int(np.min(edges[:,0]))
                        index_x_max = int(np.max(edges[:,0]))
                        index_y_min = int(np.min(edges[:,1]))
                        index_y_max = int(np.max(edges[:,1]))

                        x_diff = index_x_max - index_x_min
                        y_diff = index_y_max - index_y_min

                        extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                        index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                        index_x_max = min(np.shape(gen_x)[0], index_x_max + int(x_diff * extend_factor))
                        index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                        index_y_max = min(np.shape(gen_x)[1], index_y_max + int(y_diff * extend_factor))

                        x_array = gen_x[index_x_min:index_x_max, index_y_min:index_y_max]

                        gen_y_n = copy.deepcopy(gen_y)
                        for i in range(len(gen_y_n)):
                            gen_y_n[i] = np.apply_along_axis(lambda lmbd: lmbd - np.array([index_x_min, index_y_min]), 1, gen_y_n[i])
                        y_array = gen_y_n
                    else:
                        try:
                            prob = np.random.randint(0, 100)
                            if (prob < 50 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING":
                                kick = np.random.randint(0, 11, 4) - 5
                            else:
                                kick = np.squeeze(np.zeros((4,1)))

                            edges = np.array(gen_bb).astype("float32")
                            index_x_min = int(np.min(edges[:,0]) + kick[0])
                            index_x_max = int(np.max(edges[:,0]) + kick[1])
                            index_y_min = int(np.min(edges[:,1]) + kick[2])
                            index_y_max = int(np.max(edges[:,1]) + kick[3])

                            x_diff = index_x_max - index_x_min
                            y_diff = index_y_max - index_y_min

                            extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                            index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                            index_x_max = min(np.shape(gen_x)[0], index_x_max + int(x_diff * extend_factor))
                            index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                            index_y_max = min(np.shape(gen_x)[1], index_y_max + int(y_diff * extend_factor))

                            x_array = gen_x[index_x_min:index_x_max, index_y_min:index_y_max]

                            gen_y_n = copy.deepcopy(gen_y)
                            for i in range(len(gen_y_n)):
                                gen_y_n[i] = np.apply_along_axis(lambda lmbd: lmbd - np.array([index_x_min, index_y_min]), 1, gen_y_n[i])
                            y_array = gen_y_n
                        except:
                            x_array = gen_x
                            y_array = gen_y

                    try:
                        x = tool_general.array_resize(x_array, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                        for i in range(len(y_array)):
                            y_array[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(x_array))), 1, y_array[i])
                    except:
                        x_array = gen_x
                        y_array = gen_y

                        x = tool_general.array_resize(x_array, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                        for i in range(len(y_array)):
                            y_array[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(x_array))), 1, y_array[i])

                    y = tool_general.contour2mask(y_array, x, mode=self.mask_mode)

                elif self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 2: # crop image to bb and give bb as input -> crinU
                    prob = np.random.randint(0, 100)
                    if (prob < 5 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # 5 percent no crop
                        x_array = gen_x
                        y_array = gen_y
                        bb_array = gen_bb
                    elif (prob < 10 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # 5 percent perfect crop
                        edges = np.array(np.vstack((gen_y[0], gen_y[1]))).astype("float32")
                        index_x_min = int(np.min(edges[:,0]))
                        index_x_max = int(np.max(edges[:,0]))
                        index_y_min = int(np.min(edges[:,1]))
                        index_y_max = int(np.max(edges[:,1]))

                        gen_bb = np.array([[index_x_min, index_y_min], [index_x_min, index_y_max], [index_x_max, index_y_max], [index_x_max, index_y_min]])

                        x_diff = index_x_max - index_x_min
                        y_diff = index_y_max - index_y_min

                        extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                        crop_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                        crop_x_max = min(np.shape(gen_x)[0], index_x_max + int(x_diff * extend_factor))
                        crop_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                        crop_y_max = min(np.shape(gen_x)[1], index_y_max + int(y_diff * extend_factor))

                        x_array = gen_x[crop_x_min:crop_x_max, crop_y_min:crop_y_max]

                        gen_y_n = copy.deepcopy(gen_y)
                        for i in range(len(gen_y_n)):
                            gen_y_n[i] = np.apply_along_axis(lambda lmbd: lmbd - np.array([crop_x_min, crop_y_min]), 1, gen_y_n[i])
                        y_array = gen_y_n

                        gen_bb = np.apply_along_axis(lambda lmbd: lmbd - np.array([crop_x_min, crop_y_min]), 1, gen_bb)
                        bb_array = gen_bb
                    else:
                        try:
                            prob = np.random.randint(0, 100)
                            if (prob < 50 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING":
                                kick = np.random.randint(0, 11, 4) - 5
                            else:
                                kick = np.squeeze(np.zeros((4,1)))

                            edges = np.array(gen_bb).astype("float32")
                            index_x_min = int(np.min(edges[:,0]) + kick[0])
                            index_x_max = int(np.max(edges[:,0]) + kick[1])
                            index_y_min = int(np.min(edges[:,1]) + kick[2])
                            index_y_max = int(np.max(edges[:,1]) + kick[3])
                            
                            x_diff = index_x_max - index_x_min
                            y_diff = index_y_max - index_y_min


                            extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                            crop_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                            crop_x_max = min(np.shape(gen_x)[0], index_x_max + int(x_diff * extend_factor))
                            crop_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                            crop_y_max = min(np.shape(gen_x)[1], index_y_max + int(y_diff * extend_factor))

                            x_array = gen_x[crop_x_min:crop_x_max, crop_y_min:crop_y_max]

                            gen_y_n = copy.deepcopy(gen_y)
                            for i in range(len(gen_y_n)):
                                gen_y_n[i] = np.apply_along_axis(lambda lmbd: lmbd - np.array([crop_x_min, crop_y_min]), 1, gen_y_n[i])
                            y_array = gen_y_n

                            gen_bb_n = np.apply_along_axis(lambda lmbd: lmbd - np.array([crop_x_min, crop_y_min]), 1, gen_bb)
                            bb_array = gen_bb_n
                        except:
                            x_array = gen_x
                            y_array = gen_y
                            bb_array = gen_bb

                    try:
                        x = tool_general.array_resize(x_array, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                        for i in range(len(y_array)):
                            y_array[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(x_array))), 1, y_array[i])
                    except:
                        x_array = gen_x
                        y_array = gen_y
                        bb_array = gen_bb
                        x = tool_general.array_resize(x_array, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                        for i in range(len(y_array)):
                            y_array[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(x_array))), 1, y_array[i])

                    y = tool_general.contour2mask(y_array, x, mode=self.mask_mode)

                    prob = np.random.randint(0, 100)
                    if (prob < 5 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING":
                        bb = np.zeros(np.shape(x))
                    else:
                        try:
                            bb_array = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(x_array))), 1, bb_array)
                            bb = np.zeros(np.shape(x))
                            bb[int(np.min(bb_array[:,0])):int(np.max(bb_array[:,0])), int(np.min(bb_array[:,1])):int(np.max(bb_array[:,1]))] = 1
                        except:
                            bb = np.zeros(np.shape(x))

                    x = np.dstack((x, bb))

                else: # no crop but bb input -> bbU

                    x = tool_general.array_resize(gen_x, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    prob = np.random.randint(0, 100)
                    if (prob < 5 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # empty BB
                        bb = np.zeros(np.shape(x))
                    elif (prob < 10 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING": # perfect BB
                        for i in range(len(gen_y)):
                            gen_y[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(gen_x))), 1, gen_y[i])
                        edges = np.array(np.vstack((gen_y[0], gen_y[1]))).astype("float32")
                        index_x_min = int(np.min(edges[:,0]))
                        index_x_max = int(np.max(edges[:,0]))
                        index_y_min = int(np.min(edges[:,1]))
                        index_y_max = int(np.max(edges[:,1]))
                        bb = np.zeros(np.shape(x))
                        bb[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                    else:
                        bb_array = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(gen_x))), 1, gen_bb)

                        prob = np.random.randint(0, 100)
                        if (prob < 50 or len(gen_bb) == 1) and self.generator_type.upper()=="TRAINING":
                            kick = np.random.randint(0, 11, 4) - 5
                        else:
                            kick = np.squeeze(np.zeros((4,1)))
                        
                        bb = np.zeros(np.shape(x))
                        bb[int(np.min(bb_array[:,0]) + kick[0]):int(np.max(bb_array[:,0]) + kick[1]), int(np.min(bb_array[:,1]) + kick[2]):int(np.max(bb_array[:,1]) + kick[3])] = 1

                    x = np.dstack((x, bb))

                    for i in range(len(gen_y)):
                        gen_y[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(gen_x))), 1, gen_y[i])

                    y = tool_general.contour2mask(gen_y, np.squeeze(x[:,:,0]), mode=self.mask_mode)

            elif self.configuration.model_settings.model_predictor == "BB": # predict BB
                x = tool_general.array_resize(gen_x, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                for i in range(len(gen_y)):
                    gen_y[i] = np.apply_along_axis(lambda lmbd: lmbd * np.array([self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]] / np.array(np.shape(gen_x))), 1, gen_y[i])

                try:
                    edges = np.array(np.vstack((gen_y[0], gen_y[1]))).astype("float32")
                    index_x_min = int(np.min(edges[:,0])) #+ np.random.randint(0, 10) - 5
                    index_x_max = int(np.max(edges[:,0])) #+ np.random.randint(0, 10) - 5
                    index_y_min = int(np.min(edges[:,1])) #+ np.random.randint(0, 10) - 5
                    index_y_max = int(np.max(edges[:,1])) #+ np.random.randint(0, 10) - 5

                    y = np.zeros(np.shape(x))
                    y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                except:
                    y = np.zeros(np.shape(y))
            else:
                raise ValueError("configuration.model_settings.model.predictor is \"" + str(self.configuration.model_settings.model_predictor) + "\" but only \"SEGM\" and \"BB\" are allowed.")

            if self.configuration.data_plot:
                from marissa.toolbox.tools import tool_plot
                # original picture and contour
                tool_plot.plot_contour(self.x[i], self.y_contour[i], export=self.configuration.data_plot)
                if len(np.shape(x)) == 2:
                    tool_plot.plot_contour(x, gen_y, export=self.configuration.data_plot)
                    _, mask = tool_plot.masks2delta2rgba(y,y)
                    tool_plot.plot_masks(x, [mask], ["Test"], export=self.configuration.data_plot)
                else:
                    tool_plot.plot_contour(x[:,:,0], gen_y, export=self.configuration.data_plot)
                    _, mask = tool_plot.masks2delta2rgba(bb,y)
                    tool_plot.plot_masks(x[:,:,0], [mask], ["Test"], export=self.configuration.data_plot)

        else: # SEGMENTATION LEVEL: PIXELMAP
            if not self.x_bb == []:
                variation_stack = np.dstack((self.x[i], self.x_bb[i], self.y[i]))
            else:
                variation_stack = np.dstack((self.x[i], self.y[i]))

            if self.generator_type == "TRAINING":
                if not self.configuration.model_settings.model_crop:
                    variation, _ = tool_general.image_augmentation(variation_stack)
                else:
                    variation, _ = tool_general.image_augmentation(variation_stack, exclude="crop")
            else:
                variation, _ = variation_stack

            variation[:,:,0] = variation[:,:,0] / np.max(variation[:,:,0])

            # GENERATOR MODE
            if self.configuration.model_settings.model_predictor == "SEGM":
                if self.x_bb == [] or (not self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 1): # refU
                    x = tool_general.array_resize(variation[:,:,0], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                    y = tool_general.array_resize(variation[:,:,1], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_hadler.getLargestCC(y)
                elif self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 1: # cropU
                    prob = np.random.randint(0, 100)
                    if prob < 5:
                        x_array = variation[:,:, 0]
                        y_arary = variation[:,:, 2]
                    else:
                        try:
                            indeces = np.argwhere(np.squeeze(variation[:,:,1]) > 0.5)
                            index_x_min = np.min(indeces[:,0])
                            index_x_max = np.max(indeces[:,0])
                            index_y_min = np.min(indeces[:,1])
                            index_y_max = np.max(indeces[:,1])

                            x_diff = index_x_max - index_x_min
                            y_diff = index_y_max - index_y_min

                            extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                            index_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                            index_x_max = min(np.shape(variation[:,:,1])[0], index_x_max + int(x_diff * extend_factor))
                            index_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                            index_y_max = min(np.shape(variation[:,:,1])[1], index_y_max + int(y_diff * extend_factor))

                            x_array = variation[index_x_min:index_x_max, index_y_min:index_y_max, 0]
                            y_arary = variation[index_x_min:index_x_max, index_y_min:index_y_max, 2]

                        except:
                            x_array = variation[:,:, 0]
                            y_arary = variation[:,:, 2]

                    x = tool_general.array_resize(x_array, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                    y = tool_general.array_resize(y_arary, (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_hadler.getLargestCC(y)

                elif self.configuration.model_settings.model_crop and self.configuration.model_settings.model_input_size[2] == 2: # crobbU
                    try:
                        indeces = np.argwhere(np.squeeze(variation[:,:,1]) > 0.5)
                        index_x_min = np.min(indeces[:,0])
                        index_x_max = np.max(indeces[:,0])
                        index_y_min = np.min(indeces[:,1])
                        index_y_max = np.max(indeces[:,1])
                        mask = np.zeros(np.shape(variation[:,:,1]))
                        mask[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                        variation[:,:,1] = mask

                        x_diff = index_x_max - index_x_min
                        y_diff = index_y_max - index_y_min

                        extend_factor = (float(self.configuration.model_settings.model_crop) - 1) / 2

                        crop_x_min = max(0, index_x_min - int(x_diff * extend_factor))
                        crop_x_max = min(np.shape(variation[:,:,1])[0], index_x_max + int(x_diff * extend_factor))
                        crop_y_min = max(0, index_y_min - int(y_diff * extend_factor))
                        crop_y_max = min(np.shape(variation[:,:,1])[1], index_y_max + int(y_diff * extend_factor))

                        prob = np.random.randint(0, 100)
                        if prob >= 5:
                            variation = variation[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]

                    except:
                        variation[:,:,1] = np.zeros(np.shape(variation[:,:,1]))

                    prob = np.random.randint(0, 100)
                    if prob < 5:
                        variation[:,:,1] = np.zeros(np.shape(variation[:,:,1]))

                    x = tool_general.array_resize(variation[:,:,:2], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                    y = tool_general.array_resize(variation[:,:,2], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_hadler.getLargestCC(y)

                else: # bbU
                    try:
                        indeces = np.argwhere(np.squeeze(variation[:,:,1]) > 0.5)
                        index_x_min = np.min(indeces[:,0])
                        index_x_max = np.max(indeces[:,0])
                        index_y_min = np.min(indeces[:,1])
                        index_y_max = np.max(indeces[:,1])
                        mask = np.zeros(np.shape(variation[:,:,1]))
                        mask[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                        variation[:,:,1] = mask
                    except:
                        variation[:,:,1] = np.zeros(np.shape(variation[:,:,1]))

                    prob = np.random.randint(0, 100)
                    if prob < 5:
                        variation[:,:,1] = np.zeros(np.shape(variation[:,:,1]))

                    x = tool_general.array_resize(variation[:,:,:2], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                    y = tool_general.array_resize(variation[:,:,2], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    y = tool_hadler.getLargestCC(y)
            elif self.configuration.model_settings.model_predictor == "BB": # BB
                x = tool_general.array_resize(variation[:,:,0], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)
                y = tool_general.array_resize(variation[:,:,1], (self.configuration.model_settings.model_input_size[0], self.configuration.model_settings.model_input_size[1]), anti_aliasing=True, normalize=True)

                try:
                    y[y >= 0.5] = 1
                    y[y < 1] = 0
                    indeces = np.argwhere(y)
                    index_x_min = np.min(indeces[:,0])
                    index_x_max = np.max(indeces[:,0])
                    index_y_min = np.min(indeces[:,1])
                    index_y_max = np.max(indeces[:,1])
                    y = np.zeros(np.shape(y))
                    y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                except:
                    y = np.zeros(np.shape(y))
            else:
                raise ValueError("configuration.model_settings.model.predictor is \"" + str(self.configuration.model_settings.model_predictor) + "\" but only \"SEGM\" and \"BB\" are allowed.")


        return x.astype("float32"), y.astype("float32")

    def y_to_bb(self, index = None):
        if index is None:
            y_bb = []
            for i in range(len(self.y)):
                indeces = np.argwhere(self.y[i])
                index_x_min = np.min(indeces[:,0])
                index_x_max = np.max(indeces[:,0])
                index_y_min = np.min(indeces[:,1])
                index_y_max = np.max(indeces[:,1])

                y = np.zeros(np.shape(self.y[i]))
                y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                y_bb.append(y)
        elif type(index) == list:
            y_bb = []
            for i in index:
                try:
                    indeces = np.argwhere(self.y[i])
                    index_x_min = np.min(indeces[:,0])
                    index_x_max = np.max(indeces[:,0])
                    index_y_min = np.min(indeces[:,1])
                    index_y_max = np.max(indeces[:,1])

                    y = np.zeros(np.shape(self.y[i]))
                    y[index_x_min:index_x_max, index_y_min:index_y_max] = 1
                    y_bb.append(y)
                except IndexError:
                    y_bb.append(-1)
        else:
            try:
                indeces = np.argwhere(self.y[index])
                index_x_min = np.min(indeces[:,0])
                index_x_max = np.max(indeces[:,0])
                index_y_min = np.min(indeces[:,1])
                index_y_max = np.max(indeces[:,1])

                y_bb = np.zeros(np.shape(self.y[index]))
                y_bb[index_x_min:index_x_max, index_y_min:index_y_max] = 1
            except IndexError:
                y_bb = -1

        return y_bb

