import pydicom
from pydicom.datadict import dictionary_VR
import numpy as np
import datetime
import mock
from marissa.toolbox.tools import tool_general


def get_size(dcm):
    '''
    :param dcm: a dicom object
    :return: tuple of height and width of the dicom image
    '''
    return np.shape(dcm.pixel_array)


def get_VR(address):
    '''

    :param address: dicom tag address
    :return: returns the value representation of teh tag
    '''

    return dictionary_VR(address)


def get_pixel_data(pixel_array, tags, **kwargs):
    """
    :param pixel_array:
    :param tags:
    :param kwargs:
    :return:
    """

    options = mock.Mock()
    options.representation = kwargs.get("representation", False)
    options.rescale = kwargs.get("rescale", True)
    options.min = kwargs.get("min", 0)
    options.max = kwargs.get("max", 255)

    pixel_data = np.copy(pixel_array)

    try:
        if options.rescale and not tags["tag" + str(int("00281052", 16))] is None and not tags["tag" + str(int("00281053", 16))] is None:
            pixel_data = pixel_data * float(tags["tag" + str(int("00281053", 16))]) + float(tags["tag" + str(int("00281052", 16))])
    except:
        pass

    try:
        if options.representation and not tags["tag" + str(int("00281050", 16))] is None and not tags["tag" + str(int("00281051", 16))] is None:
            c = float(tags["tag" + str(int("00281050", 16))]) # window center
            w = float(tags["tag" + str(int("00281051", 16))]) # window width
            search_if = pixel_data <= (c-0.5)-((w-1)/2)
            search_elif = pixel_data > (c-0.5)+((w-1)/2)

            pixel_data = ((pixel_data-(c-0.5)) / (w-1) + 0.5) * (options.max - options.min) + options.min
            pixel_data[search_if] = options.min
            pixel_data[search_elif] = options.max
    except:
        pass

    return pixel_data


def transform_ics_rcs(dcm, arr_points=None):
    '''
    ics = image coordinate system (tupel of x-, y-values in units of pixel in dicom pixel data x = column, y = row)
    rcs = reference coordinate system (standard to store points in dicom tags)
    Function to do forward (ics -> rcs) or backward (rcs -> ics) transformation in dependence of arr_points
    :param dcm: object - pydicom object
    :param arr_points:  forward: array - nx2 array with x and y values (units of pixels) of the points in the dicom image plane for conversion points to / backward: None - converts all ROI Contour Data to ics data / backward: list of lists - [[roi_num], [contour_num]] -> both are specified (only these specific contours are read if existing), both are None (same behaviour as if arr_points = None), roi_num = None and contour_num is specific list (the specified contours are read from all ROIs), roi_num is specific list and contour_num = None (from the specified ROIs all contours are read)
    :return: list of lists - 1 x n x 3 in forward case, m x n x 2 in backward case
    '''
    # conversion as described in https://dicom.innolitics.com/ciods/rt-dose/roi-contour/30060039/30060040/30060050
    # modified for inversion purpose: P = Mx + S / x = M^-1(P-S)

    result = []
    matrix = np.zeros((3, 3))

    # read in relevant dicom tags
    image_position = np.asarray(dcm[0x0020, 0x0032].value, np.double)
    direction_cosine = np.asarray(dcm[0x0020, 0x0037].value, np.double)
    pixel_spacing = np.asarray(dcm[0x0028, 0x0030].value, np.double)

    # create the matrix
    matrix[0, 0] = direction_cosine[0]
    matrix[1, 0] = direction_cosine[1]
    matrix[2, 0] = direction_cosine[2]

    matrix[0, 1] = direction_cosine[3]
    matrix[1, 1] = direction_cosine[4]
    matrix[2, 1] = direction_cosine[5]

    matrix[0:3, 2] = np.cross(direction_cosine[0:3], direction_cosine[3:]) / np.linalg.norm(
        np.cross(direction_cosine[0:3], direction_cosine[3:]))

    if arr_points is None:  # backward (rcs -> ics) for all
        # x=M^-1(P-S)
        for i in range(0, len(dcm[0x3006, 0x0039].value)):
            for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                vector = np.zeros((int(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0046].value), 3))
                vector[:, 0:3] = np.subtract(np.reshape(
                    np.asarray(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0050].value, np.double),
                    (-1, 3)), image_position.reshape((1, 3)))

                product = np.transpose(np.dot(np.linalg.inv(matrix[0:3, 0:3]), np.transpose(vector)))
                product = product[:, 0:2]
                product[:, 0] = product[:, 0] / pixel_spacing[0]
                product[:, 1] = product[:, 1] / pixel_spacing[1]
                # currently column 0 = row and column 1 = column but we expect column 0 = x = column therefore switch column
                product[:, [0, 1]] = product[:, [1, 0]]

                result.append(product)
    elif type(arr_points) is list and len(arr_points) == 2 and type(arr_points[0]) is list and type(arr_points[1]) is list: # backward (rcs -> ics) for specific
        # x=M^-1(P-S)
        for i in range(0, len(dcm[0x3006, 0x0039].value)):
            if arr_points[0][0] is None or dcm[0x3006, 0x0039][i][0x3006, 0x0084].value in arr_points[0]:
                for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                    if arr_points[1][0] is None or dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value in arr_points[1]:
                        vector = np.zeros((int(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0046].value), 3))
                        vector[:, 0:3] = np.subtract(np.reshape(
                            np.asarray(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0050].value, np.double),
                            (-1, 3)), image_position.reshape((1, 3)))

                        product = np.transpose(np.dot(np.linalg.inv(matrix[0:3, 0:3]), np.transpose(vector)))
                        product = product[:, 0:2]
                        product[:, 0] = product[:, 0] / pixel_spacing[0]
                        product[:, 1] = product[:, 1] / pixel_spacing[1]
                        # currently column 0 = row and column 1 = column but we expect column 0 = x = column therefore switch column
                        product[:, [0, 1]] = product[:, [1, 0]]

                        result.append(product)
    else: # forward (ics -> rcs)
        # P = Mx+S
        # the vector x consists of tuples (x, y) where x =Columns and y= Rows
        # due to cubicspline x and y can have floating numbers
        vector = np.zeros((arr_points.shape[0], 3))
        vector[:, 1] = arr_points[:, 0] * pixel_spacing[0]
        vector[:, 0] = arr_points[:, 1] * pixel_spacing[1]

        product = np.transpose(np.dot(matrix, np.transpose(vector)))  # x y z 1
        product = np.add(product[:, 0:3], image_position.reshape((1, 3)))

        #product = np.reshape(product, -1)

        result.append(product)

    return result


def set_roi(dcm, arr_roi, **kwargs):
    '''
    Add a ROI to a dicom dataset
    :param dcm: object - pydicom object on which the data is added
    :param arr_roi: array - n x 2 array of contour data (closed plane polygonalchain)
    :param kwargs: strings - additional tag information (tag_3006_0026, tag_3006_0036)
    :return: True
    '''
    tag_3006_0022 = kwargs.get('tag_3006_0022', -1)  # ROI Number
    tag_3006_0024 = kwargs.get('tag_3006_0024', -1)  # Referenced Frame of Reference
    tag_3006_0026 = kwargs.get('tag_3006_0026', 'ROI_' + datetime.datetime.now().strftime("%Y&m%d_%H%M%S")) #ROI Name
    tag_3006_0036 = kwargs.get('tag_3006_0036', 'MANUAL') # ROI Generation Algorithm
    tag_3006_0042 = kwargs.get('tag_3006_0042', 'CLOSED_PLANAR')  # Contour Geometric Type
    tag_3006_0048 = kwargs.get('tag_3006_0048', -1)  # Contour Number

    if not [0x3006, 0x0002] in dcm:  # Structure Set Label
        dcm.add_new([0x3006, 0x0002], 'SH', 'ROI')

    if [0x3006, 0x0008] in dcm:  # Structure Set Modification Date
        dcm[0x3006, 0x0008].value = str(datetime.datetime.now().strftime("%Y%m%d"))
    else:
        dcm.add_new([0x3006, 0x0008], 'DA', str(datetime.datetime.now().strftime("%Y%m%d")))

    if [0x3006, 0x0009] in dcm:  # Structure Set Modification Time
        dcm[0x3006, 0x0009].value = str(datetime.datetime.now().strftime("%H%M%S"))
    else:
        dcm.add_new([0x3006, 0x0009], 'TM', str(datetime.datetime.now().strftime("%H%M%S")))

    append = True
    if [0x3006, 0x0010] in dcm:  # Referenced Frame of Reference Sequence
        for i in range(0, len(dcm[0x3006, 0x0010].value)):
            if dcm[0x3006, 0x0010][i][0x0020, 0x0052].value == dcm[0x0020, 0x0052].value:
                append = False
                break
        if append:
            dataset = pydicom.dataset.Dataset()
            dataset.add_new([0x0020, 0x0052], 'UI', str(dcm[0x0020, 0x0052].value))

            sequence = dcm[0x3006, 0x0010].value
            sequence.append(dataset)
            dcm[0x3006, 0x0010].value = sequence
    else:
        dataset = pydicom.dataset.Dataset()
        dataset.add_new([0x0020, 0x0052], 'UI', str(dcm[0x0020, 0x0052].value))
        dcm.add_new([0x3006, 0x0010], 'SQ', pydicom.sequence.Sequence([dataset]))

    if [0x3006, 0x0020] in dcm:  # Structure Set Roi Sequence
        roi_number = []

        for i in range(0, len(dcm[0x3006, 0x0020].value)):
            roi_number.append(int(dcm[0x3006, 0x0020][i][0x3006, 0x0022].value))

        if len(roi_number) == 0:
            roi_number = 1
        else:
            roi_number = max(roi_number) + 1
    else:
        dcm.add_new([0x3006, 0x0020], 'SQ', None)
        roi_number = 1

    if tag_3006_0022 != -1:
        roi_number = int(tag_3006_0022)

    tag_3006_0026 = tag_3006_0026 + " #" + str(roi_number) if tag_3006_0026 == 'MARISSA ROI' else tag_3006_0026

    dataset = pydicom.dataset.Dataset()
    dataset.add_new([0x3006, 0x0022], 'IS', str(roi_number))
    if tag_3006_0024 == -1:
        dataset.add_new([0x3006, 0x0024], 'UI', str(dcm[0x0020, 0x0052].value))
    else:
        dataset.add_new([0x3006, 0x0024], 'UI', str(tag_3006_0024))
    dataset.add_new([0x3006, 0x0026], 'LO', tag_3006_0026)
    dataset.add_new([0x3006, 0x0036], 'CS', tag_3006_0036)

    sequence = dcm[0x3006, 0x0020].value
    sequence.append(dataset)
    dcm[0x3006, 0x0020].value = sequence

    dataset = pydicom.dataset.Dataset()
    dataset.add_new([0x3006, 0x0084], 'IS', str(roi_number))
    contour = pydicom.dataset.Dataset()
    if tag_3006_0048 == -1:
        contour.add_new([0x3006, 0x0048], 'IS', "1")
    else:
        contour.add_new([0x3006, 0x0048], 'IS', str(tag_3006_0048))
    contour.add_new([0x3006, 0x0042], 'CS', str(tag_3006_0042))
    contour.add_new([0x3006, 0x0046], 'IS', str(np.shape(arr_roi)[0]))
    #contour.add_new([0x3006, 0x0050], 'DS', transform_ics_rcs(dcm, arr_roi)[0].tolist())
    contour.add_new([0x3006, 0x0050], 'DS', transform_ics_rcs(dcm, arr_roi)[0].flatten().tolist())
    dataset.add_new([0x3006, 0x0040], 'SQ', pydicom.sequence.Sequence([contour]))

    if not [0x3006, 0x0039] in dcm:  # ROI Contour Sequence
        dcm.add_new([0x3006, 0x0039], 'SQ', None)

    sequence = dcm[0x3006, 0x0039].value
    sequence.append(dataset)
    dcm[0x3006, 0x0039].value = sequence

    return True


def get_roi(dcm, **kwargs):
    '''
    Returns the ics data from the ROI contour tag
    :param dcm: dicom object
    :param kwargs: get: either "all" to read all roi contours or a list with [[roi_num], [contour_num]], include_values: False to get only contour points, True to get the enclosed values (only possible if geometric type is CLOSED_PLANAR or POINT)
    :return: mock object with roi_num, contour_num, geometric_type, roi_ics_points, roi_name, include_values, include_values_location
    '''

    options = mock.Mock()
    options.get = kwargs.get("get", "all")
    options.include_values = kwargs.get("include_values", False)

    result = []

    index_roi_num = []
    index_contour_num = []

    if str(options.get) == "all":
        if [0x3006, 0x0039] in dcm:
            for i in range(0, len(dcm[0x3006, 0x0039].value)):
                for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                    index_roi_num.append(dcm[0x3006, 0x0039][i][0x3006, 0x0084].value)
                    index_contour_num.append(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value)

        index_roi = np.asarray([index_roi_num, index_contour_num])

    else:
        index_roi = np.asarray(options.get)

    for i in range(len(index_roi[0])):
        read = mock.Mock()
        read.roi_num = index_roi[0, i] #dcm[0x3006, 0x0039][index_roi[0][i]][0x3006, 0x0084].value
        read.contour_num = index_roi[1, i] #dcm[0x3006, 0x0039][index_roi[0][i]][0x3006, 0x0040][index_roi[1][i]][0x3006, 0x0048].value

        for j in range(0, len(dcm[0x3006, 0x0039].value)):
            if int(read.roi_num) == int(dcm[0x3006, 0x0039][j][0x3006, 0x0084].value):
                index_roi_num = int(j)
                break

        for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
            if int(read.contour_num) == dcm[0x3006, 0x0039][index_roi_num][0x3006, 0x0040][j][0x3006, 0x0048].value:
                index_contour_num = int(j)
                break

        read.geometric_type = dcm[0x3006, 0x0039][index_roi_num][0x3006, 0x0040][index_contour_num][0x3006, 0x0042].value
        read.roi_ics_points = transform_ics_rcs(dcm, [[read.roi_num], [read.contour_num]])[0]

        for j in range(0, len(dcm[0x3006, 0x0020].value)):
            if dcm[0x3006, 0x0020][j][0x3006, 0x0022].value == read.roi_num:
                read.roi_name = dcm[0x3006, 0x0020][j][0x3006, 0x0026].value
                break

        if options.include_values:
            read.include_values, read.include_values_location = get_roi_values(dcm, get=[[read.roi_num], [read.contour_num]])
        else:
            read.include_values = [[None]]
            read.include_values_location = [[None]]

        result.append(read)

    return result


def get_roi_values(dcm, **kwargs):
    '''
    :param dcm: object - dicom object
    :param kwargs: get: either "all" to read all roi contours or a list with [[roi_num], [contour_num]]
    :return: n x 1 array of values and n x 2 array of value location in ics
    '''

    options = mock.Mock()
    options.get = kwargs.get("get", "all")

    values = []
    values_location = []
    pixel_data = get_pixel_data(dcm)

    index_roi_num = []
    index_contour_num = []

    if str(options.get) == "all":
        if [0x3006, 0x0039] in dcm:
            for i in range(0, len(dcm[0x3006, 0x0039].value)):
                for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                    index_roi_num.append(dcm[0x3006, 0x0039][i][0x3006, 0x0084].value)
                    index_contour_num.append(dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value)

        index_roi = np.asarray([index_roi_num, index_contour_num])

    else:
        index_roi = np.asarray(options.get)

    for i in range(len(index_roi[0])):
        roi_num = index_roi[0, i] #dcm[0x3006, 0x0039][index_roi[0][i]][0x3006, 0x0084].value
        contour_num = index_roi[1, i] #dcm[0x3006, 0x0039][index_roi[0][i]][0x3006, 0x0040][index_roi[1][i]][0x3006, 0x0048].value

        for j in range(0, len(dcm[0x3006, 0x0039].value)):
            if int(roi_num) == int(dcm[0x3006, 0x0039][j][0x3006, 0x0084].value):
                index_roi_num = int(j)
                break

        for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
            if int(contour_num) == dcm[0x3006, 0x0039][index_roi_num][0x3006, 0x0040][j][0x3006, 0x0048].value:
                index_contour_num = int(j)
                break

        geometric_type = dcm[0x3006, 0x0039][index_roi_num][0x3006, 0x0040][index_contour_num][0x3006, 0x0042].value
        points = transform_ics_rcs(dcm, [[roi_num], [contour_num]])[0]

        if geometric_type == "CLOSED_PLANAR":
            roi = tool_general.polygonalchain2mask(points[:, [1, 0]], pixel_data)
            values_location.append(roi[0])
            values.append(roi[2])

        elif geometric_type == "POINT":
            values.append(pixel_data[points[:, 0], points[:, 1]])
            values_location.append(points)
        else:
            values.append([None])
            values_location.append([None])

    return values, values_location


def del_roi_where(dcm, where):
    '''
    Deltes Rois where name contains specific string
    :param dcm:
    :param where: str to search in ROI name
    :return: True
    '''

    if [0x3006, 0x0020] in dcm:
        roi_num = []
        for i in range(0, len(dcm[0x3006, 0x0020].value)):
            if where in dcm[0x3006, 0x0020][i][0x3006, 0x0026].value:
                roi_num.append(dcm[0x3006, 0x0020][i][0x3006, 0x0022].value)
        roi_num = np.asarray(roi_num)

        del_roi(dcm, roi_num)

    return True


def del_roi(dcm, arr_roi=None):
    '''

    :param dcm:
    :param arr_roi:
    :return:
    '''

    del_roi = []
    del_roi_con = []

    for i in range(0, len(dcm[0x3006, 0x0039].value)):
        if int(dcm[0x3006, 0x0039][i][0x3006, 0x0084].value) in arr_roi or arr_roi is None:
            del_roi.append(dcm[0x3006, 0x0039][i][0x3006, 0x0084].value)
            if [0x3006, 0x0039] in dcm:
                for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                    del_roi_con.append([dcm[0x3006, 0x0039][i][0x3006, 0x0084].value, dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value])

    del_roi = np.asarray(del_roi)
    del_roi_con = np.asarray(del_roi_con)

    del_roi_contour(dcm, del_roi_con)

    for j in range(len(dcm[0x3006, 0x0020].value)-1, -1, -1):
        if int(dcm[0x3006, 0x0020][j][0x3006, 0x0022].value) in del_roi:
            del dcm[0x3006, 0x0020].value[j]


    #del dcm[0x3006, 0x0039].value[i]

    return True


def del_roi_contour(dcm, arr_roi_contour=None):
    '''
    Deletes one or multiple ROIs or Contours from a dicom dataset
    :param dcm: object - dicom object
    :param arr_roi_contour: None - deletes all ROI data / list of lists - [[roi_num, contour_num]] -> both are specified (only specific contour numbers are deleted from specific ROI numbers), both are None (same behaviour as if arr_points = None), roi_num = None and contour_num is specific list (specific contour number is deleted from all ROIs), roi_num is specific list and contour_num = None (specific ROIs are deleted including the Contours)
    :return: True
    '''

    if [0x3006, 0x0039] in dcm:
        if arr_roi_contour is None:
            del_arr = []
            for i in range(0, len(dcm[0x3006, 0x0039].value)):
                for j in range(0, len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)):
                    del_arr.append([dcm[0x3006, 0x0039][i][0x3006, 0x0084].value, dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value])
            del_arr = np.asarray(del_arr)
        else:
            del_arr = np.copy(arr_roi_contour)

        for i in range(len(dcm[0x3006, 0x0039].value)-1, -1, -1):
            for j in range(len(dcm[0x3006, 0x0039][i][0x3006, 0x0040].value)-1, -1, -1):
                if np.asarray([dcm[0x3006, 0x0039][i][0x3006, 0x0084].value, dcm[0x3006, 0x0039][i][0x3006, 0x0040][j][0x3006, 0x0048].value]) in del_arr:
                    del dcm[0x3006, 0x0039].value[i] #[0x3006, 0x0040].value[j]
    return True


def get_marissa_annotation(path):
    dcm = pydicom.dcmread(path)

    try:
        seq_roi = dcm[0x0033, 0x3310].value # ROI seq
        seq_bb = dcm[0x0033, 0x3320].value # BB Seq
        seq_lh = dcm[0x0033, 0x3330].value # LH Seq
        seq_rmv = dcm[0x0033, 0x3340].value # RMV Seq
        seq_map = dcm[0x0033, 0x33d0].value # Mask (stage 3)
        seq_err = dcm[0x0033, 0x33e0].value # error map
        seq_std = dcm[0x0033, 0x33f0].value # standardization map

        roi = []
        roi_uid = []
        roi_tissue = []
        for i in range(len(seq_roi)):
            roi.append(seq_roi[i][0x0033, 0x331f].value + "_" + seq_roi[i][0x0033, 0x3311].value.replace(".", "_"))
            roi_uid.append(seq_roi[i][0x0033, 0x3311].value)
            roi_tissue.append(seq_roi[i][0x0033, 0x331f].value)

        bb = [[] for _ in range(len(roi_uid))]
        bb_uid = [[] for _ in range(len(roi_uid))]
        bb_type = [[] for _ in range(len(roi_uid))]
        for i in range(len(seq_bb)):
            index = roi_uid.index(seq_bb[i][0x0033, 0x3311].value)
            bb[index].append(np.asarray(seq_bb[i][0x0033, 0x332f].value).flatten())
            bb_uid[index].append(seq_bb[i][0x0033, 0x3321].value)
            bb_type[index].append(seq_bb[i][0x0033, 0x3322].value)

        lh = [[] for _ in range(len(roi_uid))]
        for i in range(len(roi_uid)):
            lh[i].append([[] for _ in range(len(bb_uid[i]))])

        for i in range(len(seq_lh)):
            index_roi = roi_uid.index(seq_lh[i][0x0033, 0x3311].value)
            index_bb = bb_uid[index_roi].index(seq_lh[i][0x0033, 0x3321].value)
            lh[index_roi][index_bb] = np.asarray(seq_lh[i][0x0033, 0x333f].value).flatten()

        rmv = [[] for _ in range(len(roi_uid))]
        rmv_id = [[] for _ in range(len(roi_uid))]
        rmv_conn = [[] for _ in range(len(roi_uid))]
        for i in range(len(roi_uid)):
            rmv[i].append([[] for _ in range(len(bb_uid[i]))])
            rmv_id[i].append([[] for _ in range(len(bb_uid[i]))])
            rmv_conn[i].append([[] for _ in range(len(bb_uid[i]))])

        for i in range(len(seq_rmv)):
            index_roi = roi_uid.index(seq_rmv[i][0x0033, 0x3311].value)
            index_bb = bb_uid[index_roi].index(seq_rmv[i][0x0033, 0x3321].value)
            rmv_id[index_roi][index_bb].append(seq_rmv[i][0x0033, 0x3341].value)
            try:
                rmv[index_roi][index_bb].append(np.asarray(seq_rmv[i][0x0033, 0x334f].value).reshape((-1, 2)))
            except:
                pass
            rmv_conn[index_roi][index_bb].append(seq_rmv[i][0x0033, 0x3342].value)

        result = mock.Mock()

        result.roi_uid = roi_uid
        result.roi_tissue = roi_tissue
        result.roi = roi
        result.bb_uid = bb_uid
        result.bb_type = bb_type
        result.bb = bb
        result.lh = lh
        result.rmv_id = rmv_id
        result.rmv_conn = rmv_conn
        result.rmv = rmv

    except:
        print("WARNING: marissa.toolbox.tools.tool_pydicom.get_marissa_annotation: No annotation found or runtime error.")
        result = None

    return result


def get_marissa_anotation_map(pixel_data, bb, rmv, lh, num_roi, num_bb=None, **kwargs):
    stage = kwargs.get("stage", 3)
    expand = kwargs.get("expand", False)

    if num_bb is None:
        num_bb_intern = np.arange(0, len(bb[num_roi]))
    elif isinstance(num_bb, list):
        num_bb_intern = num_bb
    else:
        num_bb_intern = [num_bb]

    result = np.zeros(pixel_data.shape)

    for j in num_bb_intern:
        map_within = np.ones(np.shape(pixel_data))

        if stage >= 2:
            map_within = (pixel_data >= lh[num_roi][j][0]) & (pixel_data <= lh[num_roi][j][1])

        if stage >= 3:
            rmv_map = np.zeros(np.shape(pixel_data))
            if num_roi >= 0 and j >= 0 and len(rmv[num_roi][j]) > 0:
                for i in range(0, len(rmv[num_roi][j])):

                    if len(rmv[num_roi][j][i]) > 0:
                        arr_points = np.asarray(rmv[num_roi][j][i])
                        #arr_points[:, [0, 1]] = arr_points[:, [1, 0]]  # swap columns as image xy, dcm yx

                        if len(rmv[num_roi][j][i]) >= 3:
                            polygon = tool_general.get_cubic_spline_2D(arr_points, 100)

                            idx, map_rmv, _ = tool_general.polygonalchain2mask(polygon[:, [1, 0]], pixel_data)
                            rmv_map = rmv_map + map_rmv

            rmv_map[rmv_map > 1] = 1

            map_within = map_within - rmv_map
            map_within[map_within < 0] = 0

        within = np.asarray(map_within, np.int)

        within_bb = within[int(bb[num_roi][j][1]):(int(bb[num_roi][j][1]) + int(bb[num_roi][j][3])), int(bb[num_roi][j][0]):(int(bb[num_roi][j][0]) + int(bb[num_roi][j][2]))]

        #components, lcc_idx = general_utilities.connected_components_old(within_bb)
        components, lcc_idx = tool_general.connected_components(within_bb)

        if expand:
            components, lcc_idx = tool_general.connected_components(within, seed =np.asarray(components[lcc_idx])[0, :] + np.array([int(bb[num_roi][j][1]), int(bb[num_roi][j][0])]))
            within = np.zeros(np.shape(within))
            within[np.asarray(components[lcc_idx])[:, 0], np.asarray(components[lcc_idx])[:, 1]] = 1
        else:
            within = np.zeros(np.shape(within))
            within[int(bb[num_roi][j][1]) + np.asarray(components[lcc_idx])[:, 0], int(bb[num_roi][j][0]) + np.asarray(components[lcc_idx])[:, 1]] = 1

        result = result + within

    result[result>1] = 1
    result = np.array(result, dtype=bool)
    result = np.logical_or(result, get_segmentation_holes(result, exclude=2)).astype("int")

    return result

def get_segmentation_mask(path, **kwargs):
    annotations = get_marissa_annotation(path)
    tissue = kwargs.get("tissue", None)
    if tissue is None:
        roi_id = annotations.roi_tissue.index(tissue)
    else:
        roi_id = 0

    dcm = pydicom.dcmread(path)
    out_0 = get_marissa_anotation_map(get_pixel_data(dcm, **kwargs), annotations.bb, annotations.rmv, annotations.lh, roi_id, num_bb=None, stage=3)
    result = np.logical_or(out_0, get_segmentation_holes(out_0, exclude=2))

    return result


def get_segmentation_mask_prediction(path, model_bb, model_ps, model_corr, exclude=None):
    import cv2
    from PIL import Image

    dcm = pydicom.dcmread(path)

    h, w, num_bin = 256, 256, 100
    pixel_data=np.copy(get_pixel_data(dcm, rescale=True, representation=False))

    img = np.copy(pixel_data)
    img = img.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(img)
    img = np.dstack((img, img, img))
    img = cv2.cvtColor(np.array(Image.fromarray(img)), cv2.COLOR_RGB2BGR)

    if model_bb is not None and model_ps is not None and model_corr is not None and exclude is None:
        pred_bb = model_bb.predict([img])

        try:
            bbox = [pred_bb[0][0][2], pred_bb[0][0][3], pred_bb[0][0][4], pred_bb[0][0][5]]
        except:
            bbox = [0, 0, np.shape(pixel_data)[1], np.shape(pixel_data)[0]]

        #bbox = [pred_bb[0][0][2], pred_bb[0][0][3], pred_bb[0][0][4], pred_bb[0][0][5]]

        #pixel_data=np.copy(self.dhdcm.get_pixel_data(indices=[self.num_ima], rescale=False)[0])

        data_max = np.max(pixel_data)
        #pixel_data_bb = pixel_data.flatten()
        pixel_data_bb = pixel_data[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].flatten()

        data_hist, _, _ = tool_general.histogram(pixel_data_bb, num_bin, (np.min(pixel_data_bb), np.max(pixel_data_bb)))

        lh_limits = model_ps.predict(np.array([np.array(data_hist)]))
        lh_limits = np.sort(lh_limits * data_max)

        map_within = (pixel_data >= lh_limits[0][0]) & (pixel_data <= lh_limits[0][1])
        within = np.asarray(map_within, np.int)

        within_bb = within[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        components, lcc_idx = tool_general.connected_components(within_bb)

        components, lcc_idx = tool_general.connected_components(within, seed =np.asarray(components[lcc_idx])[0, :] + np.array([int(bbox[1]), int(bbox[0])]))
        within = np.zeros(np.shape(within))
        within[np.asarray(components[lcc_idx])[:, 0], np.asarray(components[lcc_idx])[:, 1]] = 1

        #within = np.zeros(np.shape(within))
        #within[int(bbox[1]) + np.asarray(components[lcc_idx])[:, 0], int(bbox[0]) + np.asarray(components[lcc_idx])[:, 1]] = 1
        mask = np.array(within, dtype=bool)
        mask = np.logical_or(mask, get_segmentation_holes(mask, 2))

        bb_temp = np.zeros(np.shape(pixel_data))
        bb_temp[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

        in_0 = np.array(Image.fromarray(pixel_data).resize((h, w), Image.NEAREST)).astype('float32')
        in_0 = in_0 / np.max(in_0)

        in_1 = np.array(Image.fromarray(bb_temp).resize((h, w), Image.NEAREST)).astype('float32')
        in_1 = in_1 / np.max(in_1)

        in_2 = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST)).astype('float32')
        in_2 = in_2 / np.max(in_2)

        out = model_corr.predict(np.array([np.concatenate((np.expand_dims(in_0, axis=2), np.expand_dims(in_1, axis=2), np.expand_dims(in_2, axis=2)), axis=2)]))
        out = np.array(Image.fromarray(out[0,:,:,0]).resize((np.shape(pixel_data)[1], np.shape(pixel_data)[0]), Image.NEAREST)).astype('float32')
        out = np.array(np.round(out), dtype=bool)
    elif model_corr is not None and model_bb is not None and model_ps is not None:
        if str(exclude) == "1" or exclude == "bb":
            pred_bb = model_bb.predict([img])

            try:
                bbox = [pred_bb[0][0][2], pred_bb[0][0][3], pred_bb[0][0][4], pred_bb[0][0][5]]
            except:
                bbox = [0, 0, np.shape(pixel_data)[1], np.shape(pixel_data)[0]]

            data_max = np.max(pixel_data)

            pixel_data_bb = pixel_data[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].flatten()

            data_hist, _, _ = tool_general.histogram(pixel_data_bb, num_bin, (np.min(pixel_data_bb), np.max(pixel_data_bb)))

            lh_limits = model_ps.predict(np.array([np.array(data_hist)]))
            lh_limits = np.sort(lh_limits * data_max)

            map_within = (pixel_data >= lh_limits[0][0]) & (pixel_data <= lh_limits[0][1])
            within = np.asarray(map_within, np.int)

            within_bb = within[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            components, lcc_idx = tool_general.connected_components(within_bb)

            components, lcc_idx = tool_general.connected_components(within, seed =np.asarray(components[lcc_idx])[0, :] + np.array([int(bbox[1]), int(bbox[0])]))
            within = np.zeros(np.shape(within))
            within[np.asarray(components[lcc_idx])[:, 0], np.asarray(components[lcc_idx])[:, 1]] = 1

            mask = np.array(within, dtype=bool)
            mask = np.logical_or(mask, get_segmentation_holes(mask, 2))

            bb_temp = np.zeros(np.shape(pixel_data))
            bb_temp[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

            in_0 = np.array(Image.fromarray(pixel_data).resize((h, w), Image.NEAREST)).astype('float32')
            in_0 = in_0 / np.max(in_0)

            in_2 = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST)).astype('float32')
            in_2 = in_2 / np.max(in_2)

            out = model_corr.predict(np.array([np.concatenate((np.expand_dims(in_0, axis=2), np.expand_dims(in_2, axis=2)), axis=2)]))
            out = np.array(Image.fromarray(out[0,:,:,0]).resize((np.shape(pixel_data)[1], np.shape(pixel_data)[0]), Image.NEAREST)).astype('float32')
            out = np.array(np.round(out), dtype=bool)

        elif str(exclude) == "2" or exclude == "ps":
            pred_bb = model_bb.predict([img])

            try:
                bbox = [pred_bb[0][0][2], pred_bb[0][0][3], pred_bb[0][0][4], pred_bb[0][0][5]]
            except:
                bbox = [0, 0, np.shape(pixel_data)[1], np.shape(pixel_data)[0]]

            data_max = np.max(pixel_data)

            pixel_data_bb = pixel_data[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].flatten()


            bb_temp = np.zeros(np.shape(pixel_data))
            bb_temp[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

            in_0 = np.array(Image.fromarray(pixel_data).resize((h, w), Image.NEAREST)).astype('float32')
            in_0 = in_0 / np.max(in_0)

            in_1 = np.array(Image.fromarray(bb_temp).resize((h, w), Image.NEAREST)).astype('float32')
            in_1 = in_1 / np.max(in_1)

            out = model_corr.predict(np.array([np.concatenate((np.expand_dims(in_0, axis=2), np.expand_dims(in_1, axis=2)), axis=2)]))
            out = np.array(Image.fromarray(out[0,:,:,0]).resize((np.shape(pixel_data)[1], np.shape(pixel_data)[0]), Image.NEAREST)).astype('float32')
            out = np.array(np.round(out), dtype=bool)
        else:
            raise ValueError("Value for exclude unknown, only None, 1, 2, bb or ps are feasible")

    elif model_corr is not None:
        in_0 = np.array(Image.fromarray(pixel_data).resize((h, w), Image.NEAREST)).astype('float32')
        in_0 = in_0 / np.max(in_0)
        out = model_corr.predict(np.array([np.expand_dims(in_0, axis=2)]))
        out = np.array(Image.fromarray(out[0,:,:,0]).resize((np.shape(pixel_data)[1], np.shape(pixel_data)[0]), Image.NEAREST)).astype('float32')
        out = np.array(np.round(out), dtype=bool)

    return out

def get_segmentation_holes(array, exclude=1):
    try:
        inv_array = np.ones(np.shape(array))
        inv_array[array.astype(int) == 1] = 0

        cc, _ = tool_general.connected_components(inv_array)

        result = np.zeros(np.shape(array))

        for i in range(max(0, len(cc)-exclude)):
            result[tuple(np.array(cc[i+exclude]).T.tolist())] = 1

#        cc_len = []

#        for i in range(len(cc)):
#            cc_len.append(len(cc[i]))

#        indices = np.argsort(np.array(cc_len))
#        indices = indices[-exclude:]
#        indices = np.sort(indices)

#        result = np.zeros(np.shape(array))

#        for i in range(len(cc)):
#            if not i in indices:
#                set_indices = np.array(cc[i])
#                set_indices = tuple(np.array(set_indices).T.tolist())
#                result[set_indices] = 1
    except:
        result = np.zeros(np.shape(array))
    return result.astype(bool)

def get_T1MES_phantom_ROI(path):
    dcm = pydicom.dcmread(path)
    roi = get_roi(dcm, include_values=True)

    if len(roi) > 0:
        result = mock.Mock()
        result.roi_name = []
        result.roi_values = []
        result.roi_values_location = []
        result.roi_contour = []
        result.roi_num = []

        for i in range(len(roi)):
            result.roi_name.append(roi[i].roi_name)
            result.roi_values.append(np.concatenate(roi[i].include_values, axis = None))
            result.roi_values_location.append(np.concatenate(roi[i].include_values_location, axis=0))
            result.roi_contour.append(roi[i].roi_ics_points)
            result.roi_num.append(roi[i].roi_num)
    else:
        result = None

    return result

def bb_to_mask(array, bb):
    mask = np.zeros(np.shape(np.squeeze(array)))
    mask[int(bb[1]):int(bb[1]+bb[3]), int(bb[0]):int(bb[0]+bb[2])] = 1
    return mask

def bb_to_excerpt(array, bb):
    excerpt = np.squeeze(array)[int(bb[1]):int(bb[1]+bb[3]), int(bb[0]):int(bb[0]+bb[2])]
    return excerpt

def cc_to_mask(array, cc):
    map_within = (array >= cc[0]) & (array <= cc[1])
    within = np.asarray(map_within, np.int)

    components, lcc_idx = tool_general.connected_components(map_within)
    within = np.zeros(np.shape(within))
    within[np.asarray(components[lcc_idx])[:, 0], np.asarray(components[lcc_idx])[:, 1]] = 1
    return within

def mask_to_excerpt(array, mask, flatten=True):
    if flatten:
        excerpt = array[np.nonzero(mask)].flatten()
    else:
        indeces = np.argwhere(mask)
        excerpt = array[np.min(indeces[:,0]):np.max(indeces[:,0])+1, np.min(indeces[:,1]):np.max(indeces[:,1]+1)]
    return excerpt

def main():
    #import os
    #from marissa.modules.datahandler import dcm as data_handler_dcm
    #path = "C:\\Users\\Omen\\Desktop\\Test"

    #for root, _, files in os.walk(path):
    #    for file in files:
    #        file_path = os.path.join(root, file)
    #        dhdcm = data_handler_dcm.Data()
    #        dhdcm.load(file_path, "all")
    #        dcm = dhdcm.list[0]
    #        pixel_data = get_pixel_data(dcm, False)
    #        (roi_uid, roi, bb_uid, bb_type, bb, lh, rmv_id, rmv_conn, rmv) = get_marissa_annotation(dcm)
    #        map = get_marissa_anotation_map(pixel_data,bb,rmv,lh,0,None)
    #        print("ok")

    a = np.zeros((100, 100))
    b = np.ones((20, 20))
    c = np.zeros((10,16))
    d = np.ones((10, 1))
    a[20:40, 20:40] = b
    a[25:35, 22:38] = c
    a[25:35, 25:26] = d
    res = get_segmentation_holes(a.astype(bool), 2)
    m = 0


if __name__ == "__main__":
    main()