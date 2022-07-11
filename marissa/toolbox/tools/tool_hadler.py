from scipy.ndimage.filters import gaussian_filter
import os
import pickle
import numpy as np
import copy
from skimage.measure import label, find_contours
from skimage.morphology import convex_hull_image


def mask2contour(segmentation, subpixelresolution=2, check=None):
    if check is None:
        segm_lv1 = segmentation #[:,:,0]
        segm_lv1 = getLargestCC(segm_lv1)
        segm_lv1 = np.kron(segm_lv1,np.ones((subpixelresolution, subpixelresolution))) # upscaling for high res
        segm_lv1 = gaussian_filter(segm_lv1, sigma=2)  # better contours
        contours = sorted(find_contours(segm_lv1.squeeze(), 0.5), key=len)
    else:
        contours = sorted(find_contours(segmentation.squeeze(), 0.5), key=len)
        segm_lv1 = segmentation #[:,:,0]
        segm_lv1 = getLargestCC(segm_lv1)
        segm_lv1 = np.kron(segm_lv1,np.ones((subpixelresolution, subpixelresolution))) # upscaling for high res
        segm_lv1 = gaussian_filter(segm_lv1, sigma=2)  # better contours

        if len(contours) >= check:
            for i in range(50):
                contours = sorted(find_contours(segm_lv1.squeeze(), 0.5-i/100), key=len)
                if len(contours) >= check:
                    break
        else:
            contours = sorted(find_contours(segm_lv1.squeeze(), 0.5), key=len)
            
    return contours


def prepare4LazyLunaSAX(path, data, contours, subpixelresolution=2, reader="me", num_contours=None, save_dicom=True):
    dcm = copy.deepcopy(data)
    dcm[0x0008, 0x103e].value = "OVERWRITTEN: pre_MOLLI MOCO T1 mit"
    h, w = dcm.pixel_array.shape[0], dcm.pixel_array.shape[1]
    res = np.array(dcm.PixelSpacing).astype("float")
    sopuid = dcm.SOPInstanceUID

    contours_new = copy.deepcopy(contours)

    if num_contours is None:
        num_contours = len(contours_new)

    try:
        for i in range(len(contours_new)):
            array = contours_new[i]
            array[:, [0,1]] = array[:, [1,0]]
            contours_new[i] = array
    except:
        pass

    conts_lv1 = [c.tolist() for c in contours_new]

    if len(conts_lv1)>=2 and num_contours==2:
        #conts_lv1=[[[b,a] for c in conts_lv1 for d in c for [a,b] in d]]

        lv_epi = [conts_lv1[-1]]
        lv_endo = [conts_lv1[-2]]

        d = {'lv endocardium':      {'contour': lv_endo,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv epicardium':       {'contour': lv_epi,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv papillary muscle': {'contour': []},
              'rv endocardium':      {'contour': []},
              'rv epicardium': {'contour': []},
              'rv papillary muscle': {'contour': []},
              None: {'contour': []},
              'left atrium': {'contour': []},
              'viewer': {'hotspot': [93.7963, 102.087], 'zoom': 4.74074}, # ?
              'window': {'center': 372.0, 'width': 890.0}}                # ?


    elif len(conts_lv1)==1 or num_contours==1:
        lv_epi = [conts_lv1[-1]]

        d = {'lv endocardium':      {'contour': []},
              'lv epicardium':       {'contour': lv_epi,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv papillary muscle': {'contour': []},
              'rv endocardium':      {'contour': []},
              'rv epicardium': {'contour': []},
              'rv papillary muscle': {'contour': []},
              None: {'contour': []},
              'left atrium': {'contour': []},
              'viewer': {'hotspot': [93.7963, 102.087], 'zoom': 4.74074}, # ?
              'window': {'center': 372.0, 'width': 890.0}}                # ?
    else:
        return False

    if save_dicom:
        try:
            path_dicom = os.path.join(path, "MARISSA2LazyLuna", "images", sopuid)
            os.makedirs(path_dicom)
            dcm.save_as(os.path.join(path_dicom, sopuid+'.dcm'))
        except:
            pass

    try:
        path_annotation = os.path.join(path, "MARISSA2LazyLuna", reader, sopuid, "sub_annotations")
        path_contour_file = os.path.join(path, "MARISSA2LazyLuna", reader, sopuid)
        os.makedirs(path_annotation)
    except:
        pass

    open(os.path.join(path_contour_file, "contours.txt"), "w+").close()
    with open(os.path.join(path_annotation, sopuid+'.pickle'), 'wb') as f:
        pickle.dump(d, f)
        f.close()

    return True


def getLargestCC(segmentation):
    if len(np.shape(segmentation)) == 3:
        lcc = []
        for i in range(np.shape(segmentation)[2]):
            labels = label(np.squeeze(segmentation[:, :, i]))
            if labels.max() == 0:
                lcc.append(segmentation)
            else:
                lcc.append(labels == np.argmax(np.bincount(labels.flat)[1:])+1)
        largestCC = np.array(lcc)
        largestCC = np.moveaxis(largestCC, 0, -1)
    else:
        labels = label(segmentation)
        if labels.max() == 0:
            largestCC = segmentation
        else:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def convex_hull(segmentation):
    return convex_hull_image(segmentation)


def segm2contour(segmentation, threshold=0.5):
    return find_contours(segmentation, threshold)
