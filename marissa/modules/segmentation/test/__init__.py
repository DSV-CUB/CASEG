import copy
import os
import numpy as np
import uuid
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import pandas as pd
from datetime import datetime

from marissa.modules.segmentation import models as mmsmodels
from marissa.modules.segmentation import generators as mmsgenerators
from marissa.toolbox import tools

def test_equivalence(path_intra_1, path_intra_2, path_weights, path_data, **kwargs):
    CompareVsRef=kwargs.get("comparevsref", True)


    # REFERENCE EQUIVALENCE MARGIN DATA
    gen_intra_1 = mmsgenerators.unet.Setup(path_intra_1, None, "TEST", ws_dir=path_intra_1, mask_mode="RASTERIZE")
    gen_intra_2 = mmsgenerators.unet.Setup(path_intra_2, None, "TEST", ws_dir=path_intra_2, mask_mode="RASTERIZE")
    
    info1 = np.array(gen_intra_1.information)[:,1]
    info2 = np.array(gen_intra_2.information)[:,1]
    
    matches, match_ID1, match_ID2 = np.intersect1d(info1, info2, False, True)
    
    data_reference = []
    for i in range(len(match_ID1)):
        d1 = gen_intra_1.x[match_ID1[i]]
        d2 = gen_intra_2.x[match_ID2[i]]
        c1 = gen_intra_1.y_contour[match_ID1[i]]
        c2 = gen_intra_2.y_contour[match_ID2[i]]
        m1 = tools.tool_general.contour2mask(c1, d1, False, "RASTERIZE")
        m2 = tools.tool_general.contour2mask(c2, d2, False, "RASTERIZE")
        d1 = d1[np.argwhere(m1)[:,0], np.argwhere(m1)[:,1]]
        d2 = d2[np.argwhere(m2)[:,0], np.argwhere(m2)[:,1]]
        data_reference.append(np.mean(d2) - np.mean(d1))
    data_reference = np.array(data_reference)
    
    # TEST DATA
    list_models = []
    list_names = []
    model_bb = None
    for root, _, files in os.walk(path_weights):
        for file in files:
            if file.endswith(".pickle"):
                model = mmsmodels.unet.Setup()
                try:
                    model.load(os.path.join(root, file))
                except:
                    continue
    
                if model.configuration.model_settings.model_predictor.upper() == "BB" or model.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                    model_bb = copy.copy(model)
                else:
                    list_models.append(copy.copy(model))
                    list_names.append(copy.copy(model.configuration.name))

    data = mmsgenerators.unet.Setup(path_data, None, "TEST", model_bb=model_bb, ws_dir=path_data, mask_mode="RASTERIZE")
    
    data_tests = []
    test_labels = []

    if CompareVsRef: # compare to reference
        for i in range(len(list_models)):
            data.configuration.set(**list_models[i].configuration.__dict__)
            prediction, expectation = list_models[i].predict_generator(data)

            data_test = []
            for j in range(len(prediction)):
                d1 = data.x[j][np.argwhere(expectation[j])[:,0], np.argwhere(expectation[j])[:,1]]
                d2 = data.x[j][np.argwhere(prediction[j])[:,0], np.argwhere(prediction[j])[:,1]]
                try:
                    data_test.append(np.mean(d2)-np.mean(d1))
                except:
                    data_test.append(-np.mean(d1))
            data_tests.append(data_test)
            test_labels.append(list_names[i] + " vs REF")
    else: # compare across models
        for i in range(len(list_models)-1):
            for j in range(i+1, len(list_models), 1):
                data.configuration.set(**list_models[i].configuration.__dict__)
                prediction_1, _ = list_models[i].predict_generator(data)

                data.configuration.set(**list_models[j].configuration.__dict__)
                prediction_2, _ = list_models[j].predict_generator(data)

                data_test = []
                for k in range(len(prediction_1)):
                    d1 = data.x[k][np.argwhere(prediction_1[k])[:,0], np.argwhere(prediction_1[k])[:,1]]
                    d2 = data.x[k][np.argwhere(prediction_2[k])[:,0], np.argwhere(prediction_2[k])[:,1]]
                    try:
                        data_test.append(np.mean(d2)-np.mean(d1))
                    except:
                        try:
                            data_test.append(-np.mean(d1))
                        except:
                            data_test.append(np.mean(d2))
                data_tests.append(data_test)
                test_labels.append(list_names[i] + "\nvs\n" + list_names[j])

    data_tests = np.array(data_tests)

    # run plot
    tools.tool_plot.plot_equivalence(data_tests, data_reference, 0.95, 0.05, xlabels=test_labels, **kwargs)

    return


def test_model_unet(path_data, path_weights, path_out, ws_dir=None, **kwargs):
    call = kwargs.get("call", True)
    mode=kwargs.get("mask_mode", "RASTERIZE")
    reference = kwargs.get("reference", None)
    reference_ID = (reference if type(reference) == int else None)
    name = kwargs.get("name", "")

    if call:
        data = None
        models = []

        for root, _, files in os.walk(path_weights):
            for file in files:
                if file.endswith(".pickle"):
                    model =mmsmodels.unet.Setup()
                    try:
                        model.load(os.path.join(root, file))
                    except:
                        continue

                    if model.configuration.model_settings.model_predictor.upper() == "BB" or model.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                        data = mmsgenerators.unet.Setup(path_data, None, "TEST", model_bb=model, ws_dir=ws_dir, mask_mode=mode)
                    else:
                        models.append(model)
                        if not reference is None and type(reference) == str and model.configuration.name.upper() == reference.upper():
                            reference_ID = len(models) - 1

        if data is None:
            data = mmsgenerators.unet.Setup(path_data, None, "TEST", ws_dir=ws_dir)

    else:
        data = copy.copy(path_data)
        models = path_weights

    if reference_ID is None or reference_ID > len(models) - 1:
        reference_ID = 0

    if type(models) == list:
        models_list = models
    else:
        models_list = [models]

    predictions = []
    expectations = []
    names = []

    # Model Predictions
    for model in models_list:
        data.configuration = model.configuration
        prediction, expectation = model.predict_generator(data)

        predictions.append(prediction)
        expectations.append(expectation)
        names.append(model.configuration.name)

    test_performance_export(predictions, expectations, names, data, path_out, reference_ID, name)

    return

def test_bb_unet(path_data, path_bb, path_out, ws_dir=None, **kwargs):
    call = kwargs.get("call", True)
    mode=kwargs.get("mask_mode", "RASTERIZE")
    name = kwargs.get("name", "")

    if call:
        data = None

        for root, _, files in os.walk(path_bb):
            for file in files:
                if file.endswith(".pickle"):
                    model =mmsmodels.unet.Setup()
                    try:
                        model.load(os.path.join(root, file))
                    except:
                        continue

                    if model.configuration.model_settings.model_predictor.upper() == "BB" or model.configuration.model_settings.model_predictor.upper() == "BOUNDINGBOX":
                        data = mmsgenerators.unet.Setup(path_data, None, "TEST", model_bb=model, ws_dir=ws_dir, mask_mode=mode)
                        data.configuration = model.configuration
                        break
            if not data is None:
                break

        if data is None:
            raise RuntimeError("No Bounding Box Model found")

    else:
        data = copy.copy(path_data)
        model = path_bb
        data.configuration = model.configuration

    prediction, expectation = model.predict_generator(data)
    test_performance_export([prediction], [expectation], ["BB"], data, path_out, reference_ID=0, name=name)
    return


def test_model_qcd(path_data, path_weights, path_out, path_weights_bb=None, ws_dir=None, **kwargs):
    mode=kwargs.get("mask_mode", "RASTERIZE")
    reference = kwargs.get("reference", None)
    reference_ID = (reference if type(reference) == int else None)
    name = kwargs.get("name", "")

    models = []
    model_names = []
    names_full = []
    predictions_full = []
    expectations_full = []
    names_lite = []
    predictions_lite = []
    expectations_lite = []

    # create data generator
    if path_weights_bb is None:
        model_bb = None
    else:
        model_bb =mmsmodels.unet.Setup()

        for root, _, files in os.walk(path_weights_bb):
            for file in files:
                if file.endswith(".pickle"):
                    model_bb.load(os.path.join(root, file))
                    break

    data = mmsgenerators.unet.Setup(path_data, None, "TEST", model_bb=model_bb, ws_dir=ws_dir, mask_mode=mode)

    subdirs = os.listdir(path_weights)

    for subdir in subdirs:
        if os.path.isdir(os.path.join(path_weights, subdir)) and not subdir.upper() == "UNET_BB":
            model =mmsmodels.qcd.Setup()
            model.load(os.path.join(path_weights, subdir))
            models.append(copy.copy(model))
            model_names.append(subdir)

            if not reference is None and type(reference) == str and subdir.upper() == reference.upper():
                reference_ID = 2 *(len(models) - 1)

    if reference_ID is None or reference_ID > len(models) - 1:
        reference_ID = 0

    for i in range(len(model_names)):
        prediction, expectation = models[i].predict_generator(data)

        for j in range(len(prediction[0])):
            names_full.append(model_names[i] + "_" + prediction[0][j][0] + "_qcd")
            names_lite.append(model_names[i] + "_" + prediction[0][j][0] + "_qcd_lite")

            prediction_regressor_full = []
            prediction_regressor_lite = []
            for k in range(len(prediction)):
                prediction_regressor_full.append(prediction[k][j][1])
                prediction_regressor_lite.append(prediction[k][j][2])

            predictions_full.append(prediction_regressor_full)
            predictions_lite.append(prediction_regressor_lite)

            expectations_full.append(expectation)
            expectations_lite.append(expectation)

    test_performance_export(predictions_full, expectations_full, names_full, data, path_out, reference_ID, name + "QCD")
    test_performance_export(predictions_lite, expectations_lite, names_lite, data, path_out, reference_ID, name + "QCD_Lite")

    return


def test_performance_export(predictions, expectations, names, data_copy, path_out, reference_ID=0, name=""):
    # Metrics
    metrics_list = tools.tool_general.get_metric_from_masks()
    metrics = []

    for i in range(len(names)):
        metrics_model = []
        for j in range(len(data_copy.information)):
            metrics_case = []
            for k in range(len(metrics_list)):
                #try:
                metrics_case.append(tools.tool_general.get_metric_from_masks(predictions[i][j], expectations[i][j], metrics_list[k], voxel_sizes=np.array(data_copy.pixel_spacing[j]), values=data_copy.x[j]))
                #except:
                #    wait = 1
            metrics_model.append(metrics_case)
        metrics.append(metrics_model)

    metrics = np.array(metrics)
    average = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    median = np.median(metrics, axis=1)
    quantile25 = np.quantile(metrics, 0.25, axis=1)
    quantile75 = np.quantile(metrics, 0.75, axis=1)
    min = np.min(metrics, axis = 1)
    max = np.max(metrics, axis=1)

    stattests = []
    for k in range(len(metrics_list)): # metric
        stattests_metric = []
        for i in range(len(names)): # model
            stattests_model =[]
            try:
                stattests_model.append(round(stats.shapiro(metrics[i, :, k].flatten())[-1], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                stattests_model.append(round(stats.shapiro(np.log(np.array(metrics[i, :, k].flatten() + 1e-100)))[-1], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                statdata = []
                for ii in range(np.shape(metrics)[0]):
                    statdata.append(np.array(metrics[ii, :, k].flatten()))
                stattests_model.append(round(stats.friedmanchisquare(*statdata)[-1], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                statdata = []
                for ii in range(np.shape(metrics)[0]):
                    col_metrics = np.array(metrics[ii, :, k].flatten())
                    col_patient = np.arange(0, len(col_metrics))
                    col_model = np.ones(np.shape(col_metrics)) * ii
                    stack = np.stack((col_patient, col_model, col_metrics))

                    if len(statdata) == 0:
                        statdata = stack
                    else:
                        statdata = np.hstack((statdata, stack))

                    #input.append(np.array(data[metrics[j], model_descriptions[l]].tolist()))
                #statistic = stats.f_oneway(*input)
                #pval = round(statistic[-1], 4)


                df = pd.DataFrame((statdata.T), columns=["patient", "model", "metric_value"])
                arm = AnovaRM(df, "metric_value", "patient", ["model"])
                armresult = arm.fit()
                stattests_model.append(round(armresult.anova_table["Pr > F"][0], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                statdata = []
                for ii in range(np.shape(metrics)[0]):
                    col_metrics = np.log(np.array(metrics[ii, :, k].flatten() + 1e-100))
                    col_patient = np.arange(0, len(col_metrics))
                    col_model = np.ones(np.shape(col_metrics)) * ii
                    stack = np.stack((col_patient, col_model, col_metrics))

                    if len(statdata) == 0:
                        statdata = stack
                    else:
                        statdata = np.hstack((statdata, stack))

                    #input.append(np.array(data[metrics[j], model_descriptions[l]].tolist()))
                #statistic = stats.f_oneway(*input)
                #pval = round(statistic[-1], 4)


                df = pd.DataFrame((statdata.T), columns=["patient", "model", "metric_value"])
                arm = AnovaRM(df, "metric_value", "patient", ["model"])
                armresult = arm.fit()
                stattests_model.append(round(armresult.anova_table["Pr > F"][0], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                stattests_model.append(round(stats.wilcoxon(metrics[reference_ID, :, k].flatten(), metrics[i, :, k].flatten(), alternative="two-sided")[-1], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                stattests_model.append(round(stats.ttest_rel(metrics[reference_ID, :, k].flatten(), metrics[i, :, k].flatten(), nan_policy="omit")[-1], 4))
            except:
                stattests_model.append(float("nan"))

            try:
                stattests_model.append(round(stats.ttest_rel(metrics[reference_ID, :, k].flatten(), metrics[i, :, k].flatten(), nan_policy="omit")[-1], 4))
            except:
                stattests_model.append(float("nan"))

            stattests_metric.append(stattests_model)
        stattests.append(stattests_metric)

    stattests = np.array(stattests)
    global_tests = ["shapiro", "shapiro log", "Friedman", "Annova RM", "Annova RM log", "Wilcoxon ref: " + str(reference_ID+1), "t test ref: " + str(reference_ID+1), "t test log ref: " + str(reference_ID+1)]

    # Values
    values = []

    for i in range(len(names)+1):
        values_model = []
        for j in range(len(data_copy.information)):

            if i >= len(names):
                indeces = np.argwhere(expectations[0][j])
            else:
                indeces = np.argwhere(predictions[i][j])

            if len(indeces) > 0:
                val = data_copy.x[j][indeces[:,0], indeces[:,1]]
                values_case = []
                values_case.append(np.min(val))
                values_case.append(np.max(val))
                values_case.append(np.mean(val))
                values_case.append(np.std(val))
                values_case.append(np.median(val))
                values_case.append(np.quantile(val, 0.25))
                values_case.append(np.quantile(val, 0.75))
            else:
                values_case = [0, 0, 0, 0, 0, 0, 0]

            values_model.append(values_case)
        values.append(values_model)

    global_statistics = ["average", "std", "median", "quantile25", "quantile75", "min", "max"]

    # WRITE TO OUTPUT
    if name == "":
        file_path = os.path.join(path_out, "MMS_test_unet_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(uuid.getnode()) + ".xlsx")
    else:
        file_path = os.path.join(path_out, name + "_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(uuid.getnode()) + ".xlsx")

    excel = tools.tool_excel.Setup(file_path)

    # WRITE METRICS
    excel.add_worksheet("Metrics")

    header1 = ["case"]
    for i in range(len(metrics_list)):
        for j in range(len(names)):
            header1.append(metrics_list[i])
    excel.write_header("Metrics", header1)

    header2 = [""]
    for i in range(len(metrics_list)):
        for j in range(len(names)):
            header2.append(names[j])
    excel.write_header("Metrics", header2)
    excel.set_freeze_panes("Metrics", 2, 1)

    # Metric values per case
    for i in range(len(data_copy.information)):
        row = [data_copy.information[i][1]]
        for j in range(len(metrics_list)):
            for k in range(len(names)):
                row.append(metrics[k][i][j])
        excel.write_line("Metrics", row)

    # global statistics
    for i in range(len(global_statistics)):
        row = [global_statistics[i]]
        for j in range(len(metrics_list)):
            for k in range(len(names)):
                row.append(round(eval(global_statistics[i] + "[k][j]"),2))
        excel.write_line("Metrics", row)

    # statistic tests
    for i in range(len(global_tests)):
        row = [global_tests[i]]
        for j in range(len(metrics_list)):
            for k in range(len(names)):
                row.append(stattests[j][k][i])
        excel.write_line("Metrics", row)

    # VALUES
    excel.add_worksheet("Values")
    values_list = ["min", "max", "mean", "std", "median", "25% percentile", "75% percentile"]

    header1 = ["case"]
    for i in range(len(values_list)):
        for j in range(len(names)+1):
            header1.append(values_list[i])
    excel.write_header("Values", header1)

    header2 = [""]
    for i in range(len(values_list)):
        for j in range(len(names)+1):
            if j >= len(names):
                header2.append("REFERENCE")
            else:
                header2.append(names[j])
    excel.write_header("Values", header2)
    excel.set_freeze_panes("Values", 2, 1)

    for i in range(len(data_copy.information)):
        row = [data_copy.information[i][1]]
        for j in range(len(values_list)):
            for k in range(len(names)+1):
                row.append(values[k][i][j])
        excel.write_line("Values", row)

    excel.save()

    return


def test_on_array(array, reference_ID=0):

    stattests = []
    for i in range(np.shape(array)[1]): # model
        stattests_model =[]
        try:
            stattests_model.append(round(stats.shapiro(array[:, i].flatten())[-1], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            stattests_model.append(round(stats.shapiro(np.log(np.array(array[:, i].flatten() + 1e-100)))[-1], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            statdata = []
            for ii in range(np.shape(array)[1]):
                statdata.append(np.array(array[:, ii].flatten()))
            stattests_model.append(round(stats.friedmanchisquare(*statdata)[-1], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            statdata = []
            for ii in range(np.shape(array)[1]):
                col_metrics = np.array(array[:, ii].flatten())
                col_patient = np.arange(0, len(col_metrics))
                col_model = np.ones(np.shape(col_metrics)) * ii
                stack = np.stack((col_patient, col_model, col_metrics))

                if len(statdata) == 0:
                    statdata = stack
                else:
                    statdata = np.hstack((statdata, stack))

                #input.append(np.array(data[metrics[j], model_descriptions[l]].tolist()))
            #statistic = stats.f_oneway(*input)
            #pval = round(statistic[-1], 4)


            df = pd.DataFrame((statdata.T), columns=["patient", "model", "metric_value"])
            arm = AnovaRM(df, "metric_value", "patient", ["model"])
            armresult = arm.fit()
            stattests_model.append(round(armresult.anova_table["Pr > F"][0], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            statdata = []
            for ii in range(np.shape(array)[1]):
                col_metrics = np.log(np.array(array[:, ii].flatten() + 1e-100))
                col_patient = np.arange(0, len(col_metrics))
                col_model = np.ones(np.shape(col_metrics)) * ii
                stack = np.stack((col_patient, col_model, col_metrics))

                if len(statdata) == 0:
                    statdata = stack
                else:
                    statdata = np.hstack((statdata, stack))

                #input.append(np.array(data[metrics[j], model_descriptions[l]].tolist()))
            #statistic = stats.f_oneway(*input)
            #pval = round(statistic[-1], 4)


            df = pd.DataFrame((statdata.T), columns=["patient", "model", "metric_value"])
            arm = AnovaRM(df, "metric_value", "patient", ["model"])
            armresult = arm.fit()
            stattests_model.append(round(armresult.anova_table["Pr > F"][0], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            stattests_model.append(round(stats.wilcoxon(array[:, reference_ID].flatten(), array[:, i].flatten(), alternative="two-sided")[-1], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            stattests_model.append(round(stats.ttest_rel(array[:, reference_ID].flatten(), array[:, i].flatten(), nan_policy="omit")[-1], 4))
        except:
            stattests_model.append(float("nan"))

        try:
            stattests_model.append(round(stats.ttest_rel(array[:, reference_ID].flatten(), array[:, i].flatten(), nan_policy="omit")[-1], 4))
        except:
            stattests_model.append(float("nan"))

        stattests.append(stattests_model)

    stattests = np.transpose(np.array(stattests))
    global_tests = ["shapiro", "shapiro log", "Friedman", "Annova RM", "Annova RM log", "Wilcoxon ref: " + str(reference_ID+1), "t test ref: " + str(reference_ID+1), "t test log ref: " + str(reference_ID+1)]

    values = []
    values.append(np.mean(array, axis=0))
    values.append(np.std(array, axis=0))
    values.append(np.median(array, axis=0))
    values.append(np.quantile(array, 0.25, axis=0))
    values.append(np.quantile(array, 0.75, axis=0))
    values.append(np.min(array, axis = 0))
    values.append(np.max(array, axis=0))
    values = np.array(values)

    global_statistics = ["average", "std", "median", "quantile25", "quantile75", "min", "max"]

    return np.vstack((np.squeeze(values), np.squeeze(stattests))), global_statistics+global_tests