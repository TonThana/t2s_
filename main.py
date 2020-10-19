import sys
import argparse
import os
import glob
import json
import numpy as np
import numpy.ma as ma
import nibabel as nib
import multiprocessing
import time
import math
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from overlayhelper import overlay_helper

X_DIM = 128
Y_DIM = 128
Z_DIM = 28
T_DIM = 20


def load_echo_times(datapath):
    echoTimes = []
    jsons_list = []
    for json_file in glob.iglob("{datapath}/*.json".format(datapath=datapath)):
        jsons_list.append(json_file)
    jsons_list.sort()
    for json_file in jsons_list:
        with open(json_file) as j:
            data = json.load(j)
            echoTimes.append(data["EchoTime"])

    return np.array(echoTimes)


def load_images(datapath):
    imagePaths = []
    for image in glob.iglob("{}/realign*.nii".format(datapath)):
        imagePaths.append(image)
    imagePaths.sort()
    t2s_4d = np.zeros(shape=(X_DIM, Y_DIM, Z_DIM, T_DIM))
    for (index, imPath) in enumerate(imagePaths):
        data = nib.load(imPath)
        pixData = data.get_fdata()
        t2s_4d[:, :, :, index] = pixData
    return t2s_4d


def load_brain_mask(datapath):
    brainmaskpath = [x for x in glob.iglob(
        "{}/*brain_mask.nii.gz".format(datapath))]
    assert len(brainmaskpath) == 1
    # load
    bm = nib.load(brainmaskpath[0])
    bm = bm.get_fdata()
    return bm


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not valid path")


def calc_determination(arguments):
    fitparam, actualdata, echoTimes = arguments
    # print(fitparam)
    # if fitparam[0] == 0:
    #     return 0
    p = np.poly1d(fitparam)
    yhat = p(echoTimes)

    return r2_score(actualdata, yhat)


def t2s_parameter_estimation(data, echoTimes):
    # if less than 0 -> set to zero
    zerosAndBelowMask = data < 0
    data[zerosAndBelowMask] = 0

    # shift +1
    data = data + 1
    assert np.min(data) == 1
    logData = np.log(data)
    logData = np.reshape(logData, (X_DIM * Y_DIM * Z_DIM, T_DIM))
    logData = np.transpose(logData)
    assert logData.shape == (T_DIM, X_DIM * Y_DIM * Z_DIM)

    # least-square fit (degree 1)
    fitResult = np.polyfit(echoTimes, logData, 1)
    print("SLOPE max,min ", np.max(
        fitResult[0, :]), ", ", np.min(fitResult[0, :]))
    slopes = fitResult[0, :]
    # print(slopes.shape)

    # determination - R^2
    # parallel ?
    start = time.time()
    determination_result = np.zeros((fitResult.shape[1]))
    cpu_count = multiprocessing.cpu_count()
    print(fitResult.shape, logData.shape)
    with multiprocessing.Pool(processes=cpu_count) as p:
        determination_result[:] = p.map(calc_determination, ((
            fitResult[:, i], logData[:, i], echoTimes) for i in range(fitResult.shape[1])))
    print("Time taken to calculate R^2: ", time.time() - start)
    print("Determination: max: {}, min: {}".format(
        np.max(determination_result), np.min(determination_result)))

    determination_result = np.reshape(
        determination_result, (X_DIM, Y_DIM, Z_DIM))
    # mult by -1 to reverse sign
    slopes = np.reshape(slopes, (X_DIM, Y_DIM, Z_DIM)) * -1
    determination_mask = (determination_result <= 0.90).astype("uint8")
    # slopes = slopes * determination_mask
    return (slopes, determination_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--datapath",
                    help="path to data folder", required=True, type=dir_path)
    args = vars(ap.parse_args())
    datapath = args["datapath"]
    echoTimes = load_echo_times(datapath=datapath)
    print("[INFO] ", echoTimes)

    t2s_4d = load_images(datapath=datapath)
    slopes, determination_mask = t2s_parameter_estimation(
        data=t2s_4d, echoTimes=echoTimes)

    # overlay_helper(
    #     imageData=slopes, title="before brain mask overlay", roiData=determination_mask)
    # slopes => 1/T2s

    # USE BRAINMASK
    bm = load_brain_mask(datapath)
    print("mask info: shape: {}, min: {}, max: {}".format(
        bm.shape, np.min(bm), np.max(bm)))
    # intersect slope and mask
    slopes = slopes * bm

    print("[INFO] after brainmask, max: {}, min: {} ".format(
        np.max(slopes), np.min(slopes)))
    # overlay_helper(
    #    imageData=slopes, title="after brain mask overlay")

    # save
    nib.save(slopes, os.path.join(datapath, 'R2S.img'))


# def power_spectrum_analysis(data, mask):
#     # data normalisation


if __name__ == "__main__":
    main()
