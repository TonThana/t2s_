import sys
import argparse
import os
import glob
import json
import numpy as np
import nibabel as nib


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


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not valid path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--datapath",
                    help="path to data folder", required=True, type=dir_path)
    args = vars(ap.parse_args())
    datapath = args["datapath"]
    echoTimes = load_echo_times(datapath=datapath)
    print(echoTimes)
    t2s_4d = load_images(datapath=datapath)
    print(t2s_4d.shape)
    minimumFloat = sys.float_info.min
    zerosAndBelowMask = t2s_4d <= 0
    t2s_4d[zerosAndBelowMask] = minimumFloat
    print(np.min(t2s_4d))


if __name__ == "__main__":
    main()
