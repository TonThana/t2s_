import copy
import matplotlib.cm as cm
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def factors(n):
    return tuple(reduce(list.__add__,
                        ([i, n//i] for i in range(1, int(n**0.5) + 1)
                         if n % i == 0)))


def closest_to_mul_of_n(x, base=5):
    new = int(base*np.ceil(x/base))
    diff = new - x
    return (new, diff)


def overlay_helper(imageData=None, roiData=None, title=None):
    # rois will be treated as mask to extract value from cons
    if roiData is None:
        roiData = np.zeros(shape=imageData.shape[:3])
    (x, y, z) = imageData.shape
    (new, diff) = closest_to_mul_of_n(z, 5)
    # diff for number of only 0s 2d image
    empty_image = np.zeros(shape=(x, y))
    for i in range(diff):
        roiData = np.dstack((roiData, empty_image))
        imageData = np.dstack((imageData, empty_image))
    print(roiData.shape, imageData.shape)

    factors_list = factors(new)
    rowNum = factors_list[-1]
    colNum = factors_list[-2]
    print("[INFO] numRow: {}, numCol {}".format(rowNum, colNum))
    fig, axs = plt.subplots(nrows=rowNum, ncols=colNum,
                            figsize=(colNum, rowNum), facecolor=(0, 0, 0),
                            num=title)
    my_cmap = copy.copy(cm.get_cmap("spring"))
    my_cmap.set_under('w', alpha=0)
    my_cmap.set_bad('w', alpha=0)
    imageMin = np.min(imageData)
    imageMax = np.max(imageData)
    for i in range(int(rowNum*colNum)):
        row = int(np.floor(i/colNum))
        col = int(i % colNum)
        ax = axs[row, col]
        ax.imshow(np.rot90(imageData[:, :, i]),
                  cmap='gray', interpolation='nearest', vmin=imageMin, vmax=imageMax)
        ax.imshow(np.rot90(roiData[:, :, i]), cmap=my_cmap,
                  vmin=0.5, vmax=1, alpha=0.5,
                  interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')

    fig.tight_layout()
    plt.show()
