import logging
import os

import imageio

logger = logging.getLogger(__name__)
path = os.getenv('T1DPATH', '../')


def getName(data):
    start = data.date.values[0]
    end = data.date.values[len(data.date.values) - 1]
    return "animated-" + start + "-" + end


def makeGif(directory, data):
    # create gif from files ins(directory
    logger.info("make gif from plots in " + directory)
    imgs = []
    for filename in os.listdir(directory):
        logger.info(directory + filename)
    imgs.append(imageio.imread(directory + filename))

    name = getName(data)
    if not os.path.exists(path + "results/gifs/"):
        os.makedirs(path + "results/gifs/")

    imageio.mimsave(path + 'results/gifs/' + name + '.gif', imgs, loop = 0, fps = 5, subrectangles = True)
