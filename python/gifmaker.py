import os

import imageio
import logging

logger = logging.getLogger(__name__)


def getName(data):
    start = data.date.values[0]
    end = data.date.values[len(data.date.values) -1 ]
    return "animated-" + start + "-" + end

def makeGif(path, data):
    # create gif from files ins path
    logger.info("make gif from plots in " + path)
    imgs = []
    for filename in os.listdir(path):
        logger.info(path + filename)
        imgs.append(imageio.imread(path + filename))

    name = getName(data)
    if not os.path.exists("/t1d/results/gifs/"):
        os.makedirs("/t1d/results/gifs/")

    imageio.mimsave('/t1d/results/gifs/' + name + '.gif', imgs, loop=0, fps=5, subrectangles=True)
