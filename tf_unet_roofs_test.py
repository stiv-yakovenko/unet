import image_slicer
import time

from tf_unet import unet, util, image_util
import numpy as np
import tensorflow as tf
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image

def inference(path,ckpt):
    # setup & training
    net = unet.Unet(layers=2, features_root=256, channels=3, n_class=2, cost='bce_dice_coefficient')
    im = np.array(Image.open(path), np.float32)
    tiles = image_slicer.slice(path, 20)
    new_im = Image.new('L', (im.shape[1],im.shape[0]))
    c = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        net.restore(sess, ckpt)
        print("after restore")
        start = time.time()
        for tile in tiles:
            f = np.array(tile.image)/256.
            inp = f[None, ...]
            y_dummy = np.empty((inp.shape[0], inp.shape[1], inp.shape[2], net.n_class))
            print("before run")
            prediction = sess.run(net.predicter, feed_dict={net.x: inp, net.y: y_dummy, net.keep_prob: 1.})[0, ..., 1]
            print("after run")
            p1 = resize(prediction*255, (tile.image.size[1],tile.image.size[0]),anti_aliasing=True)
            result = Image.fromarray((p1).astype(np.uint8))
            result.save('out' + str(c) + '.png')
            c += 1
            new_im.paste(result, tile.coords)
        end = time.time()
        print(end-start)
    new_im.save('out.png')

path = "C:/Users/steve/Project/unet/data/roof_big_pics/b2_DJI_0145.JPG"
ckpt = "./checkpoints/roofs_not_roofs/model.ckpt"
inference(path,ckpt)
