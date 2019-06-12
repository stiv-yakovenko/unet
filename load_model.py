import time
import os
import image_slicer
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.transform import  resize


def inference_by_blocks(png_file,pb_folder, out_png_file):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "./models/roofs_not_roofs/")
        imm = Image.open(png_file)
        im = np.array(imm, np.float32)[None, ...] / 256
        tiles = image_slicer.slice(png_file, 100)
        new_im = Image.new('L', (im.shape[2], im.shape[1]))
        x = loaded_graph.get_tensor_by_name('x:0')
        y = loaded_graph.get_tensor_by_name('y:0')
        predictor = loaded_graph.get_tensor_by_name('results/pixel_wise_softmax/predicter:0')
        keep_prob = loaded_graph.get_tensor_by_name('dropout_probability:0')
        for tile in tiles:
            print(str(tile.number)+"/"+str(len(tiles)))
            f = np.array(tile.image)/256.
            inp = f[None, ...]
            y_dummy = np.empty((inp.shape[0], inp.shape[1], inp.shape[2], 2))
            res = sess.run(predictor, feed_dict={x: inp, keep_prob: 1.0, y: y_dummy})[0, ..., 1]
            # ri = Image.fromarray((res * 256).astype(np.uint8))
            upsize = (tile.image.size[1],tile.image.size[0])
            print(res.shape,"->",upsize)
            p1 = resize(res*255, upsize,anti_aliasing=True)
            result = Image.fromarray((p1).astype(np.uint8))
            new_im.paste(result, tile.coords)
    new_im.save(out_png_file)
os.environ["OMP_NUM_THREADS"]="1"
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
def inference_single():
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph,config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "./models/roofs_not_roofs/")
        x = loaded_graph.get_tensor_by_name('x:0')
        y = loaded_graph.get_tensor_by_name('y:0')
        predictor = loaded_graph.get_tensor_by_name('results/pixel_wise_softmax/predicter:0')
        keep_prob = loaded_graph.get_tensor_by_name('dropout_probability:0')
        img=Image.open("./test_pic/b2_DJI_0145_02_05.png").convert("RGB")
        f = np.array(img) / 256.
        inp = f[None, ...]
        y_dummy = np.empty((inp.shape[0], inp.shape[1], inp.shape[2], 2))
        res = sess.run(predictor, feed_dict={x: inp, keep_prob: 1.0, y: y_dummy})[0, ..., 1]
        result = Image.fromarray((res*256.).astype(np.uint8))
        result.save("res.png")
        print("saved")
inference_single()