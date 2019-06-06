import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image

batch_size = 32
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "./models/20190607-024242")
    im = np.array(Image.open("./test_pic/b2_DJI_0145_02_05.png"), np.float32)[None,...]/256
    x = loaded_graph.get_tensor_by_name('x:0')
    y = loaded_graph.get_tensor_by_name('y:0')
    print([n.name for n in loaded_graph.as_graph_def().node])
    predictor = loaded_graph.get_tensor_by_name('results/pixel_wise_softmax/predicter:0')
    keep_prob = loaded_graph.get_tensor_by_name('dropout_probability:0')
    print("imshape",im.shape)
    y_dummy = np.empty((im.shape[0], im.shape[1], im.shape[2], 2))
    res = sess.run(predictor,feed_dict= {x: im,keep_prob: 1.0,y:y_dummy})[0, ..., 1]
    print(res.shape)
    ri=Image.fromarray((res*256).astype(np.uint8))
    ri.save("pred.png");
