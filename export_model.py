import time
import os
import tensorflow as tf

trained_checkpoint_prefix = 'solar_predict/model.ckpt'
export_dir = os.path.join('models', time.strftime("%Y%m%d-%H%M%S"))
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Restore from checkpoint
    loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
        print(   i)

        # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    legacy_init_op = tf.group(tf.tables_initializer(),name="legacy_init_op");
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         strip_default_attrs=True,legacy_init_op=legacy_init_op)
builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
builder.save()