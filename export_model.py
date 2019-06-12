import time
import os
import tensorflow as tf

def export_model(model):
    trained_checkpoint_prefix = 'checkpoints/%s/model.ckpt'%model
    export_dir = './models/%s/'%model
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        legacy_init_op = tf.group(tf.tables_initializer(),name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         strip_default_attrs=True,legacy_init_op=legacy_init_op)
    builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
    builder.save()