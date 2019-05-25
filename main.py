from keras.callbacks import *
import sys
import numpy
import matplotlib.pyplot as plt

from clr_callback import CyclicLR
from model import *
from data import *
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log.csv', append=True, separator=';')
if (len(sys.argv)<2):
    print("set dataset folder")
    sys.exit(0)
dataset = sys.argv[1]
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/'+dataset+'/train','image','label',data_gen_args,save_to_dir = None,image_color_mode='rgb')
# while True:
#     x,y = myGene.__next__()
#     for i in range(0,1):
#         image = y[i]
#         pic = image.transpose(2,1,0)
#         plt.imshow(pic.transpose())
#         plt.show()
class onEpoch(Callback):
    def __init__(self):
        super(onEpoch, self).__init__()
    def on_epoch_end(self, batch, logs=None):
        print("saving predict data")
        results = model.predict_generator(testGene, 10, verbose=1)
        saveResult("data/" + dataset + "/predict", results)

testGene = testGenerator("data/"+dataset+"/test",as_gray=False)
#pretrained_weights='unet_'+dataset+'.hdf5'
model = unet()
model_checkpoint = ModelCheckpoint('unet_'+dataset+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
clr = CyclicLR(base_lr=0.000005, max_lr=0.00009,
                        step_size=100.)
hist=model.fit_generator(myGene,steps_per_epoch=300,
                         epochs=58,callbacks=[model_checkpoint,csv_logger,clr],
                         verbose=2,shuffle=True)

results = model.predict_generator(testGene,10,verbose=1)
saveResult("data/"+dataset+"/predict",results)