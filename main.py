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
data_gen_args = dict()
myGene = trainGenerator(1,'data/'+dataset+'/train','image','label',data_gen_args,save_to_dir = None,image_color_mode='rgb')
# while True:
#     x,y = myGene.__next__()
#     for i in range(0,1):
#         image = y[i]
#         pic = image.transpose(2,1,0)
#         plt.imshow(pic.transpose())
#         plt.show()
testGene = testGenerator("data/"+dataset+"/test",as_gray=False)
#pretrained_weights='unet_'+dataset+'.hdf5'
model = unet()
model_checkpoint = ModelCheckpoint('unet_'+dataset+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
class onEpoch(Callback):
    def on_epoch_end(self, batch, logs=None):
        print("saving predict data")
        testGene = testGenerator("data/" + dataset + "/test", as_gray=False)
        results = model.predict_generator(testGene,10,verbose=1)
        saveResult("data/"+dataset+"/predict",results)

clr = CyclicLR(base_lr=0.00001, max_lr=0.0006,
                        step_size=280.)
hist=model.fit_generator(myGene,steps_per_epoch=40,
                         epochs=858,callbacks=[model_checkpoint,csv_logger,clr,onEpoch()],
                         verbose=2,shuffle=True)

results = model.predict_generator(testGene,10,verbose=1)
saveResult("data/"+dataset+"/predict",results)