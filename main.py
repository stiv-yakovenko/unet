import sys

from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
myGene = trainGenerator(2,'data/'+dataset+'/train','image','label',data_gen_args,save_to_dir = None,image_color_mode='rgb')
testGene = testGenerator("data/"+dataset+"/test",as_gray=False)

model = unet()
model_checkpoint = ModelCheckpoint('unet_'+dataset+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=900,epochs=12,callbacks=[model_checkpoint])

results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/"+dataset+"/test",results)