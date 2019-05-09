from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/cracks/train','image','label',data_gen_args,save_to_dir = None,image_color_mode='rgb')
testGene = testGenerator("data/cracks/test",as_gray=False)

model = unet()
model_checkpoint = ModelCheckpoint('unet_cracks.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/cracks/test",results)