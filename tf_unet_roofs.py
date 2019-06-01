from tf_unet import unet, util, image_util
import matplotlib.pyplot as plt
#preparing data loading
data_provider = image_util.ImageDataProvider("./data/roof_not_roof/*.png",data_suffix=".png",mask_suffix="_mask.png")
output_path = "./solar_predict/"
#setup & training
net = unet.Unet(layers=3, features_root=256, channels=3, n_class=2,cost='bce_dice_coefficient')#
trainer = unet.Trainer(net,verification_batch_size=32,batch_size=1)#
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)

