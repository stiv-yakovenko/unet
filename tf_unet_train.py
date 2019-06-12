from export_model import export_model
from tf_unet import unet, image_util
import optparse

parser = optparse.OptionParser()
parser.add_option('-d', dest="data", help="data folder")
parser.add_option('-m', dest="model", help="model name")

options, args = parser.parse_args()
if (not options.model or not options.data):
    parser.print_help()
    exit()

data_provider = image_util.ImageDataProvider("./data/%s/*.png"%options.data,data_suffix=".png",mask_suffix="_mask.png")
output_path = "./checkpoints/%s/"%options.model
net = unet.Unet(layers=3, features_root=256, channels=3, n_class=2,cost='bce_dice_coefficient')#
trainer = unet.Trainer(net,verification_batch_size=64,batch_size=1)#
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)
export_model(options.model)
