from tf_unet import unet, util, image_util
import matplotlib.pyplot as plt
#preparing data loading
data_provider = image_util.ImageDataProvider("./data/solar_tf_unet/*.png",data_suffix=".png",mask_suffix="_mask.png")
output_path = "./solar_predict/"
#setup & training
net = unet.Unet(layers=2, features_root=64, channels=1, n_class=2,cost='dice_coefficient')
trainer = unet.Trainer(net,verification_batch_size=16)#,batch_size=4
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)

x_test, y_test = test_data_provider(1)
prediction = net.predict(path, x_test)
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
ax[2].imshow(prediction[0,...,1], aspect="auto")
#
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
#
# img = util.combine_img_prediction(data, label, prediction)
# util.save_image(img, "prediction.jpg")