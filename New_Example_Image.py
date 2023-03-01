import rasterio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.gridspec as gridspec
BT474_Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/BT474/All/Ch1/7.tif'  # path to the channel 1 images
HMBEC_Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/HMBEC_V2/All/Ch1/7.tif'  # path to the channel 1 images

transform = transforms.Compose([
    transforms.ToPILImage(),
transforms.Resize((256, 256))])


plt.figure(2,figsize=(5,5))

plt.subplot(1,2,1)
image = rasterio.open(BT474_Ch1).read().squeeze(0)
im = transform(image)
plt.title('BT474',fontsize=16)
plt.imshow(im, cmap='gray')
plt.ylabel('Ch1')
plt.axis('off')

plt.subplot(1,2,2)
image = rasterio.open(HMBEC_Ch1).read().squeeze(0)
im = transform(image)
plt.title('HMBEC')
plt.imshow(im, cmap='gray')
# plt.show()
plt.axis(False)

plt.show()
