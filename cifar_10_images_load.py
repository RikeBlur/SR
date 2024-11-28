import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

batch_size = 100
# Cifar10 dataset                    #选择数据的根目录   #选择训练集    #从网上下载图片
train_dataset = dsets.CIFAR10(root='../datasets', train=True, download=True)
# 选择数据的根目录   #选择训练集    #从网上下载图片
test_dataset = dsets.CIFAR10(root='../datasets', train=False, download=True)

# 加载数据
# 将数据打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
digit = train_loader.dataset.data[5900]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(classes[train_loader.dataset.targets[0]])
toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值

for i in range(1,6000):
    tsr = torch.from_numpy(train_dataset.data[i])
    tsr_p = tsr.permute(2, 0, 1)
    pic = toPIL(tsr_p)
    pic.save(f'datasets/image_{i}.jpg')
