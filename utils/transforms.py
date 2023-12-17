import random
import math
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target


# class RandomErasing:
#     def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
#         self.EPSILON = EPSILON
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#
#     def __call__(self, img, target):
#
#         if random.uniform(0, 1) > self.EPSILON:
#             return img,target
#
#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]
#
#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1 / self.r1)
#
#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))
#
#             if w < img.size()[2] and h < img.size()[1]:
#                 x1 = random.randint(0, img.size()[1] - h)
#                 y1 = random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     # img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
#                     # img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
#                     # img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
#                     # img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
#                 else:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
#                 return img,target
#
#         return img,target




# class RandomContrast:
#     def __init__(self, lower=0.5, upper=1.5):
#         self.lower = lower
#         self.upper = upper
#
#     def __call__(self, image, boxes):
#         # if random.randint(2):
#         #     # 生成随机因子
#         alpha = random.uniform(self.lower, self.upper)
#         image *= alpha
#         return image, boxes

#颜色变化

# # VAE model
# class VAE(nn.Module):
#     def __init__(self, image_size=784, h_dim=400, z_dim=20):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(image_size, h_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)  # 均值 向量
#         self.fc3 = nn.Linear(h_dim, z_dim)  # 保准方差 向量
#         self.fc4 = nn.Linear(z_dim, h_dim)
#         self.fc5 = nn.Linear(h_dim, image_size)
#
#     # 编码过程
#     def encode(self, x):
#         print("1:" + str(x.shape))
#         h = F.relu(self.fc1(x))
#         print("2:" + str(h.shape))
#         return self.fc2(h), self.fc3(h)
#
#     # 随机生成隐含向量
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(log_var / 2)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     # 解码过程
#     def decode(self, z):
#         h = F.relu(self.fc4(z))
#         return F.sigmoid(self.fc5(h))
#
#     # 整个前向传播过程：编码-》解码
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         print("3:" + str(mu.shape))
#         print("4:" + str(log_var.shape))
#         z = self.reparameterize(mu, log_var)
#         print("5:" + str(z.shape))
#         x_reconst = self.decode(z)
#         print("6:" + str(x_reconst.shape))
#         return x_reconst, mu, log_var





def build_transforms(is_train):
    transforms = []
    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
        # transforms.append(RandomErasing())
        # transforms.append(RandomContrast())
    return Compose(transforms)
