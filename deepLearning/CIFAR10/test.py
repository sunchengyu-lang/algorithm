import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# img_path = "./images/dog.jpg"
img_path = "./images/horse.jpg"
image = Image.open(img_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)
print(image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, padding=2),
            MaxPool2d(2),
            Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("model_9.pth", map_location=device).to(device)
print(model)
image = image.to(device)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(labels[output.argmax(1)])
