#%%
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


#%%
# Apply transformations to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Update dataset to use the transform
dataset = MNIST(root='./data', train=True, download=True, transform=transform)

toPIL = transforms.ToPILImage()

# display 10 images from the dataset
fig, axes = plt.subplots(1, 10, figsize=(15, 1))
for i in range(10):
    image, label = dataset[i]
    axes[i].imshow(toPIL(image), cmap='gray')
    axes[i].axis('off')
plt.show()

# Create a simple DataLoader from the MNIST dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of iterating through the DataLoader
for images, labels in dataloader:
    print(images.shape, labels.shape)
    break


#%%

