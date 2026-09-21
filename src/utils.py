import torch
import torchvision.transforms.v2 as v2
import torchvision.datasets as dset

# Directory containing the data.
root = '../dataset/Train'
def get_EMNIST(params):

    # Compose the transformations to be applied to the images.
    # Transformation: Grayscale -> Resize to (32, 32) -> Convert to Image -> Convert to Tensor with dtype float32 -> Normalize
    transform = v2.Compose([
        v2.Grayscale(),
        v2.Resize((32, 32)),            
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader

