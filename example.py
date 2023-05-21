import torch
from model import AlexNet
"""
This creates a tensor with the same dimensions as
a batch of ImageNet images (i.e., 227x227 pixels and 3 color channels)
and runs this through the AlexNet model.
The output tensor,
which contains the model's predictions for each of the 1000 classes,
is then printed to the console.
"""
def main():
    model = AlexNet()

    # random tensor that mimics the size of an ImageNet image batch (10 batch, 3 channels, 227 height, 227 width)
    example_tensor = torch.randn(10, 3, 227, 227)

    # run the tensor
    output = model(example_tensor)
    print(output)

if __name__ == "__main__":
    main()