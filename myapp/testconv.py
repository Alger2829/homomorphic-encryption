import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Define the EncConvNet class again
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()


    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# Load model function
def load_model(filename, torch_nn):
    with open(filename, 'rb') as f:
        model_params = pickle.load(f)

    model = EncConvNet(torch_nn)
    model.conv1_weight = model_params['conv1_weight']
    model.conv1_bias = model_params['conv1_bias']
    model.fc1_weight = model_params['fc1_weight']
    model.fc1_bias = model_params['fc1_bias']
    model.fc2_weight = model_params['fc2_weight']
    model.fc2_bias = model_params['fc2_bias']

    return model

# Predict function
def predict(context, model, image, kernel_shape, stride):
    # Encoding and encryption
    x_enc, windows_nb = ts.im2col_encoding(
            context, image.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )

    # Encrypted evaluation
    enc_output = model(x_enc, windows_nb)
    print(f'Encrypted image: {enc_output}')
    # Decryption of result
    output = enc_output.decrypt()
    output = torch.tensor(output).view(1, -1)

    # Obtain predicted label
    _, pred = torch.max(output, 1)
    return pred.item()

def load_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    transform = transforms.ToTensor()  # Transform to tensor
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def get_title(label):
    switcher = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return switcher.get(label, "Invalid label")


# Main function
def main():
    # Load MNIST dataset
    # test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor())

    # Define the SimpleNN torch model again (same architecture as used in training)
    class ConvNet(torch.nn.Module):
        def __init__(self, hidden=64, output=10):
            super(ConvNet, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
            self.fc1 = torch.nn.Linear(256, hidden)
            self.fc2 = torch.nn.Linear(hidden, output)

        def forward(self, x):
            x = self.conv1(x)
            # the model uses the square activation function
            x = x * x
            # flattening while keeping the batch axis
            x = x.view(-1, 256)
            x = self.fc1(x)
            x = x * x
            x = self.fc2(x)
            return x


    model = ConvNet()

    # Load the saved encrypted model
    enc_model = load_model('enc_convnet_model.pkl', model)

    # Encryption parameters
    bits_scale = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()


   
    
    # This is for getting mnist images, this won't be in final code 
    # random_image, random_label = test_data[421]
    # torch.save(random_image, 'mnist_image.pt')

    #send path to input/uploaded file as parameter
    loaded_image = torch.load('input_conv/ankleboot.pt')
    loaded_image_array = loaded_image.numpy().squeeze()

    # Display the original image; a
    plt.figure(figsize=(4, 4))
    plt.imshow(loaded_image_array, cmap='gray')
    plt.axis('off')
    # plt.show()
    plt.savefig('output_image.png')  # Save the image to a file
    plt.close()

    

    # Predict the label using the encrypted model
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]
    predicted_label = predict(context, enc_model, loaded_image, kernel_shape, stride)

    
    print(f'The given item is: {get_title(predicted_label)}')
    


if __name__ == '__main__':
    main()
