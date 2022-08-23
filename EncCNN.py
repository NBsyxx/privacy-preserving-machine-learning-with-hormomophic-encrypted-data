import tenseal as ts
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time



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
        start = time.time()
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        end = time.time()
        print("cov2d takes", end - start)
        
        
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        end = time.time()
        print("fc1 takes", end - start)
        # square activation
        enc_x.square_()
        # fc2 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        end = time.time()
        print("fc2 takes", end - start)
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    cnt = 0
    for data, target in test_loader:
        
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = enc_model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
        cnt += 1
        if cnt == 100:
            break


    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    print(class_correct)
    print(class_total)



if __name__ == "__main__":

    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())


    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)




    def train(model, train_loader, criterion, optimizer, n_epochs=10):
        # model in training mode
        model.train()
        for epoch in range(1, n_epochs+1):

            train_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # calculate average losses
            train_loss = train_loss / len(train_loader)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        # model in evaluation mode
        model.eval()
        return model


    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    start = time.time()
    model = train(model, train_loader, criterion, optimizer, 10)
    end = time.time()


    def test(model, test_loader, criterion):
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        # model in evaluation mode
        model.eval()

        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            

        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader)
        print(f'Test Loss: {test_loss:.6f}\n')

        for label in range(10):
            print(
                f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
                f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
            )

        print(
            f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
            f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
        )

    start = time.time()
    test(model, test_loader, criterion)
    end = time.time()

    print("model testing takes",end - start,"s")

    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    # required for encoding
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]


    ## Encryption Parameters

    # controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()


    # enc_model = EncConvNet(model)
    enc_model = EncConvNet(model)
    start = time.time()
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
    end = time.time()
    print("model testing takes",end - start,"s")