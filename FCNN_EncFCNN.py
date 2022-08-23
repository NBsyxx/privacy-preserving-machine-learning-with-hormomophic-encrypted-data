import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time

torch.manual_seed(73)

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        
        super(FullyConnectedNet, self).__init__()  
        self.fc1 = torch.nn.Linear(784, 1024)
        self.fc2 = torch.nn.Linear(1024, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = x.view(-1, 784)
        # flattening while keeping the batch axis
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        return x


def train(model, train_loader, criterion, optimizer, n_epochs=10):
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


model = FullyConnectedNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


start = time.time()
model = train(model, train_loader, criterion, optimizer, 10)
end = time.time()

print("model training takes",end - start,"s")


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


import tenseal as ts


class EncFullyConnectedNet:
    def __init__(self, torch_nn):
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()
        
        
    def forward(self, enc_x):
        # conv layer
#         enc_channels = []
#         for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
#             y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
#             enc_channels.append(y)
#         # pack all channels into a single flattened vector
#         enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
#         enc_x.square_()
        # fc1 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        end = time.time()
        print("fc1 takes", end - start)
        
        # square activation
        start = time.time()
        enc_x.square_()
        end = time.time()
        print("square activation takes", end - start)
        
        # fc2 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        end = time.time()
        print("fc2 takes", end - start)
        
        start = time.time()
        enc_x.square_()
        end = time.time()
        print("square activation takes", end - start)
        
        # fc3 layer
        start = time.time()
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        end = time.time()
        print("fc3 takes", end - start)
        
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    
def enc_test(context, model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    cnt = 0
    for data, target in test_loader:
        
#         Encoding and encryption
        print(data.view(-1,784).shape)
        x_enc = ts.ckks_vector(context, data.view(-1,784)[0])
        # Encrypted evaluation
        enc_output = enc_model(x_enc)
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
        if cnt == 50:
            break


    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    print(class_correct)
    print(class_total)




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


enc_model = EncFullyConnectedNet(model)
start = time.time()
enc_test(context, enc_model, test_loader, criterion)
end = time.time()
print("model testing takes",end - start,"s")