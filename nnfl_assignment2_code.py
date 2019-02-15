import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.optim as optim
import matplotlib.pyplot as plt

class CNN(nn.Module):
	def __init__(self):
		super().__init__()

		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.ReLU1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.ReLU2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.bn2 = nn.BatchNorm2d(64)

		self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.ReLU3 = nn.ReLU()
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(4*4*64, 500)
		self.ReLU4 = nn.ReLU()
		self.drop = nn.Dropout(p=0.5)

		self.fc2 = nn.Linear(500, 10)

	def forward(self,x):
		# print(x.size())
		out = self.cnn1(x)
		# print(out.size())
		out = self.ReLU1(out)
		out = self.maxpool1(out)
		# print(out.size())
		out = self.cnn2(out)
		# print(out.size())
		out = self.ReLU2(out)
		out = self.maxpool2(out)
		# print(out.size())
		out = self.bn2(out)
		# print(out.size())
		out = self.cnn3(out)
		# print(out.size())
		out = self.ReLU3(out)
		out = self.maxpool3(out)
		# print(out.size())
		out = self.bn3(out)
		# print(out.size())
		out = out.view(out.size(0), -1)
		# print(out.size())
		out = self.fc1(out)
		# print(out.size())
		out = self.ReLU4(out)
		out = self.drop(out)
		out = self.fc2(out)
		# print(out.size())

		return out

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

model = CNN()
# model.load_state_dict(torch.load('mytraining.pt'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

n_epochs = 100
loss_list = []
accuracy_list = []

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=5000, shuffle=False)

N_test = len(validation_loader)
N_train = len(train_loader)

count0 = 1
for epoch in range(n_epochs):
	print ("Epoch", count0)
	count0+=1
	count = 1
	model.train()
	for x,y in train_loader:
		print("Train",count)
		count+=1
		optimizer.zero_grad()

		z = model(x)
		loss = criterion(z,y)
		loss.backward()
		optimizer.step()

	correct = 0
	count = 1
	model.eval()
	for x_test, y_test in validation_loader:
		print("Test",count)
		count+=1
		z = model(x_test)
		_, yhat = torch.max(z.data, 1)
		correct += (yhat==y_test).sum().item()

	accuracy = correct/N_test
	accuracy_list.append(accuracy)
	loss_list.append(loss.data)

torch.save(model.state_dict(), 'saved_model_state.pt')

plt.plot(accuracy_list, 'r', loss_list, 'b')
plt.xlabel("Epoch")
plt.legend()
plt.show()

def __init__(self,text_file,root_dir,transform=transformMnistm):
        """
            text_file: path to text file
            root_dir: directory with all train images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}

        return sample


mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt')
                                   root_dir = 'Downloads/mnist_m/mnist_m_train')

mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)
