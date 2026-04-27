import torch
import torch.nn as nn

device = 'cpu'

classifier = nn.Sequential(
    nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.Flatten(),
    nn.Linear(64 * 676, 10),
    nn.Softmax(dim=1)
)

state_dict = torch.load("/u/zup7mn/Classes/NN/digit4/state/mnist_classifier/mnist_mixup_classifier.pth", map_location=device)
classifier.load_state_dict(state_dict)
print("Loaded successfully!")
