import torch
from torchvision import transforms, models


class DigitClassifier(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(DigitClassifier, self).__init__()
        self.device = device
        self.model = models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = torch.nn.Linear(512, 1)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1736, std=0.3317),
        ])
        self.model.load_state_dict(torch.load("mnist_classifier/best_model.pth", map_location=torch.device('cpu')))
        self.model.to(device)

    def forward(self, x):
        x = self.transforms(x.to(self.device)).unsqueeze(0)  # Add batch dimension
        return self.model(x)
