import torch
import torch.nn as nn
from torchvision import transforms, models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## Load the discriminator model

# discriminator = nn.Sequential(
#     nn.Conv2d(1, 64, (3,3), (2,2), padding=1), ## 14 x 14
#     nn.BatchNorm2d(64),
#     nn.LeakyReLU(0.02),
#     nn.Conv2d(64, 64, (3,3), (2,2), padding=1), ## 7 x 7
#     nn.BatchNorm2d(64),
#     nn.LeakyReLU(0.02),

#     nn.Flatten(),
#     nn.Linear(3136, 1),
#     nn.Sigmoid()
# ).to(device)
# discriminator.load_state_dict(torch.load('/u/zup7mn/Classes/NN/digit4/src/models/discriminator_last.pth'))
# discriminator.eval()

digit_classifier = nn.Sequential(
    nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.Flatten(),
    nn.Linear(64 * 676, 10),
    nn.Softmax(dim=1)
).to(device)



### Load classifier
CLASSIFIER_PATH = "/u/zup7mn/Classes/NN/digit4/state/mnist_classifier/mnist_mixup_classifier.pth"
digit_classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
digit_classifier.eval()


def novelty(images: torch.Tensor):
    with torch.no_grad():
        predictions = digit_classifier(images).squeeze().cpu().numpy()
        two_preds = predictions[:, 2]
        six_preds = predictions[:, 6]
    return two_preds * six_preds

