import torch
ckpt = torch.load('/u/zup7mn/Classes/NN/digit4/src/mnist_classifier/mnist_mixup_classifier.pth', map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif isinstance(ckpt, dict):
    state_dict = ckpt
else:
    state_dict = ckpt.state_dict()
for k, v in state_dict.items():
    print(k, v.shape)
