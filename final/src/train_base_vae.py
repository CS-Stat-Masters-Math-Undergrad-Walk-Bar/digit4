import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import io
from PIL import Image

from project_paths import DATA_ROOT, BASE_VAE_PATH

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

mnist_train = torchvision.datasets.MNIST(root=DATA_ROOT, train = True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root=DATA_ROOT, train = False, download=True, transform=transforms.ToTensor())

BATCH_SIZE = 256
GRID_SIZE = 8  # Number of images per row/column in the visualization grid

train_loader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size= BATCH_SIZE,
)
test_loader = torch.utils.data.DataLoader(
    mnist_test,
    batch_size= BATCH_SIZE,
)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.first_hidden = nn.Linear(input_dim, hidden_dim)
        self.second_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        y_ = self.flatten(x)
        y_ = self.first_hidden(y_)
        y = self.activation(y_)
        y_ = self.second_hidden(y)
        y = self.activation(y_)
        mean = self.mean(y)
        log_var = self.var(y)
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.first_hidden = nn.Linear(latent_dim, hidden_dim)
        self.second_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y_ = self.first_hidden(x)
        y_ = self.activation(y_)
        y_ = self.second_hidden(y_)
        y_ = self.activation(y_)
        y_ = self.output(y_)
        y = self.sigmoid(y_)
        return y
    
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + epsilon * var
        return z
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(log_var / 2))
        reconstruction = self.Decoder(z)
        return reconstruction, mean, log_var
    
def KL_latent_loss(mean, log_var):
    # Sum over latent dims, mean over batch — matches BCELoss(reduction="mean")
    return -0.5 * torch.mean(torch.sum(1 + log_var - torch.exp(log_var) - mean**2, dim=1))

EPOCHS = 25

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 300
LATENT_DIM = 10

encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_SIZE, INPUT_SIZE)
model = VAE(encoder, decoder).to(device)


optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "name": "encoder"},
        {"params": decoder.parameters(), "name": "decoder"}
    ],
    lr = 0.001,
    weight_decay = 1e-5)
reconstr_loss = torch.nn.BCELoss(reduction="mean")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

lrs = []
test_losses = []
frames = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        prediction, mean, log_var = model(images)
        kl_loss = KL_latent_loss(mean, log_var) 
        prediction = prediction.reshape(images.size(0), 1, 28, 28)
        recon_loss = reconstr_loss(prediction, images)
        loss = recon_loss + kl_loss

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val, current = loss.item(), batch * BATCH_SIZE + len(images)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")

    # Step scheduler once per epoch
    scheduler.step(epoch_loss / len(train_loader))

    # save model

    torch.save(model.state_dict(), BASE_VAE_PATH)


    with torch.no_grad():
        test_loss = 0
        model.eval()
        for images, labels in test_loader:
            images = images.to(device) 
            prediction, mean, log_var = model(images)
            kl_loss = KL_latent_loss(mean, log_var) 
            prediction = prediction.reshape(images.size(0), 1, 28, 28)
            recon_loss = reconstr_loss(prediction, images)
            loss = recon_loss + kl_loss 
            
            test_loss += loss.item()
            

        test_loss /= len(test_loader)

        test_losses.append(test_loss)

        # clear previous output
        clear_output(wait=True)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Test Error: \n Avg loss: {test_loss:>8f}")

        # Visualize input vs reconstructed images
        sample_images, _ = next(iter(test_loader))
        sample_images = sample_images[:GRID_SIZE * GRID_SIZE]
        sample_images = sample_images.to(device) 
        reconstructed, _, _ = model(sample_images)
        reconstructed = reconstructed.view(-1, 28, 28) * 255.0
        
        input_images = sample_images.squeeze(1)
        
        fig, axes = plt.subplots(2, GRID_SIZE, figsize=(GRID_SIZE * 2, 4))
        
        for i in range(GRID_SIZE):
            axes[0, i].imshow(input_images[i].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Input', fontsize=10)
        
        for i in range(GRID_SIZE):
            axes[1, i].imshow(reconstructed[i].cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        fig.suptitle(f'Epoch {epoch+1}/{EPOCHS}', fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

        # plt.show()

# Save GIF
if frames:
    frames[0].save(
        VAE_OUT_DIR / 'reconstruction_progress.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print("Saved: reconstruction_progress.gif")