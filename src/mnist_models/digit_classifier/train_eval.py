import torch

from src.mnist_models.digit_classifier import mnist_classifier, emnist_classifier


device = "cuda:0"
print(f"Using {device} device")

print("\n=== Training MNIST classifier with mixup ===")
mnist_model, mnist_test_loader = mnist_classifier.train(
    epochs=10, use_mixup=True, device=device
)

print("\n=== Training EMNIST classifier with mixup ===")
emnist_model, emnist_test_loader = emnist_classifier.train(
    epochs=20, use_mixup=True, device=device
)


def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


mnist_on_mnist = compute_accuracy(mnist_model, mnist_test_loader)
mnist_on_emnist = compute_accuracy(mnist_model, emnist_test_loader)
emnist_on_mnist = compute_accuracy(emnist_model, mnist_test_loader)
emnist_on_emnist = compute_accuracy(emnist_model, emnist_test_loader)

print("\n=== Final accuracy comparison ===")
print(f"{'Model':<18}{'MNIST test':>14}{'EMNIST test':>14}")
print("-" * 46)
print(f"{'MNIST-trained':<18}{mnist_on_mnist:>14.4f}{mnist_on_emnist:>14.4f}")
print(f"{'EMNIST-trained':<18}{emnist_on_mnist:>14.4f}{emnist_on_emnist:>14.4f}")
