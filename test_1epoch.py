import torch
import itertools
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from load_data import CapsNet, MarginLoss 

# Vérification du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Lancement de la recherche d'hyperparamètres sur {device}...\n")

# Définir les hyperparamètres à tester
learning_rates = [0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
reconstruction_weights = [0.0005, 0.0003, 0.0007, 0.001, 0.002]

# Chargement du dataset MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# Initialisation de TensorBoard
writer = SummaryWriter(log_dir="runs/capsnet_grid_search_1epoch")

# Variables pour stocker le meilleur résultat
best_hyperparams = None
best_val_acc = 0.0
global_step = 0

# Tester toutes les combinaisons (LR, Weight Recon Loss)
for lr, recon_w in itertools.product(learning_rates, reconstruction_weights):
    print(f"Testing: LR={lr}, Recon Weight={recon_w}")

    # Charger le modèle sauvegardé
    model = CapsNet().to(device)
    

    # Définir l'optimiseur et la fonction de perte
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = MarginLoss()

    # === Phase Entraînement (1 Epoch) ===
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        labels_onehot = torch.eye(10).to(device).index_select(dim=0, index=labels)

        optimizer.zero_grad()
        caps_output = model(images)
        loss = loss_function(caps_output, labels_onehot)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        pred_classes = torch.norm(caps_output, dim=2).argmax(dim=1)
        correct_train += (pred_classes == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = correct_train / total_train_samples
    print(f"Train Accuracy: {train_acc:.4f} | Train Loss: {avg_train_loss:.4f}")

    # === Phase Validation ===
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels_onehot = torch.eye(10).to(device).index_select(dim=0, index=labels)

            caps_output = model(images)
            loss = loss_function(caps_output, labels_onehot)

            total_val_loss += loss.item()
            pred_classes = torch.norm(caps_output, dim=2).argmax(dim=1)
            correct_val += (pred_classes == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct_val / total_val_samples
    print(f"Validation Accuracy: {val_acc:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # Enregistrement des résultats sur TensorBoard
    writer.add_scalar("Train/Loss", avg_train_loss, global_step)
    writer.add_scalar("Train/Accuracy", train_acc, global_step)
    writer.add_scalar("Validation/Loss", avg_val_loss, global_step)
    writer.add_scalar("Validation/Accuracy", val_acc, global_step)
    writer.add_text("Hyperparameters", f"LR={lr}, Recon Weight={recon_w}", global_step)

    global_step += 1  # Incrémentation du compteur

    # Mise à jour du meilleur modèle si meilleure précision trouvée
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_hyperparams = (lr, recon_w)

writer.close()

# Affichage des meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres: LR={best_hyperparams[0]}, Recon Weight={best_hyperparams[1]}")
print(f" Meilleure Validation Accuracy: {best_val_acc:.4f}")

