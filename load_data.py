import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Data Augmentation : Translation de ±2 pixels
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),  # Translation ±2 pixels sur 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chargement du dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# Séparer le dataset d'entraînement en train (50k) et validation (10k)
train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Nombre d'images d'entraînement : {len(train_dataset)}")
print(f"Nombre d'images de validation : {len(val_dataset)}")
print(f"Nombre d'images de test : {len(test_dataset)}")

# Exemple pour afficher quelques images transformées
examples = iter(train_loader)
images, labels = next(examples)

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')

plt.show()

#Création de l'architecture:
# === Conv1 Layer ===
class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=9, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))


# === Primary Capsules ===
class PrimaryCapsules(nn.Module):
    def __init__(self):
        super(PrimaryCapsules, self).__init__()
        self.conv_caps = nn.Conv2d(256, 256, kernel_size=9, stride=2)

    def forward(self, x):
        x = self.conv_caps(x)
        x = x.view(x.size(0), 32, 8, 6, 6)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(x.size(0), -1, 8)
        return self.squash(x)

    def squash(self, x):
        norm = (x ** 2).sum(dim=2, keepdim=True)
        scale = norm / (1 + norm)
        return scale * x / torch.sqrt(norm + 1e-8)


# === Digit Capsules ===
class DigitCapsules(nn.Module):
    def __init__(self, num_routes=1152, num_capsules=10, in_dim=8, out_dim=16, num_iterations=3,dropout_prob=0.2):
        super(DigitCapsules, self).__init__()
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)

        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout ajouté

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(2).unsqueeze(4)
        x = x.expand(-1, -1, self.num_capsules, -1, -1)

        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)

        b_ij = torch.randn(batch_size, self.num_routes, self.num_capsules, device=x.device) * 0.01

        for _ in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            c_ij = self.dropout(c_ij)
            c_ij = c_ij.unsqueeze(-1)
            s_j = (c_ij * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
            b_ij = b_ij + agreement

        return v_j
    
    def squash(self, s_j):
        norm = (s_j ** 2).sum(dim=2, keepdim=True)
        scale = norm / (1 + norm)
        return scale * s_j / torch.sqrt(norm + 1e-8)


# === CapsNet ===
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = ConvLayer()
        self.primary_caps = PrimaryCapsules()
        self.digit_caps = DigitCapsules()

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x


# === Decoder for Reconstruction ===
class Decoder(nn.Module):
    def __init__(self,dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, capsule_output, labels):
        capsule_masked = (capsule_output * labels.unsqueeze(-1)).sum(dim=1)
        reconstructed = self.decoder(capsule_masked)
        return reconstructed


# === Margin Loss ===
class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()
        self.m_pos = 0.9
        self.m_neg = 0.1
        self.lambda_ = 0.5

    def forward(self, capsule_output, labels):
        lengths = torch.norm(capsule_output, dim=2)
        loss_pos = labels * F.relu(self.m_pos - lengths) ** 2
        loss_neg = (1 - labels) * F.relu(lengths - self.m_neg) ** 2
        loss = loss_pos + self.lambda_ * loss_neg
        return loss.sum(dim=1).mean()


# === Reconstruction Loss ===
def reconstruction_loss(reconstructed, images):
    images = images.view(-1, 28 * 28)
    return F.mse_loss(reconstructed, images, reduction='mean')


# === Initialisation des modèles et de l’optimiseur ===
model = CapsNet().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.0003)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
margin_loss_fn = MarginLoss()

if __name__ == "__main__":
    # === Entraînement ===
    writer = SummaryWriter()

    num_epochs = 100
    log_interval = 1000

    global_step = 0

    best_val_loss = float('inf')
    patience = 5  # Arrêt de l'entrainement si pas d'amélioration après 5 épochs
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # === Phase Train ===
        model.train()
        total_train_loss = 0
        correct_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels_onehot = torch.eye(10).to(device).index_select(dim=0, index=labels)

            caps_output = model(images)
            lengths = torch.norm(caps_output, dim=2)
            #print(f"Longueurs capsules train (moyenne) : {lengths.mean().item():.4f}")
            margin_loss = margin_loss_fn(caps_output, labels_onehot)
            reconstructed = decoder(caps_output, labels_onehot)
            recon_loss = reconstruction_loss(reconstructed, images)
            loss = margin_loss + 0.0007 * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logs train
            total_train_loss += loss.item()
            pred_classes = torch.norm(caps_output, dim=2).argmax(dim=1)
            correct_train += (pred_classes == labels).sum().item()

            global_step += images.size(0)
            if global_step % log_interval == 0:
                print(f"[Epoch {epoch + 1}/{num_epochs}] {global_step}/{len(train_loader.dataset)} images - Train Loss: {loss.item():.4f}")
                writer.add_scalar('Train/Loss', loss.item(), global_step)

        # === Phase Validation ===
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels_onehot = torch.eye(10).to(device).index_select(dim=0, index=labels)

                caps_output = model(images)
                lengths = torch.norm(caps_output, dim=2)
                #print(f"Longueurs capsules validation (moyenne) : {lengths.mean().item():.4f}")
                margin_loss = margin_loss_fn(caps_output, labels_onehot)
                reconstructed = decoder(caps_output, labels_onehot)
                recon_loss = reconstruction_loss(reconstructed, images)
                loss = margin_loss + 0.0005 * recon_loss

                total_val_loss += loss.item()
                pred_classes = torch.norm(caps_output, dim=2).argmax(dim=1)
                correct_val += (pred_classes == labels).sum().item()

        # === Moyennes des pertes et précisions ===
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = correct_train / len(train_loader.dataset)
        val_accuracy = correct_val / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


        # Vérifier si la val_loss est la meilleure
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # sauvegarde du meilleur modèle :
            torch.save(model.state_dict(), 'best_capsnet_model.pth')
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping à l’époque {epoch + 1}")
            break

        # === Learning Rate Scheduling ===
        scheduler.step(avg_val_loss)    
        
        
        # === Logs TensorBoard ===
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Epoch_Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation/Epoch_Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/Epoch_Accuracy', val_accuracy, epoch)

    writer.close()

    # ===Évaluation sur le Test Set ===
    print("Chargement du meilleur modèle pour évaluation sur le Test Set...")
    model.load_state_dict(torch.load('best_capsnet_model.pth'))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            caps_output = model(images)
            pred_classes = torch.norm(caps_output, dim=2).argmax(dim=1)

            correct_test += (pred_classes == labels).sum().item()
            total_test += labels.size(0)

    test_accuracy = correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.4f}")