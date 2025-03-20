import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from load_data import CapsNet  

print("Lancement du test d'overfitting sur 500 images...\n")

#dataset de 500 images pour overfitting
transform = transforms.ToTensor()
full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
small_train_dataset = Subset(full_train_dataset, range(500))  
train_loader = DataLoader(small_train_dataset, batch_size=32, shuffle=True)

# Charger le modèle et vérifier GPU dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsNet().to(device)
print(f" Modèle chargé sur : {device}")

#  Définir la loss et l'optimiseur
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Entraîner sur 10 epochs 
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  

        # Vérification des shapes
        #print(f" Outputs shape: {outputs.shape}")  
        #print(f" Labels shape: {labels.shape}")  

        outputs_norm = outputs.norm(dim=-1) 
        loss = criterion(outputs_norm, labels)  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs_norm.argmax(dim=1) 

        #Calcul de l'accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Affichage des résultats
    train_acc = correct / total
    print(f" Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

    
    if train_acc == 1.0:
        print(" Modèle a atteint 100% d’accuracy → Il overfit bien sur 500 images ")
        break  

print(" Test d'overfitting terminé !")
