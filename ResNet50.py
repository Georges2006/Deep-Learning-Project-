import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMAGES_DIR    = r"C:\Users\georg\OneDrive\Bureau\ISEP\DEEP LEARNING\DEEP Learning - Project\Dataset\flickr30k_images"
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 0.001
IMAGE_SIZE    = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")

# ==============================================================================
# ETAPE 1 : CHARGEMENT DE RESNET-50 PRE-ENTRAINE
# ResNet-50 est entraine sur ImageNet (1000 classes, 1.2M images)
# On recupere ses poids pour extraire des features visuelles riches
# ==============================================================================
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

print(f"Parametres ResNet-50 : {sum(p.numel() for p in resnet50.parameters()):,}")

# ==============================================================================
# ETAPE 2 : FINE-TUNING DES COUCHES FULLY CONNECTED
# On gele le backbone CNN (extraction de features)
# On remplace uniquement la couche FC finale pour nos classes specifiques
# ==============================================================================
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Gel du backbone : les features CNN restent celles apprises sur ImageNet
    for param in model.parameters():
        param.requires_grad = False

    # Remplacement de la couche FC pour l'adapter a nos classes
    # ResNet-50 produit un vecteur de 2048 features avant la FC
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),  # 2048 -> 512
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)             # 512 -> nombre de classes
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parametres entrainables (FC) : {trainable:,} / {total:,}")

    return model

# ==============================================================================
# ETAPE 3 : DATASET FLICKR30K
# Charge les images du dossier, label = index de l'image
# ==============================================================================
class Flickr30kDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.transform   = transform
        self.image_paths = []

        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(images_dir, fname))

        print(f"Images chargees : {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(idx, dtype=torch.long)

# ==============================================================================
# ETAPE 4 : TRANSFORMATIONS
# ResNet-50 attend des images 224x224 normalisees avec les stats ImageNet
# ==============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ==============================================================================
# ETAPE 5 : CHARGEMENT DES DONNEES (80% train / 20% val)
# ==============================================================================
full_dataset = Flickr30kDataset(IMAGES_DIR, transform=train_transform)
NUM_CLASSES  = len(full_dataset)

val_size   = int(0.2 * NUM_CLASSES)
train_size = NUM_CLASSES - val_size

train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train : {train_size} | Val : {val_size}")

# ==============================================================================
# ETAPE 6 : CONSTRUCTION DU MODELE
# ==============================================================================
model = build_model(num_classes=NUM_CLASSES).to(DEVICE)

# ==============================================================================
# ETAPE 7 : EXTRACTION DE FEATURES VISUELLES
# On retire la couche FC pour obtenir les vecteurs 2048-D du backbone
# Ces features representent le contenu visuel de chaque image
# ==============================================================================
def extract_visual_features(model, loader):
    model.eval()

    # Backbone sans la FC finale = extracteur de features pur
    backbone = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)

    all_features = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(DEVICE)
            features = backbone(images)           # (batch, 2048, 1, 1)
            features = features.flatten(1)        # (batch, 2048)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

# ==============================================================================
# ETAPE 8 : INFERENCE SUR UNE IMAGE
# ==============================================================================
def predict(model, image_path, top_k=5):
    image  = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs       = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_idx = torch.topk(probabilities, top_k)

    print(f"\nTop {top_k} predictions :")
    for i in range(top_k):
        print(f"  Classe {top_idx[i].item()} : {top_prob[i].item()*100:.2f}%")

# ==============================================================================
# ETAPE 9 : ENTRAINEMENT
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
    # phase entrainement
    model.train()
    train_loss, train_correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # phase validation
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            val_loss      += criterion(outputs, labels).item()
            val_correct   += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch}/{NUM_EPOCHS} "
          f"| Train Loss: {train_loss/len(train_loader):.4f} "
          f"Acc: {train_correct/train_size*100:.2f}% "
          f"| Val Loss: {val_loss/len(val_loader):.4f} "
          f"Acc: {val_correct/val_size*100:.2f}%")

# ==============================================================================
# ETAPE 10 : SAUVEGARDE
# ==============================================================================
torch.save(model.state_dict(), "resnet50_flickr30k.pth")
print("Modele sauvegarde : resnet50_flickr30k.pth")

# Extraction et sauvegarde des features visuelles
features, labels = extract_visual_features(model, train_loader)
torch.save({"features": features, "labels": labels}, "flickr30k_features.pth")
print(f"Features visuelles sauvegardees : {features.shape}")

# Exemple d'inference sur une image
# predict(model, r"C:\Users\georg\OneDrive\Bureau\ISEP\DEEP LEARNING\DEEP Learning - Project\Dataset\flickr30k_images\exemple.jpg")