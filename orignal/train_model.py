import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import json
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------------
# Model Architecture Functions
# ----------------------
def get_model(model_name, num_classes, pretrained=True):
    """Get model with proper classifier modification"""
    model_name = model_name.lower()
    
    # ResNet variants
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    # EfficientNet variants
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                        f"Supported models: resnet18, resnet34, resnet50, resnet101, "
                        f"efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3")
    
    return model

def get_image_size(model_name):
    """Get appropriate image size for different models"""
    if "efficientnet_b0" in model_name:
        return 224
    elif "efficientnet_b1" in model_name:
        return 240
    elif "efficientnet_b2" in model_name:
        return 260
    elif "efficientnet_b3" in model_name:
        return 300
    else:  # ResNet variants
        return 224

# ----------------------
# Dataset Functions
# ----------------------
def has_train_val_structure(root):
    return os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val"))

def has_class_subdirs(root):
    try:
        entries = [e for e in os.scandir(root) if e.is_dir()]
        entries = [e for e in entries if e.name.lower() not in ("train", "val", "test", ".git", "__pycache__")]
        return len(entries) > 0
    except FileNotFoundError:
        return False

def get_data_loaders(data_dir, batch_size, image_size, train_split=0.8, seed=42):
    """Create data loaders with appropriate transforms"""
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if has_train_val_structure(data_dir):
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)
        class_names = train_dataset.classes
        print("✓ Using existing train/val folder structure")
    elif os.path.isdir(data_dir) and has_class_subdirs(data_dir):
        print(f"✓ No train/val folders found. Creating {train_split:.0%}/{1-train_split:.0%} split from root")
        base_dataset = datasets.ImageFolder(data_dir)
        n_total = len(base_dataset)
        n_train = int(train_split * n_total)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transforms), train_indices)
        val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transforms), val_indices)
        class_names = base_dataset.classes
    else:
        contents = []
        if os.path.isdir(data_dir):
            contents = [e.name for e in os.scandir(data_dir)]
        raise FileNotFoundError(
            f"Dataset not found in expected structure at: {data_dir}\n"
            f"Expected either:\n"
            f"  - {data_dir}/train/<class folders>, {data_dir}/val/<class folders>\n"
            f"  - Or {data_dir}/<class folders> (will auto-split)\n"
            f"Found entries: {contents}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, class_names

# ----------------------
# Training Functions
# ----------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples += batch_size

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{running_corrects.double() / total_samples:.4f}'
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{running_corrects.double() / total_samples:.4f}'
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc.item(), all_preds, all_labels

def save_model(model, model_name, class_names, epoch, best_acc, save_dir="models"):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}_acc{best_acc:.4f}.pth"
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'class_names': class_names,
        'epoch': epoch,
        'accuracy': best_acc,
        'timestamp': timestamp
    }, filepath)
    
    print(f"✓ Model saved: {filepath}")
    return filepath

def plot_training_history(train_losses, val_losses, train_accs, val_accs, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'bo-', label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, 'ro-', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/{model_name}_training_history_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved: {plot_filename}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/{model_name}_confusion_matrix_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {plot_filename}")
    plt.show()

# ----------------------
# Main Training Function
# ----------------------
def train_model(args):
    """Main training function"""
    print("="*60)
    print("🧬 CANDIDA INFECTION CLASSIFICATION TRAINING")
    print("="*60)
    print(f"📊 Model: {args.model_name}")
    print(f"📁 Dataset: {args.data_dir}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"📈 Epochs: {args.epochs}")
    print(f"🎯 Learning rate: {args.learning_rate}")
    print("-"*60)
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print("-"*60)
    
    # Get image size for model
    image_size = get_image_size(args.model_name)
    print(f"🖼️  Image size: {image_size}x{image_size}")
    
    # Load data
    print("📥 Loading dataset...")
    train_loader, val_loader, class_names = get_data_loaders(
        args.data_dir, args.batch_size, image_size, args.train_split, args.seed
    )
    
    print(f"📊 Classes: {class_names}")
    print(f"🏋️  Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print("-"*60)
    
    # Create model
    print(f"🏗️  Creating {args.model_name} model...")
    model = get_model(args.model_name, len(class_names), pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    print("-"*60)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    patience_counter = 0
    
    print("🚀 Starting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n📍 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check for best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            if args.save_model:
                save_model(model, args.model_name, class_names, epoch+1, best_acc)
            print(f"🎉 New best validation accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            break
            
        print("-" * 60)
    
    print("\n🎯 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print("="*60)
    
    # Final evaluation
    print("\n📋 Final Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot results
    if len(train_losses) > 1:
        plot_training_history(train_losses, val_losses, train_accs, val_accs, args.model_name)
        plot_confusion_matrix(all_labels, all_preds, class_names, args.model_name)
    
    return model, best_acc

# ----------------------
# Command Line Interface
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train CNN models for Candida infection classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet50 (default)
  python train_model.py --data-dir my_dataset

  # Train EfficientNet-B0
  python train_model.py --data-dir my_dataset --model efficientnet_b0

  # Custom training settings
  python train_model.py --data-dir my_dataset --model resnet18 --epochs 30 --batch-size 16 --lr 1e-3

  # Quick test run
  python train_model.py --data-dir my_dataset --epochs 5 --no-save

Supported models:
  ResNet: resnet18, resnet34, resnet50, resnet101
  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
        """
    )
    
    # Data arguments
    parser.add_argument("--data-dir", default="my_dataset", 
                        help="Path to dataset directory (default: my_dataset)")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train split ratio if no train/val folders exist (default: 0.8)")
    
    # Model arguments
    parser.add_argument("--model", "--model-name", dest="model_name", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", 
                                "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"],
                        help="Model architecture (default: resnet50)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (default: 1e-5)")
    
    # Optimization arguments
    parser.add_argument("--no-lr-scheduler", dest="lr_scheduler", action="store_false",
                        help="Disable learning rate scheduler")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (0 to disable, default: 10)")
    
    # Output arguments
    parser.add_argument("--no-save", dest="save_model", action="store_false",
                        help="Don't save the best model")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Train the model
    model, best_acc = train_model(args)
    
    print(f"\n🎊 Training finished! Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()