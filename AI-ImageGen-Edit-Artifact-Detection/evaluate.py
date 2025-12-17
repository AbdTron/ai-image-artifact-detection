"""
Comprehensive Evaluation Script for AI Image Detection
Features:
- All classification metrics (TP, TN, FP, FN, Accuracy, Precision, Recall, etc.)
- Multiple visualizations (Confusion Matrix, ROC, Precision-Recall, etc.)
- JPEG robustness test
- Detailed report generation
"""
import argparse
from pathlib import Path
import io

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef
)
from torch.cuda.amp import autocast
from tqdm import tqdm

import config
from dataset import get_dataloaders, get_val_transforms, AIImageDataset
from model import create_model


def apply_jpeg_compression(image_tensor, quality=70):
    """Apply JPEG compression to simulate real-world sharing."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = image_tensor * std + mean
    img = (img * 255).clamp(0, 255).byte()
    
    img_pil = Image.fromarray(img.permute(1, 2, 0).numpy())
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer).convert("RGB")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    return transform(img_compressed)


@torch.no_grad()
def evaluate(model, loader, device, apply_jpeg=False, jpeg_quality=70):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        if apply_jpeg:
            images = torch.stack([apply_jpeg_compression(img, jpeg_quality) for img in images])
        
        images = images.to(device)
        
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
        
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate all classification metrics."""
    # Confusion matrix values
    # For binary with 0=AI (positive), 1=Real (negative):
    # cm = [[TP, FN],   <- Actual class 0 (AI)
    #       [FP, TN]]   <- Actual class 1 (Real)
    # ravel() gives row-major order: TP, FN, FP, TN
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    
    # Core metrics (AI=0 is positive class)  
    accuracy = accuracy_score(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # TP / (TP + FP)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # TP / (TP + FN) = TPR
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    misclassification_rate = 1 - accuracy
    balanced_accuracy = (recall + specificity) / 2
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # AUC (using probs for Real class, so flip for AI-Generated)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, 1 - y_probs, pos_label=0)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    return {
        # Confusion matrix values
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        # Core metrics
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall/TPR": recall * 100,
        "Specificity/TNR": specificity * 100,
        "F1 Score": f1 * 100,
        # Additional metrics
        "FPR": fpr * 100,
        "NPV": npv * 100,
        "Misclassification Rate": misclassification_rate * 100,
        "Balanced Accuracy": balanced_accuracy * 100,
        "MCC": mcc,
        "AUC": roc_auc,
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = config.CLASS_NAMES
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix\nAI-Generated/AI-Edited vs Real/Authentic Detection',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot and save ROC curve."""
    # For AI-Generated as positive class
    fpr, tpr, _ = roc_curve(y_true, 1 - y_probs, pos_label=0)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve - AI-Generated/AI-Edited Detection', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved: {save_path}")
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, save_path):
    """Plot and save Precision-Recall curve."""
    # For AI-Generated as positive class
    precision, recall, _ = precision_recall_curve(y_true, 1 - y_probs, pos_label=0)
    ap = average_precision_score(y_true, 1 - y_probs, pos_label=0)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=3, label=f'PR curve (AP = {ap:.3f})')
    plt.fill_between(recall, precision, alpha=0.3, color='green')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - AI-Generated/AI-Edited Detection', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved: {save_path}")
    return ap


def plot_metrics_bar_chart(metrics, save_path):
    """Plot all metrics as a bar chart."""
    # Select percentage-based metrics for bar chart
    bar_metrics = {
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall/TPR': metrics['Recall/TPR'],
        'Specificity/TNR': metrics['Specificity/TNR'],
        'F1 Score': metrics['F1 Score'],
        'Balanced Acc': metrics['Balanced Accuracy'],
        'AUC×100': metrics['AUC'] * 100,
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bar_metrics.keys(), bar_metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c', '#e67e22'])
    
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Metrics\nAI-Generated/AI-Edited vs Real/Authentic Detection', fontsize=14)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
    
    # Add value labels on bars
    for bar, val in zip(bars, bar_metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics bar chart saved: {save_path}")


def plot_class_distribution(train_dataset, test_dataset, save_path):
    """Plot dataset class distribution."""
    # Count classes
    train_fake = sum(1 for _, label in train_dataset.samples if label == 0)
    train_real = len(train_dataset.samples) - train_fake
    test_fake = sum(1 for _, label in test_dataset.samples if label == 0)
    test_real = len(test_dataset.samples) - test_fake
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    colors = ['#e74c3c', '#2ecc71']
    labels = ['AI-Generated/\nAI-Edited', 'Real/Authentic']
    
    axes[0].pie([train_fake, train_real], labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11})
    axes[0].set_title(f'Training Set Distribution\n(n={len(train_dataset.samples):,})', fontsize=12)
    
    # Test set
    axes[1].pie([test_fake, test_real], labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11})
    axes[1].set_title(f'Test Set Distribution\n(n={len(test_dataset.samples):,})', fontsize=12)
    
    plt.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Class distribution chart saved: {save_path}")


def plot_jpeg_comparison(acc_normal, acc_jpeg, save_path):
    """Plot JPEG robustness comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(['Original Images', 'JPEG Compressed\n(Quality=70)'], 
                  [acc_normal, acc_jpeg], color=['#3498db', '#e67e22'], width=0.5)
    
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('JPEG Robustness Test\nModel Performance Before/After Compression', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, [acc_normal, acc_jpeg]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add accuracy drop annotation
    drop = acc_normal - acc_jpeg
    ax.annotate(f'Drop: {drop:.2f}%', xy=(0.5, min(acc_normal, acc_jpeg) - 5),
                ha='center', fontsize=12, color='red' if drop > 5 else 'green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ JPEG comparison chart saved: {save_path}")


def save_evaluation_report(metrics, output_path, jpeg_metrics=None):
    """Save detailed evaluation report to text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AI-GENERATED/AI-EDITED IMAGE DETECTION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL: EfficientNet-B4\n")
        f.write("TASK: Binary Classification (AI vs Real)\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("CONFUSION MATRIX VALUES\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Positives (TP):  {metrics['TP']}\n")
        f.write(f"True Negatives (TN):  {metrics['TN']}\n")
        f.write(f"False Positives (FP): {metrics['FP']}\n")
        f.write(f"False Negatives (FN): {metrics['FN']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:              {metrics['Accuracy']:.2f}%\n")
        f.write(f"Precision:             {metrics['Precision']:.2f}%\n")
        f.write(f"Recall (TPR):          {metrics['Recall/TPR']:.2f}%\n")
        f.write(f"Specificity (TNR):     {metrics['Specificity/TNR']:.2f}%\n")
        f.write(f"F1 Score:              {metrics['F1 Score']:.2f}%\n")
        f.write(f"False Positive Rate:   {metrics['FPR']:.2f}%\n")
        f.write(f"Negative Pred Value:   {metrics['NPV']:.2f}%\n")
        f.write(f"Misclassification:     {metrics['Misclassification Rate']:.2f}%\n")
        f.write(f"Balanced Accuracy:     {metrics['Balanced Accuracy']:.2f}%\n")
        f.write(f"MCC:                   {metrics['MCC']:.4f}\n")
        f.write(f"AUC:                   {metrics['AUC']:.4f}\n\n")
        
        if jpeg_metrics:
            f.write("-" * 40 + "\n")
            f.write("JPEG ROBUSTNESS TEST (Quality=70)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy after JPEG:   {jpeg_metrics['Accuracy']:.2f}%\n")
            f.write(f"Accuracy drop:         {metrics['Accuracy'] - jpeg_metrics['Accuracy']:.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Evaluation report saved: {output_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = create_model().to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.2f}%")
    
    # Load data
    train_loader, _, test_loader = get_dataloaders()
    
    # Get dataset objects for class distribution
    train_dataset = AIImageDataset(config.TRAIN_DIR, transform=get_val_transforms())
    test_dataset = AIImageDataset(config.TEST_DIR, transform=get_val_transforms())
    
    output_dir = Path(args.model).parent
    
    # Plot class distribution
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    plot_class_distribution(train_dataset, test_dataset, output_dir / "class_distribution.png")
    
    # Standard evaluation
    print("\n" + "=" * 50)
    print("STANDARD EVALUATION")
    print("=" * 50)
    
    preds, labels, probs = evaluate(model, test_loader, device)
    metrics = calculate_all_metrics(labels, preds, probs)
    
    # Print metrics
    print("\n" + "-" * 40)
    print("CONFUSION MATRIX VALUES")
    print("-" * 40)
    print(f"True Positives (TP):  {metrics['TP']}")
    print(f"True Negatives (TN):  {metrics['TN']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    
    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Accuracy:              {metrics['Accuracy']:.2f}%")
    print(f"Precision:             {metrics['Precision']:.2f}%")
    print(f"Recall (TPR):          {metrics['Recall/TPR']:.2f}%")
    print(f"Specificity (TNR):     {metrics['Specificity/TNR']:.2f}%")
    print(f"F1 Score:              {metrics['F1 Score']:.2f}%")
    print(f"False Positive Rate:   {metrics['FPR']:.2f}%")
    print(f"NPV:                   {metrics['NPV']:.2f}%")
    print(f"Misclassification:     {metrics['Misclassification Rate']:.2f}%")
    print(f"Balanced Accuracy:     {metrics['Balanced Accuracy']:.2f}%")
    print(f"MCC:                   {metrics['MCC']:.4f}")
    print(f"AUC:                   {metrics['AUC']:.4f}")
    
    # Classification report
    print("\n" + "-" * 40)
    print("CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(labels, preds, target_names=config.CLASS_NAMES))
    
    # Generate plots
    print("\n" + "-" * 40)
    print("GENERATING VISUALIZATIONS")
    print("-" * 40)
    plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    plot_roc_curve(labels, probs, output_dir / "roc_curve.png")
    plot_precision_recall_curve(labels, probs, output_dir / "precision_recall_curve.png")
    plot_metrics_bar_chart(metrics, output_dir / "metrics_bar_chart.png")
    
    # JPEG robustness test
    jpeg_metrics = None
    if args.jpeg_test:
        print("\n" + "=" * 50)
        print(f"JPEG ROBUSTNESS TEST (quality={args.jpeg_quality})")
        print("=" * 50)
        
        preds_jpeg, labels_jpeg, probs_jpeg = evaluate(
            model, test_loader, device, 
            apply_jpeg=True, jpeg_quality=args.jpeg_quality
        )
        
        jpeg_metrics = calculate_all_metrics(labels_jpeg, preds_jpeg, probs_jpeg)
        
        print(f"\nAccuracy after JPEG compression: {jpeg_metrics['Accuracy']:.2f}%")
        print(f"Accuracy drop: {metrics['Accuracy'] - jpeg_metrics['Accuracy']:.2f}%")
        
        plot_jpeg_comparison(metrics['Accuracy'], jpeg_metrics['Accuracy'], 
                           output_dir / "jpeg_robustness.png")
    
    # Save report
    save_evaluation_report(metrics, output_dir / "evaluation_report.txt", jpeg_metrics)
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print("=" * 50)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--jpeg-test", action="store_true", help="Run JPEG robustness test")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    args = parser.parse_args()
    main(args)
