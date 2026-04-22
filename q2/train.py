import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_loaders
from model import UNet
from metrics import compute_iou, compute_dice
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("outputs", exist_ok=True)

train_loader, test_loader = get_loaders("data")

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

losses, miou_scores, mdice_scores = [], [], []

for epoch in range(15):
    model.train()
    total_loss = 0

    for i, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)

        # 🔥 DEBUG ONLY FIRST BATCH
        if epoch == 0 and i == 0:
            print("Pred shape:", preds.shape)
            print("Mask shape:", masks.shape)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    # Evaluation
    model.eval()
    ious, dices = [], []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)

            ious.append(compute_iou(preds, masks))
            dices.append(compute_dice(preds, masks))

    miou = torch.mean(torch.stack(ious)).item()
    mdice = torch.mean(torch.stack(dices)).item()

    miou_scores.append(miou)
    mdice_scores.append(mdice)

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, mIoU={miou:.4f}, Dice={mdice:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")

# Save plots
plt.plot(losses)
plt.title("Training Loss")
plt.savefig("outputs/loss.png")

plt.figure()
plt.plot(miou_scores)
plt.title("mIoU")
plt.savefig("outputs/miou.png")

plt.figure()
plt.plot(mdice_scores)
plt.title("mDice")
plt.savefig("outputs/mdice.png")