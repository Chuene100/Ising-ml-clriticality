
# Script to train CNN model
import torch
import torch.nn as nn
import torch.optim as optim
from ising.models.cnn import VCNN

def train_cnn(data_train, labels_train, epochs=10, lr=1e-3, batch_size=128):
    """Train a VCNN model on Ising data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VCNN().to(device)
    loss_fn = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(data_train), batch_size):
            x_batch = torch.tensor(data_train[i:i+batch_size], dtype=torch.float32, device=device)
            y_batch = torch.tensor(labels_train[i:i+batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.mean().item()
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return model, losses
