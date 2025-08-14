
# Adversarial training routines
import torch
import torch.nn as nn
import torch.optim as optim

def adversarial_training(model, data_train, labels_train, epsilon=0.1, epochs=10, lr=1e-3, batch_size=128):
    """Adversarial training for Ising models using FGSM."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(data_train), batch_size):
            x_batch = torch.tensor(data_train[i:i+batch_size], dtype=torch.float32, device=device, requires_grad=True)
            y_batch = torch.tensor(labels_train[i:i+batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.mean().backward()
            # FGSM adversarial example
            data_grad = x_batch.grad.data
            x_adv = x_batch + epsilon * data_grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
            outputs_adv = model(x_adv)
            loss_adv = loss_fn(outputs_adv, y_batch)
            loss_total = (loss + loss_adv).mean()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            epoch_loss += loss_total.item()
        print(f"Adversarial Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return model
