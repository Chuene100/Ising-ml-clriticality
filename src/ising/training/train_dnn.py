# Script to train DNN model
import torch
import torch.nn as nn
import torch.optim as optim
from ising.models.dnn import VDNN

def train_dnn(
    data_train,
    labels_train,
    depth=4,
    epochs=10,
    lr=1e-3,
    batch_size=128,
    data_validate=None,
    labels_validate=None,
    is_ab=False,
    is_shuffling=False,
    shuffle_injection_frequency=50,
    print_every=1000,
    factor=0.3,
    class_balance=0.5,
    verbose=True
):
    """Train a VDNN model on Ising data with augmentation and balancing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VDNN(depth).to(device)
    loss_fn = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    losses = []
    model.train()
    NN_train = len(data_train)
    batch_no = int(NN_train / batch_size)
    count = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(NN_train)
        for k in range(batch_no):
            optimizer.zero_grad()
            inds = perm[k*batch_size:(k+1)*batch_size]
            x_batch = torch.tensor(data_train[inds], dtype=torch.float32, device=device)
            y_batch = torch.tensor(labels_train[inds], dtype=torch.float32, device=device)

            # Random sign flip
            sgn = (2 * torch.bernoulli(torch.ones(batch_size, device=device) * 0.5) - 1).unsqueeze(-1)
            dt = x_batch * sgn

            # Adversarial noise (optional)
            if is_ab:
                a, b = torch.rand(2, batch_size, device=device)
                a = 0.25 + 2 * a.unsqueeze(-1)
                b = (b.unsqueeze(-1) - 0.5)
                dt = dt * a
                sgn = (2 * torch.bernoulli(torch.ones(batch_size, device=device) * 0.5) - 1).unsqueeze(-1)
                dt = dt * sgn + b

            # Shuffling (optional)
            if is_shuffling:
                s_inds = torch.randperm(dt.shape[1], device=device)
                dt[:shuffle_injection_frequency, :] = dt[:shuffle_injection_frequency, s_inds]
                y_batch[:shuffle_injection_frequency] = 0

            # Random lattice shift
            i, j = torch.randint(40, (2,), device=device)
            dt2 = dt.reshape(batch_size, 40, 40)
            dt = torch.roll(dt2, shifts=(i.item(), j.item()), dims=(1, 2)).reshape(batch_size, 1600)

            # Forward pass
            outputs = model(dt)
            targets = y_batch.unsqueeze(1)

            # Class balancing
            with torch.no_grad():
                wghts = (class_balance * targets + (1 - class_balance) * (1 - targets))
                wghts = wghts / torch.sum(wghts)
            loss_tmp = loss_fn(outputs, targets)
            loss = torch.sum(loss_tmp.squeeze(1) * wghts.squeeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Periodic validation
            if (k % print_every == 0) and (data_validate is not None) and (labels_validate is not None):
                model.eval()
                with torch.no_grad():
                    dval = torch.tensor(data_validate, dtype=torch.float32, device=device)
                    lval = torch.tensor(labels_validate, dtype=torch.float32, device=device)
                    outval = model(dval)
                    loss_val = loss_fn(outval, lval.unsqueeze(-1)).mean().item()
                if verbose:
                    print(f"Epoch {epoch+1}, Batch {k}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f}")
                model.train()

        epoch_loss /= batch_no
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        count += 1

    return model, losses
