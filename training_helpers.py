import torch
import inspect

def train_model(
    model,
    X_train,
    Y_train,
    epochs,
    lr=1e-3,
    weight_decay=1e-4,
    min_lr=1e-8,
    batch_size=64,
    max_norm=10.0,
    init_ratio=0.7,
    device="cuda",
):
    torch.backends.cudnn.benchmark = True
    model.to(device)
    model.train()

    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=10,
        threshold = 1e-4,
        min_lr=min_lr,
    )

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    
    def forward_pass(xb, yb):
        # check if model supports teacher_forcing_ratio
        sig = inspect.signature(model.forward)
        if "teacher_forcing_ratio" in sig.parameters:
            return model(xb, yb, teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            return model(xb)

    for epoch in range(epochs):
        teacher_forcing_ratio = max(init_ratio * (1 - epoch/epochs), 0.1)
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda"):
                preds = forward_pass(xb, yb)
                loss = criterion(preds.float(), yb.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)    
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss:.6f} | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")