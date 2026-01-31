import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def test_long_horizon(model, X_test, scaler, test_days, horizon, original_prices=None, mode="price", device="cuda"):
    print("X_test shape:", X_test.shape)
    print("X_test ndim:", X_test.ndim)
    
    model.eval()
    model.to(device)
    
    current_window = X_test[-1:].to(device)  # (1, WINDOW, F)
    preds_all = []
    steps = test_days // horizon

    with torch.no_grad():
        for _ in range(steps):
            # Predict next horizon
            preds = model(current_window)          # (1, HORIZON, F)
            preds_all.append(preds.cpu())

            # Slide window forward
            current_window = torch.cat(
                [current_window[:, horizon:, :], preds],
                dim=1
            )

    preds_all = torch.cat(preds_all, dim=1).squeeze(0).numpy()

    # Inverse scaling
    if mode == "price":
        # scaled raw prices → just inverse transform
        preds_real = scaler.inverse_transform(preds_all)
    elif mode == "log_return":
        # first inverse transform to get actual log returns
        preds_unscaled = scaler.inverse_transform(preds_all)
        last_price = original_prices[-1]  # last actual price in test set
        preds_real = last_price * np.exp(np.cumsum(preds_unscaled, axis=0))
    elif mode == "pct_change":
        preds_unscaled = scaler.inverse_transform(preds_all)
        last_price = original_prices[-1]
        preds_real = last_price * np.cumprod(1 + preds_unscaled, axis=0)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return preds_real
    
def directional_accuracy(y_true, y_pred):
    return np.mean(
        np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    )
    
def stats(y_true, y_pred, feature_idx=3):
    rmse = np.sqrt(mean_squared_error(
        y_true[:, feature_idx],
        y_pred[:, feature_idx]
    ))

    mae = mean_absolute_error(
        y_true[:, feature_idx],
        y_pred[:, feature_idx]
    )

    da = np.mean(
        np.sign(np.diff(y_true[:, feature_idx])) ==
        np.sign(np.diff(y_pred[:, feature_idx]))
    )

    print(f"RMSE (Close): {rmse:.4f}")
    print(f"MAE (Close): {mae:.4f}")
    print(f"Directional Accuracy: {da:.3f}")
    
def plot(y_true, y_pred, feature_idx=3, title="Forecast"):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:, feature_idx], label="Actual", linewidth=2)
    plt.plot(y_pred[:, feature_idx], label="Predicted", linestyle="--")
    plt.xlabel("Trading Days Ahead")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()