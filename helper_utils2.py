import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_data(distances, times, normalize=False):
    plt.figure(figsize=(8,6))
    plt.plot(distances.numpy(),times.numpy(),marker='o',color='orange',linestyle='None',label='Actual Delivery Times')
    if normalize:
        plt.title("Normalized Delivery Data (Cars & Bikes)")
        plt.xlabel("Normalized Distance")
        plt.ylabel("Normalized Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        plt.title("Delivery Data (Cars & Bikes)")
        plt.xlabel("Distance (miles)")
        plt.ylabel("Time (mins)")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    model.eval()
    with torch.no_grad():
        prediction_norm=model(distances_norm)
    
    # normalized = (value - mean) / std
    # original = (normalized * std) + mean
    predicted_times = (prediction_norm*times_std) + times_mean
    
    plt.figure(figsize=(8,6))
    plt.plot(distances.numpy(),times.numpy(),color='orange',marker='o',linestyle='None',label='Actual Data (bikes & cars)')
    plt.plot(distances.numpy(), predicted_times.numpy(), color="Dark Green",label='Non-Linear Model Predictions')
    plt.title("Non-Linear Model Fit vs Actual Data")
    plt.xlabel("Distance(miles)")
    plt.ylabel("Time(mins)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    clear_output(wait=True)
    predicted_norm=model(distances_norm)
    x_plot=distances_norm.numpy() 
    y_plot=times_norm.numpy()
    y_pred_plot=predicted_norm.detach().numpy()
    sorted_indices=x_plot.argsort(axis=0).flatten()

    plt.figure(figsize=(8,6))
    plt.plot(x_plot,y_plot,color='orange',marker='o',linestyle='None',label='Actual Normalized Data')
    plt.plot(x_plot[sorted_indices],y_pred_plot[sorted_indices], color='green', label='Model Predictions')
    plt.title(f"Epoch: {epoch+1} | Normalized Training Progress")
    plt.xlabel("Normalized Distance")
    plt.ylabel("Normalzied Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    time.sleep(0.05)

