import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    model.eval()
    with torch.no_grad():
        predicted_times=model(distances)
    plt.figure(figsize=(8,6))
    plt.plot(distances.numpy(),times.numpy(),color='orange',marker='o',linestyle='None',label="Actual Delivery Times")
    plt.plot(distances.numpy(),predicted_times.numpy(),color='blue',label="Predicted Line", marker='None')
    plt.title("Actual vs Predicted Delivery Times")
    plt.xlabel("Distances(miles)")
    plt.ylabel("Time(mins)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_nonlinear_comparison(model,new_distances,new_times):
    model.eval()
    with torch.no_grad():
        predictionss = model(new_distances)
    plt.figure(figsize=(8,6))
    plt.plot(new_distances.numpy(),new_times.numpy(),color='olive',marker='o',linestyle='None',label='Actual Data(Bikes & Cars)')
    plt.plot(new_distances.numpy(), predictionss.numpy(),color='yellow',marker='None',label='Linear Model Predictions')
    plt.title("Linear Model vs Non-Linear Reality")
    plt.xlabel("Distances(miles)")
    plt.ylabel("Times(mins)")
    plt.grid(True)
    plt.legend()
    plt.show()

