import matplotlib.pyplot as plt
import wandb


def plot_results(epochs, results_per_epoch, corruption_types, metric="wer"):
    plt.figure(figsize=(10, 6))

    for corruption_type in corruption_types:
        plt.plot(epochs, [results_per_epoch[epoch][corruption_type][metric] for epoch in epochs], label=corruption_type)
    
    plt.xlabel('Epochs')
    plt.ylabel(f'Average {metric.upper()}')
    plt.title(f'{metric.upper()} over Epochs')
    plt.legend()
    plt.grid(True)
    wandb.log({"plot": wandb.Image(plt)})
    plt.close()