import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Define a simple neural network with 1 hidden layer
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 input features, 2 neurons in hidden layer
        self.output = nn.Linear(2, 1)  # 2 neurons in hidden, 1 output neuron
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_output = self.sigmoid(self.hidden(x))
        output = self.sigmoid(self.output(hidden_output))
        return output, hidden_output  # Return both output and hidden layer activations

# Initialize the model, loss function, and optimizer
model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training the model
epochs = 1000
losses = []
accuracies = []
hidden_activations = []
output_activations = []
weights_hidden = []
biases_hidden = []
weights_output = []
biases_output = []

for epoch in range(epochs):
    # Forward pass
    output, hidden_output = model(X_tensor)
    
    # Compute loss
    loss = criterion(output, y_tensor)
    losses.append(loss.item())
    
    # Compute accuracy
    predicted = (output >= 0.5).float()  # Apply threshold of 0.5
    accuracy = (predicted == y_tensor).float().mean()
    accuracies.append(accuracy.item())
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track the parameters
    hidden_activations.append(hidden_output.detach().numpy())
    output_activations.append(output.detach().numpy())
    weights_hidden.append(model.hidden.weight.detach().numpy())
    biases_hidden.append(model.hidden.bias.detach().numpy())
    weights_output.append(model.output.weight.detach().numpy())
    biases_output.append(model.output.bias.detach().numpy())

# Helper function to simplify repetitive plotting
def plot_data(ax, data, title, xlabel, ylabel, legend_labels=None):
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_labels:
        ax.legend(legend_labels)

# Plot the results using subplots
fig, axes = plt.subplots(4, 2, figsize=(12, 10))

# Plot Loss Curve
plot_data(axes[0, 0], losses, "Loss over Epochs", "Epochs", "Loss")

# Plot Accuracy Curve
plot_data(axes[0, 1], accuracies, "Accuracy over Epochs", "Epochs", "Accuracy")

# Plot Weight Changes for Hidden Layer
plot_data(axes[1, 0], [np.linalg.norm(w) for w in weights_hidden], 
          "Hidden Layer Weight Norm over Epochs", "Epochs", "Weight Norm")

# Plot Bias Changes for Hidden Layer
plot_data(axes[1, 1], [np.linalg.norm(b) for b in biases_hidden], 
          "Hidden Layer Bias Norm over Epochs", "Epochs", "Bias Norm")

# Plot Weight Changes for Output Layer
plot_data(axes[2, 0], [np.linalg.norm(w) for w in weights_output], 
          "Output Layer Weight Norm over Epochs", "Epochs", "Weight Norm")

# Plot Bias Changes for Output Layer
plot_data(axes[2, 1], [np.linalg.norm(b) for b in biases_output], 
          "Output Layer Bias Norm over Epochs", "Epochs", "Bias Norm")

# Plot Neuron Activations for Hidden Layer (for a single input)
hidden_activations = np.array(hidden_activations)
plot_data(axes[3, 0], hidden_activations[:, 0], "Hidden Layer Activations over Epochs", 
          "Epochs", "Activation", legend_labels=["Neuron 1"])
plot_data(axes[3, 0], hidden_activations[:, 1], "Hidden Layer Activations over Epochs", 
          "Epochs", "Activation", legend_labels=["Neuron 2"])

# Plot Neuron Activations for Output Layer (for a single input)
output_activations = np.array(output_activations)
plot_data(axes[3, 1], output_activations[:, 0], "Output Layer Activations over Epochs", 
          "Epochs", "Activation", legend_labels=["Output Neuron"])

plt.tight_layout()
plt.show()
