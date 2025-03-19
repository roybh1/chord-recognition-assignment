import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.acoustic_model import AcousticModel
from model.dataset import create_dataset

def get_dataset(audio_dir, labels_dir, output_path):
    return create_dataset(audio_dir, labels_dir, output_path)

def load_dataset(dataset_path):
    return torch.load(dataset_path)

def split_dataset(dataset, train_size=0.8, val_size=0.2):
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def get_data_loaders(dataset, batch_size=32):
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch [{epoch+1}], "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2%}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered")
                break

    # Load the best model after training
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AcousticModel(input_dim=512, hidden_dim=128, num_chords=25)

    # Get dataset
    dataset = get_dataset('lab_and_audio_files', 'lab_and_audio_files', 'chord_dataset_full.npz')

    # Get data loaders
    train_loader, val_loader = get_data_loaders(dataset)

    # Train model
    model = train(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()

