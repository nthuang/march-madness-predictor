import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler


class MMDataset(Dataset):
    def __init__(self, annotations_file, years=None, scaler=None, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file)

        # If specific years are declared for testing/training, the data is truncated to only include those years.
        if years is not None:
            self.data = self.data[self.data['Season'].isin(years)].reset_index(drop=True)

        self.labels = self.data['w'].values
        
        # Raw feature data is without scaling, and just takes the raw CSV values and drops specific columns
        # that the model shouldn't be able to see. These include who won and the win margin, which
        # would cause immediate overfitting if seen by the model.
        features_raw = self.data.drop(columns=['w', 'Season', 'Team1', 'Team2', 'margin', 'T1_SeedN', 'T2_SeedN', 'Diff_SeedN', 'T1_HasSeed', 'T2_HasSeed', 'Diff_HasSeed'], errors='ignore').values
        
        self.transform = transform
        self.target_transform = target_transform

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_raw)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features_raw)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_row = self.features[idx]
        label = self.labels[idx]

        feature_tensor = torch.tensor(feature_row, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
            
        return feature_tensor, label_tensor

## all_years = pd.read_csv('W_2026_train_aug.csv')['Season'].unique()
## train_years = [y for y in all_years if y < 2024]
## test_years = [2025]

## train_data = MMDataset('W_2026_train_aug.csv', years=train_years)

## test_data = MMDataset('W_2026_train_aug.csv', years=test_years, scaler=train_data.scaler)

all_seasons = pd.read_csv('W_2026_train_aug.csv')['Season'].unique()
test_seasons = [y for y in sorted(all_seasons) if y >= 2018]
results_list = []

for held_out_year in test_seasons:
    print(f"\nSeason: {held_out_year}")

    train_years = [y for y in all_seasons if y != held_out_year]
    test_years = [held_out_year]

    train_data = MMDataset('W_2026_train_aug.csv', years=train_years)
    test_data = MMDataset('W_2026_train_aug.csv', years=test_years, scaler=train_data.scaler)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = MMModel().to(device) 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    epochs = 50
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)

    model.eval()
    test_loss, test_brier, correct = 0, 0, 0
    size = len(test_data)
    
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            probs = torch.sigmoid(pred)
            test_brier += torch.sum((probs - y)**2).item()
            predictions = (probs >= 0.5).float()
            correct += (predictions == y).sum().item()

    fold_metrics = {
        'Season': held_out_year,
        'Loss': test_loss / len(test_dataloader),
        'Brier': test_brier / size,
        'Accuracy': (correct / size) * 100
    }
    results_list.append(fold_metrics)
    print(f"Season {held_out_year} Result: {fold_metrics['Accuracy']:.2f}% Accuracy")

results_df = pd.DataFrame(results_list)
print("\n" + "="*30)
print("FINAL LOSO AVG RESULTS")
print(results_df[['Loss', 'Brier', 'Accuracy']].mean())
print("="*30)

from torch import nn

device = device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class MMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(54, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = MMModel().to(device)
print(model)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

from torch.utils.data import DataLoader

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad() 
        loss.backward()      
        optimizer.step()    

        ##if batch % 10 == 0:
           ## print(f"loss: {loss.item():>7f}")

        # pred_prob = torch.sigmoid(pred)
        # predictions = (pred_prob > 0.5).float()
        # correct = (predictions == y).sum().item()
        # accuracy = correct / len(y)
        # print(f"Accuracy: {accuracy:>7f}")

epochs = 50
for t in range(epochs):
    print(f"Epoch: {t+1}\n")
    train(train_dataloader, model, loss_fn, optimizer)

model.eval()
test_loss, test_brier, correct = 0, 0, 0
num_batches = len(test_dataloader)
size = len(test_dataloader.dataset)

with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        
        test_loss += loss_fn(pred, y).item()
        
        probs = torch.sigmoid(pred)
    
        test_brier += torch.sum((probs - y)**2).item()
        
        predictions = (probs >= 0.5).float()
        correct += (predictions == y).type(torch.float).sum().item()

avg_loss = test_loss / num_batches
avg_brier = test_brier / size
accuracy = (correct / size) * 100

print(f"Results: \n Avg Loss: {avg_loss:>8f} \n Brier Score: {avg_brier:>8f} \n Accuracy: {accuracy:>2f}%")

df = pd.read_csv('W_2026_train_aug.csv')

df['favorite_won'] = ((df['T1_SeedN'] < df['T2_SeedN']) & (df['w'] == 1)) | \
                     ((df['T2_SeedN'] < df['T1_SeedN']) & (df['w'] == 0))

seed_accuracy = df.groupby('Season')['favorite_won'].mean()
print(seed_accuracy)