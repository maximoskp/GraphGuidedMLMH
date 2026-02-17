from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class LSTMHarmonyModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T)
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)  # (B, T, vocab)
        return logits, hidden
# end LSTMHarmonyModel

def train_lstm(
        model,
        trainloader,
        valloader,
        optimizer,
        device,
        save_path,
        num_epochs=10):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        batch_num = 0
        with tqdm(trainloader, unit='batch', position=0) as tepoch:
            tepoch.set_description(f'Epoch {epoch}| trn')
            for batch in tepoch:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                # attention_mask = batch["attention_mask"].to(device)
                optimizer.zero_grad()
                logits, _ = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
                loss.backward()
                optimizer.step()

                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                tepoch.set_postfix(loss=train_loss)
        # validation loop
        model.eval()
        with torch.no_grad():
            running_loss = 0
            batch_num = 0
            with tqdm(valloader, unit='batch', position=0) as tepoch:
                tepoch.set_description(f'Epoch {epoch}| val')
                for batch in tepoch:
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    # attention_mask = batch["attention_mask"].to(device)
                    logits, _ = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1)
                    )
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    tepoch.set_postfix(loss=val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('saving...')
            torch.save(model.state_dict(), save_path)
# end train_lstm

class TransitionMatrixAutoencoder(nn.Module):
    def __init__(self, D, latent_dim=64):
        super().__init__()

        input_dim = D * D

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        self.D = D

    def forward(self, matrix):
        # matrix: (B, D, D)
        B = matrix.size(0)
        x = matrix.view(B, -1)

        z = self.encoder(x)
        recon = self.decoder(z)

        recon = recon.view(B, self.D, self.D)
        return recon, z
# end TransitionMatrixAutoencoder

def train_matrix_ae(
        model,
        trainloader,
        valloader,
        optimizer,
        device,
        save_path,
        num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        batch_num = 0
        with tqdm(trainloader, unit='batch', position=0) as tepoch:
            tepoch.set_description(f'Epoch {epoch}| trn')
            for batch in tepoch:
                matrix = batch["transition_matrix"].to(device)
                optimizer.zero_grad()
                recon, _ = model(matrix)
                loss = F.kl_div(
                    F.log_softmax(recon, dim=-1),
                    matrix,
                    reduction='batchmean'
                )
                loss.backward()
                optimizer.step()

                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                tepoch.set_postfix(loss=train_loss)
        # validation loop
        model.eval()
        with torch.no_grad():
            running_loss = 0
            batch_num = 0
            with tqdm(valloader, unit='batch', position=0) as tepoch:
                tepoch.set_description(f'Epoch {epoch}| val')
                for batch in tepoch:
                    matrix = batch["transition_matrix"].to(device)
                    recon, _ = model(matrix)
                    loss = F.kl_div(
                        F.log_softmax(recon, dim=-1),
                        matrix,
                        reduction='batchmean'
                    )
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    tepoch.set_postfix(loss=val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('saving...')
            torch.save(model.state_dict(), save_path)
# end train_matrix_ae


class BagOfTransitionsAutoencoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )

    def forward(self, bow):
        # bow: (B, V)
        z = self.encoder(bow)
        recon_logits = self.decoder(z)
        return recon_logits, z
# end BagOfTransitionsAutoencoder

def train_bot_ae(model,
            trainloader,
            valloader,
            optimizer,
            device,
            save_path,
            num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        batch_num = 0
        with tqdm(trainloader, unit='batch', position=0) as tepoch:
            tepoch.set_description(f'Epoch {epoch}| trn')
            for batch in tepoch:
                bow = batch["bag_of_transitions"].to(device)
                optimizer.zero_grad()
                recon_logits, _ = model(bow)
                loss = F.kl_div(
                    F.log_softmax(recon_logits, dim=-1),
                    bow,
                    reduction='batchmean'
                )
                loss.backward()
                optimizer.step()

                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                tepoch.set_postfix(loss=train_loss)
        # validation loop
        model.eval()
        with torch.no_grad():
            running_loss = 0
            batch_num = 0
            with tqdm(valloader, unit='batch', position=0) as tepoch:
                tepoch.set_description(f'Epoch {epoch}| val')
                for batch in tepoch:
                    bow = batch["bag_of_transitions"].to(device)
                    recon_logits, _ = model(bow)
                    loss = F.kl_div(
                        F.log_softmax(recon_logits, dim=-1),
                        bow,
                        reduction='batchmean'
                    )
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    tepoch.set_postfix(loss=val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('saving...')
            torch.save(model.state_dict(), save_path)
# end train_bow_ae