import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMHarmonyModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
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

def train_lstm(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        seq = batch["chord_sequence"].to(device)  # (B, T)

        input_seq = seq[:, :-1]
        target_seq = seq[:, 1:]

        optimizer.zero_grad()

        logits, _ = model(input_seq)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
# end train_lstm

class TransitionMatrixAutoencoder(nn.Module):
    def __init__(self, D, latent_dim=256):
        super().__init__()

        input_dim = D * D

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
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

def train_matrix_ae(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        matrix = batch["transition_matrix"].to(device)

        optimizer.zero_grad()
        recon, _ = model(matrix)

        # loss = F.mse_loss(recon, matrix)
        loss = F.kl_div(
            F.log_softmax(recon, dim=-1),
            matrix,
            reduction='batchmean'
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
# end train_matrix_ae


class BagOfTransitionsAutoencoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)
        )

    def forward(self, bow):
        # bow: (B, V)
        z = self.encoder(bow)
        recon_logits = self.decoder(z)
        return recon_logits, z
# end BagOfTransitionsAutoencoder

def train_bow_ae(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
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

        total_loss += loss.item()

    return total_loss / len(dataloader)
# end train_bow_ae