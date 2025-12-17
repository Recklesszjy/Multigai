"""
This Python script provides a unified implementation of multimodal integration and mapping/imputation
in the MultiGAI framework.
"""

import random
import numpy as np
import torch
from torch.distributions import Distribution, Normal, constraints
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import issparse
from torch.optim import Adam
import math
from tqdm import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ZeroInflatedNegativeBinomial(Distribution):
    """
    Zero-Inflated Negative Binomial (ZINB) distribution.

    This distribution is commonly used to model over-dispersed count data
    with excessive zeros, such as scRNA-seq gene expression counts.
    """

    # Constraints on distribution parameters
    arg_constraints = {
        "mu": constraints.greater_than_eq(0),      # Mean of the NB distribution
        "theta": constraints.greater_than_eq(0),   # Inverse dispersion parameter
        "zi_logits": constraints.real,              # Logits for zero-inflation probability
        "scale": constraints.greater_than_eq(0),   # Optional scaling factor (e.g. library size)
    }

    # Support of the distribution: non-negative integers
    support = constraints.nonnegative_integer

    def __init__(self, mu, theta, zi_logits, scale, eps=1e-8, validate_args=False):
        """
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the Negative Binomial distribution.
        theta : torch.Tensor
            Inverse dispersion parameter of the NB distribution.
        zi_logits : torch.Tensor
            Logits controlling the zero-inflation probability.
        scale : torch.Tensor
            Scaling factor applied to the mean (e.g. size factor).
        eps : float
            Small constant for numerical stability.
        validate_args : bool
            Whether to validate distribution arguments.
        """
        self.mu = mu
        self.theta = theta
        self.zi_logits = zi_logits
        self.scale = scale 
        self.eps = eps

        # Initialize base Distribution class
        super().__init__(validate_args=validate_args)

    def log_prob(self, x):
        """
        Compute log-probability of observed counts under ZINB.

        Parameters
        ----------
        x : torch.Tensor
            Observed count data.

        Returns
        -------
        torch.Tensor
            Log-likelihood of each observation.
        """

        # Convert zero-inflation logits to probability
        pi = torch.sigmoid(self.zi_logits)

        # Log-probability under the Negative Binomial distribution
        log_nb = (
            torch.lgamma(x + self.theta)
            - torch.lgamma(self.theta)
            - torch.lgamma(x + 1)
            + self.theta * torch.log(self.theta + self.eps)
            + x * torch.log(self.mu + self.eps)
            - (x + self.theta) * torch.log(self.mu + self.theta + self.eps)
        )

        # Zero-inflated mixture:
        # - If x == 0: mixture of structural zero and NB zero
        # - If x > 0: NB probability scaled by (1 - pi)
        log_prob = torch.where(
            (x == 0),
            torch.log(pi + (1 - pi) * torch.exp(log_nb) + self.eps),
            torch.log(1 - pi + self.eps) + log_nb,
        )

        return log_prob

class multigai(nn.Module):
    """
    MultiGAI: A multi-modal generative integration model.

    This model supports joint representation learning and cross-modality
    reconstruction using attention-based latent fusion and ZINB decoders.
    """

    def __init__(
        self,
        input_dim1,
        input_dim2,
        input_dim3,
        n_hidden,
        hidden,
        z_dim,
        batch_dim,
        q_dim=128,
        kv_n=64,
        dropout_rate=0.1
    ):
        super().__init__()

        # ===== Hyperparameters =====
        self.kv_n = kv_n              # Number of key/value tokens
        self.q_dim = q_dim            # Query embedding dimension
        self.z_dim = z_dim            # Latent space dimension
        self.batch_dim = batch_dim    # Batch covariate dimension
        self.hidden = hidden          # Hidden layer width

        # ===== Shared encoder constructor =====
        def make_encoder(in_dim):
            """
            Build a multi-layer MLP encoder with LayerNorm and Dropout.
            """
            layers = []
            for _ in range(n_hidden):
                layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.LayerNorm(hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    )
                )
                in_dim = hidden
            return nn.Sequential(*layers)

        # ===== Modality-specific encoders =====
        self.encoder1 = make_encoder(input_dim1)
        self.encoder2 = make_encoder(input_dim2)
        self.encoder3 = make_encoder(input_dim3)

        # ===== Projection to query space =====
        self.q_net1 = nn.Linear(hidden, q_dim)
        self.q_net2 = nn.Linear(hidden, q_dim)
        self.q_net3 = nn.Linear(hidden, q_dim)

        # ===== Key / Value network constructor =====
        def make_kv(is_value):
            """
            Build key/value networks for attention-based latent fusion.
            """
            layers = []
            in_dim = kv_n
            for _ in range(n_hidden):
                layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.LayerNorm(hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    )
                )
                in_dim = hidden
            if not is_value:
                layers.append(nn.Linear(hidden, q_dim))
            return nn.Sequential(*layers)

        # ===== Modality-specific key/value banks =====
        self.keys1 = make_kv(is_value=False)
        self.values1 = make_kv(is_value=True)
        self.keys2 = make_kv(is_value=False)
        self.values2 = make_kv(is_value=True)
        self.keys3 = make_kv(is_value=False)
        self.values3 = make_kv(is_value=True)

        # ===== Shared key/value banks =====
        self.keys = make_kv(is_value=False)
        self.values = make_kv(is_value=True)

        # ===== Latent Gaussian parameter heads =====
        self.m_net = nn.Linear(hidden, z_dim)   # Mean of q(z|x)
        self.l_net = nn.Linear(hidden, z_dim)   # Log-variance of q(z|x)

        # ===== Shared decoder backbone =====
        self.decoder_base = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim + batch_dim if i == 0 else hidden + batch_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(n_hidden)
        ])

        # ===== ZINB decoders for each modality =====
        # Modality 1
        self.fc_scale1 = nn.Sequential(
            nn.Linear(hidden + batch_dim, input_dim1),
            nn.Softmax(dim=-1)
        )
        self.fc_dropout1 = nn.Linear(hidden + batch_dim, input_dim1)
        self.fc_r1 = nn.Parameter(torch.randn(input_dim1))  # Dispersion

        # Modality 2
        self.fc_scale2 = nn.Sequential(
            nn.Linear(hidden + batch_dim, input_dim2),
            nn.Softmax(dim=-1)
        )
        self.fc_dropout2 = nn.Linear(hidden + batch_dim, input_dim2)
        self.fc_r2 = nn.Parameter(torch.randn(input_dim2))

        # Modality 3
        self.fc_scale3 = nn.Sequential(
            nn.Linear(hidden + batch_dim, input_dim3),
            nn.Softmax(dim=-1)
        )
        self.fc_dropout3 = nn.Linear(hidden + batch_dim, input_dim3)
        self.fc_r3 = nn.Parameter(torch.randn(input_dim3))

    def compute_mu_var(self, device, q1=None, q2=None, q3=None, m=None):
        """
        Compute latent mean and variance using attention-based fusion
        of modality-specific query embeddings.
        """
        I = torch.eye(self.kv_n, device=device)   # Identity tokens
        scale = math.sqrt(self.q_dim)

        attn1 = attn2 = attn3 = None

        # ===== Modality-pair attention fusion =====
        if m == 12:
            ker1, val1 = self.keys1(I), self.values1(I)
            ker2, val2 = self.keys2(I), self.values2(I)

            attn1 = torch.softmax((q1 @ ker1.T) / scale, dim=-1) @ val1
            attn2 = torch.softmax((q2 @ ker2.T) / scale, dim=-1) @ val2

            ae = (attn1 + attn2) / 2.0

        elif m == 13:
            ker1, val1 = self.keys1(I), self.values1(I)
            ker3, val3 = self.keys3(I), self.values3(I)

            attn1 = torch.softmax((q1 @ ker1.T) / scale, dim=-1) @ val1
            attn3 = torch.softmax((q3 @ ker3.T) / scale, dim=-1) @ val3

            ae = (attn1 + attn3) / 2.0
        else:
            raise ValueError(f"Unsupported modality m={m}")

        # ===== Shared attention refinement =====
        ker, val = self.keys(I), self.values(I)
        attn = torch.softmax((ae @ ker.T) / scale, dim=-1) @ val

        # ===== Latent Gaussian parameters =====
        mu = self.m_net(attn)
        logvar = self.l_net(attn)
        var = torch.exp(logvar) + 1e-8

        return mu, var, attn1, attn2, attn3, attn

    def decode_from_z(self, dz, m1, m2, m3, m, batch):
        """
        Decode latent variables into modality-specific ZINB distributions.
        """
        for layer in self.decoder_base:
            dz = torch.cat([dz, batch], dim=1)
            dz = layer(dz)

        final = torch.cat([dz, batch], dim=1)

        p1 = p2 = p3 = None

        if m in [12, 13]:
            # ===== Modality 1 =====
            scale1 = self.fc_scale1(final)
            dropout1 = self.fc_dropout1(final)
            library1 = torch.log(m1.sum(1, keepdim=True) + 1e-8)
            rate1 = torch.exp(library1) * scale1

            p1 = ZeroInflatedNegativeBinomial(
                mu=rate1,
                theta=torch.exp(self.fc_r1),
                zi_logits=dropout1,
                scale=scale1
            )

            # ===== Modality 2 =====
            if m == 12:
                scale2 = self.fc_scale2(final)
                dropout2 = self.fc_dropout2(final)
                library2 = torch.log(m2.sum(1, keepdim=True) + 1e-8)
                rate2 = torch.exp(library2) * scale2

                p2 = ZeroInflatedNegativeBinomial(
                    mu=rate2,
                    theta=torch.exp(self.fc_r2),
                    zi_logits=dropout2,
                    scale=scale2
                )

            # ===== Modality 3 =====
            if m == 13:
                scale3 = self.fc_scale3(final)
                dropout3 = self.fc_dropout3(final)
                library3 = torch.log(m3.sum(1, keepdim=True) + 1e-8)
                rate3 = torch.exp(library3) * scale3

                p3 = ZeroInflatedNegativeBinomial(
                    mu=rate3,
                    theta=torch.exp(self.fc_r3),
                    zi_logits=dropout3,
                    scale=scale3
                )

        return p1, p2, p3

    def forward(self, m1, m2, m3, m, batch):
        """
        Forward pass of MultiGAI.
        """
        device = batch.device
        batch_size = m1.size(0)

        # ===== Encode =====
        q1 = q2 = q3 = None
        if m in [12, 13]:
            e1 = self.encoder1(m1)
            q1 = self.q_net1(e1)

            if m == 12:
                e2 = self.encoder2(m2)
                q2 = self.q_net2(e2)

            if m == 13:
                e3 = self.encoder3(m3)
                q3 = self.q_net3(e3)

        # ===== Latent inference =====
        mu, var, a1, a2, a3, ae = self.compute_mu_var(device, q1, q2, q3, m)
        var = torch.clamp(var, min=1e-6)

        qz = Normal(mu, var.sqrt())
        z = qz.rsample()
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        # ===== Decode =====
        p1, p2, p3 = self.decode_from_z(z, m1, m2, m3, m, batch)

        return z, p1, p2, p3, qz, pz, a1, a2, a3, ae

    def loss_function(self, m1, m2, m3, m, p1, p2, p3, q, p, a1, a2, a3, w):
        """
        Compute total loss:
        reconstruction + KL divergence + cosine alignment loss.
        """
        device = m1.device
        cos_loss = torch.tensor(0.0, device=device)

        if m == 12:
            reconst_loss = (
                -p1.log_prob(m1).sum(-1).mean()
                -p2.log_prob(m2).sum(-1).mean()
            )
            cos_loss = (1 - F.cosine_similarity(a1, a2, dim=1)).mean()

        elif m == 13:
            reconst_loss = (
                -p1.log_prob(m1).sum(-1).mean()
                -p3.log_prob(m3).sum(-1).mean()
            )
            cos_loss = (1 - F.cosine_similarity(a1, a3, dim=1)).mean()
        else:
            raise ValueError(f"Unsupported modality m={m}")

        kl = torch.distributions.kl_divergence(q, p).sum(dim=-1).mean()

        loss = reconst_loss + w * kl + cos_loss
        return loss, reconst_loss, kl, cos_loss

class M2L(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout_p=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),        
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(p=dropout_p),        
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MultiOmicsDataset(Dataset):
    """
    PyTorch Dataset for multi-omics data.

    Each sample contains:
    - Three modality feature vectors (m1, m2, m3)
    - A modality indicator m
    - A batch covariate
    """

    def __init__(self, *args):
        self.m1_data = args[0]     # Modality 1 data (e.g., scRNA-seq)
        self.m2_data = args[1]     # Modality 2 data (e.g., scATAC-seq)
        self.m3_data = args[2]     # Modality 3 data (e.g., ADT)
        self.m_data = args[3]      # Modality indicator (12 or 13)
        self.batch_data = args[4]  # Batch labels / covariates

    def __len__(self):
        # Number of samples
        return len(self.batch_data)

    def __getitem__(self, idx):
        # Convert each modality to float tensor
        m1 = torch.tensor(self.m1_data[idx], dtype=torch.float32).squeeze(0)
        m2 = torch.tensor(self.m2_data[idx], dtype=torch.float32).squeeze(0)
        m3 = torch.tensor(self.m3_data[idx], dtype=torch.float32).squeeze(0)

        # Modality combination indicator
        m = torch.tensor(self.m_data[idx], dtype=torch.float32).squeeze(0)

        # Batch covariate
        batch = self.batch_data[idx]

        return m1, m2, m3, m, batch, idx   
    
def rnaatacmapping(output_path, adata, *args, num_epochs=200):

    """
    Functionality:
    --------------
    Train a latent space using Multiome modality (joint RNA + ATAC) with the MultiGAI model,
    then map RNA and ATAC modalities into this latent space. The resulting latent representations
    can be used for downstream analyses or imputation.

    Parameters:
    -----------
    output_path : str
        Path to save the resulting h5ad file.
    adata : AnnData
        Input AnnData object containing RNA and ATAC data; also serves as the container
        to store the computed latent variables.
    *args : tuple
        Model parameters, including:
        n_hidden : int
            Number of hidden layers for all neural network components (encoder & decoder)
        hidden : int
            Hidden dimension for all layers in the network
        z_dim : int
            Dimension of the latent space
        q_dim : int
            Dimension of the query vector (Q) in attention
        kv_n : int
            Number of key-value (K-V) pairs in attention
    num_epochs : int, optional
        Number of training epochs (default: 200)

    Returns:
    --------
    None
        Results (latent variables) are stored in adata.obsm and written to the specified h5ad file.
    """
    
    # Unpack model parameters
    n_hidden, hidden, z_dim, q_dim, kv_n = args

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Separate RNA and ATAC data
    rna = adata[:, adata.var['feature_types'] == 'GEX'].copy()
    atac = adata[:, adata.var['feature_types'] == 'ATAC'].copy()

    # Use paired cells as training set
    rna_t  = rna[rna.obs_names[rna.obs["Modality"] == "multiome"]].copy()
    atac_t = atac[atac.obs_names[atac.obs["Modality"] == "multiome"]].copy()
    z1_index = rna_t.obs_names.values 

    # Get count matrices
    rna_t_d, atac_t_d = rna_t.layers['counts'], atac_t.layers['counts']
    adt_t_d = np.zeros((rna_t_d.shape[0], 1))  # Initialize ADT data as zeros
    rna_dim, atac_dim, adt_dim = rna.shape[1], atac.shape[1], 1

    # Convert sparse matrix to dense if necessary
    if issparse(rna_t_d): rna_t_d = rna_t_d.toarray()
    if issparse(atac_t_d): atac_t_d = atac_t_d.toarray()
    if issparse(adt_t_d): adt_t_d = adt_t_d.toarray()

    # Map modality to numeric vector
    modality_map = {'multiome': 12, 'cite': 13}
    modality_vector = rna_t.obs['Modality'].map(modality_map)
    modality_d = modality_vector.to_numpy().astype(float)

    # One-hot encode batch
    batch_indices = torch.from_numpy(rna_t.obs['batch'].astype('category').cat.codes.values).long()
    batch_encoded = torch.nn.functional.one_hot(batch_indices)
    batch_dim = batch_encoded.shape[1]

    # Construct training dataset
    dataset_t = MultiOmicsDataset(rna_t_d, atac_t_d, adt_t_d, modality_d, batch_encoded)
    train_loader = DataLoader(dataset_t, batch_size=512, shuffle=True)
    test1_loader = DataLoader(dataset_t, batch_size=512, shuffle=False)

    # === Initialize model ===
    model = multigai(rna_dim, atac_dim, adt_dim, n_hidden, hidden, z_dim, batch_encoded.shape[1], q_dim, kv_n).to(device)
    optimizer_main = Adam(model.parameters(), lr=0.001)
    scheduler_main = torch.optim.lr_scheduler.StepLR(optimizer_main, step_size=50, gamma=0.9)

    # === Train main model ===
    for epoch in range(num_epochs):
        running_loss = 0.0
        kl_weight = 0.0 if epoch < 100 else 0.1  # KL loss not weighted for first 100 epochs
        model.train()
        for batch_data in train_loader:
            optimizer_main.zero_grad()
            m_values = batch_data[3]  # modality labels
            unique_m = m_values.unique()  # all modalities in this batch

            # Split sub-batch by modality
            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, m_tensor, batch_tensor, idx = [x.to(device) for x in sub_batch]

                    # Forward pass
                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3, int(m_curr.item()), batch_tensor
                    )

                    # Compute loss
                    loss, reconst_loss, kl_loss, cos_loss = model.loss_function(
                        m1, m2, m3, int(m_curr.item()), p1, p2, p3, qz, pz, a1, a2, a3, kl_weight
                    )
                    loss.backward()
                    optimizer_main.step()
                    running_loss += loss.item()

        scheduler_main.step()
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, KL_weight: {kl_weight}")

    # === Extract latent embeddings ===
    model.eval()
    z1_list, a1_list, a2_list, ae_list = [], [], [], []
    with torch.no_grad():
        for batch_data in test1_loader:
            m_values = batch_data[3]
            unique_m = m_values.unique()
            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, _, batch_tensor, idx = [x.to(device) for x in sub_batch]
                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3, int(m_curr.item()), batch_tensor
                    )
            z1_list.append(z)
            a1_list.append(a1)
            a2_list.append(a2)
            ae_list.append(ae)

    # Concatenate all batches
    z1_tensor  = torch.cat(z1_list, dim=0)
    a1_tensor  = torch.cat(a1_list, dim=0)
    a2_tensor  = torch.cat(a2_list, dim=0)
    ae_tensor  = torch.cat(ae_list, dim=0)

    # === MLP distillation ===
    def train_mlp(x_tensor, y_tensor, input_dim, output_dim, epochs=200):
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        mlp = M2L(input_dim, output_dim).to(device)
        optimizer = Adam(mlp.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        mlp.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                y_pred = mlp(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch+1) % 50 == 0:
                print(f"[MLP Epoch {epoch+1}/{epochs}] Loss: {running_loss/len(loader):.4f}")
        return mlp

    # Train MLP to map each modality's intermediate embedding to joint embedding
    mlp1 = train_mlp(a1_tensor, ae_tensor, a1_tensor.shape[1], ae_tensor.shape[1])
    mlp2 = train_mlp(a2_tensor, ae_tensor, a2_tensor.shape[1], ae_tensor.shape[1])

    def infer_mlp(mlp, x_tensor):
        # Inference in batches
        batch_size = 512
        preds = []
        mlp.eval()
        with torch.no_grad():
            for i in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[i:i+batch_size]
                preds.append(mlp(batch))
        return torch.cat(preds, dim=0)

    # Prepare RNA tensor
    gene_mask = adata.var['feature_types'] == 'GEX'
    rna_cells = adata.obs['Modality'] == 'rna'
    x_gene_np = adata[rna_cells, :][:, gene_mask].layers['counts']
    x_gene_tensor = torch.tensor(x_gene_np.toarray() if issparse(x_gene_np) else x_gene_np,
                                 dtype=torch.float32, device=device)
    z2_index = adata[rna_cells, :].obs_names.values  # RNA cell names

    # Prepare ATAC tensor
    peaks_mask = adata.var['feature_types'] == 'ATAC'
    atac_cells = adata.obs['Modality'] == 'atac'
    x_peaks_np = adata[atac_cells, :][:, peaks_mask].layers['counts']
    x_peaks_tensor = torch.tensor(x_peaks_np.toarray() if issparse(x_peaks_np) else x_peaks_np,
                                  dtype=torch.float32, device=device)
    z3_index = adata[atac_cells, :].obs_names.values  # ATAC cell names

    # Map x1 → a1, x2 → a2
    def x_to_a1_a2(x_gene_tensor, x_peaks_tensor, model):
        batch_size = 512
        a1_list, a2_list = [], []
        I = torch.eye(model.kv_n, device=x_gene_tensor.device)
        scale = math.sqrt(model.q_dim)
        model.eval()
        with torch.no_grad():
            # RNA
            for i in range(0, x_gene_tensor.shape[0], batch_size):
                batch = x_gene_tensor[i:i+batch_size]
                e1 = model.encoder1(batch)
                q1 = model.q_net1(e1)
                ker = model.keys1(I)
                var = model.values1(I)
                attn_logits1 = (q1 @ ker.T) / scale
                attn_weights1 = torch.softmax(attn_logits1, dim=-1)
                a1_list.append(attn_weights1 @ var)
            # ATAC
            for i in range(0, x_peaks_tensor.shape[0], batch_size):
                batch = x_peaks_tensor[i:i+batch_size]
                e2 = model.encoder2(batch)
                q2 = model.q_net2(e2)
                kea = model.keys2(I)
                vaa = model.values2(I)
                attn_logits2 = (q2 @ kea.T) / scale
                attn_weights2 = torch.softmax(attn_logits2, dim=-1)
                a2_list.append(attn_weights2 @ vaa)
        a1 = torch.cat(a1_list, dim=0)
        a2 = torch.cat(a2_list, dim=0)
        return a1, a2
     
    a1, a2 = x_to_a1_a2(x_gene_tensor, x_peaks_tensor, model)
    e1_tensor = infer_mlp(mlp1, a1)
    e2_tensor = infer_mlp(mlp2, a2)

    # Map e → z
    def e_to_z(e_tensor):
        batch_size = 512
        z_list = []
        with torch.no_grad():
            for i in range(0, e_tensor.shape[0], batch_size):
                batch = e_tensor[i:i+batch_size]
                mu = model.m_net(batch)
                logvar = model.l_net(batch)
                var = torch.exp(logvar) + 1e-8
                qz = Normal(mu, var.sqrt())
                z_list.append(qz.rsample())
        return torch.cat(z_list, dim=0)

    z2_tensor = e_to_z(e1_tensor)
    z3_tensor = e_to_z(e2_tensor)

    latent_dim = z1_tensor.shape[1]
    latent_matrix = torch.full((adata.n_obs, latent_dim), float('nan'), device=device)

    # Build cell_name -> row index mapping
    adata_index = adata.obs_names.values
    adata_mapping = {cell: i for i, cell in enumerate(adata_index)}

    # Fill latent matrix
    def fill_latent(tensor, tensor_index):
        idx = [adata_mapping[cell] for cell in tensor_index]
        latent_matrix[idx, :] = tensor

    fill_latent(z1_tensor, z1_index) # Multiome
    fill_latent(z2_tensor, z2_index) # RNA
    fill_latent(z3_tensor, z3_index) # ATAC

    # Save latent embeddings to adata.obsm
    adata.obsm['latent'] = latent_matrix.cpu().numpy()
    adata.write_h5ad(output_path)

def rnaadtmapping(output_path, adata, *args, num_epochs=200):
    
    """
    Functionality:
    --------------
    Train a latent space using CITE modality (ADT + RNA) with MultiGAI model,
    then map RNA and ADT modalities into this latent space. The resulting latent
    representations can be used for downstream analyses or imputation.

    Parameters:
    -----------
    output_path : str
        Path to save the resulting h5ad file.
    adata : AnnData
        Input AnnData object containing RNA and ADT data; also serves as the
        container to store the computed latent variables.
    *args : tuple
        Model parameters, including:
        n_hidden : int
            Number of hidden layers for all neural network components (encoder & decoder)
        hidden : int
            Hidden dimension for all layers in the network
        z_dim : int
            Dimension of the latent space
        q_dim : int
            Dimension of the query vector (Q) in attention
        kv_n : int
            Number of key-value (K-V) pairs in attention
    num_epochs : int, optional
        Number of training epochs (default: 200)

    Returns:
    --------
    None
        Results (latent variables) are stored in adata.obsm and written to the specified h5ad file.
    """

    # multigai parameters
    n_hidden, hidden, z_dim, q_dim, kv_n = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split RNA and ADT modalities
    rna = adata[:, adata.var['feature_types'] == 'GEX'].copy()
    adt = adata[:, adata.var['feature_types'] == 'ADT'].copy()

    # Select CITE cells for training
    rna_t  = rna[rna.obs_names[rna.obs["Modality"] == "cite"]].copy()
    adt_t = adt[adt.obs_names[adt.obs["Modality"] == "cite"]].copy()

    z1_index = rna_t.obs_names.values

    # Extract raw count matrices
    rna_t_d, adt_t_d = rna_t.layers['counts'], adt_t.layers['counts']
    atac_t_d = np.zeros((rna_t_d.shape[0], 1))  # placeholder for unused modality
    rna_dim, atac_dim, adt_dim = rna.shape[1], 1, adt.shape[1]
    if issparse(rna_t_d): rna_t_d = rna_t_d.toarray()
    if issparse(atac_t_d): atac_t_d = atac_t_d.toarray()
    if issparse(adt_t_d): adt_t_d = adt_t_d.toarray() 

    # Map modality to numeric IDs
    modality_map = {
        'multiome': 12,
        'cite': 13,
    }
    modality_vector = rna_t.obs['Modality'].map(modality_map)
    modality_d = modality_vector.to_numpy().astype(float)

    # One-hot encode batch information
    batch_indices = torch.from_numpy(rna_t.obs['batch'].astype('category').cat.codes.values).long()
    batch_encoded = torch.nn.functional.one_hot(batch_indices)
    batch_dim = batch_encoded.shape[1]

    # Create dataset and data loaders
    dataset_t = MultiOmicsDataset(rna_t_d, atac_t_d, adt_t_d, modality_d, batch_encoded)
    train_loader = DataLoader(dataset_t, batch_size=512, shuffle=True)
    test1_loader = DataLoader(dataset_t, batch_size=512, shuffle=False)

    # Initialize the model
    model = multigai(rna_dim, atac_dim, adt_dim, n_hidden, hidden, z_dim, batch_encoded.shape[1], q_dim, kv_n).to(device)
    optimizer_main = Adam(model.parameters(), lr=0.001)
    scheduler_main = torch.optim.lr_scheduler.StepLR(optimizer_main, step_size=50, gamma=0.9)

    # Train the main model
    for epoch in range(num_epochs):
        running_loss = 0.0
        kl_weight = 0.0 if epoch < 100 else 0.1
        model.train()
        for batch_data in train_loader:
            optimizer_main.zero_grad()

            m_values = batch_data[3]  # modality indicator
            unique_m = m_values.unique()

            # Split batch by modality
            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, m_tensor, batch_tensor, idx = [x.to(device) for x in sub_batch]

                    # Forward pass
                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3, int(m_curr.item()), batch_tensor
                    )
                    # Compute loss
                    loss, reconst_loss, kl_loss, cos_loss = model.loss_function(
                        m1, m2, m3, int(m_curr.item()), p1, p2, p3, qz, pz, a1, a2, a3, kl_weight
                    )
                    loss.backward()
                    optimizer_main.step()

                    running_loss += loss.item()

        scheduler_main.step()
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, KL_weight: {kl_weight}")

    model.eval()
    z1_list, a1_list, a3_list, ae_list = [], [], [], []

    # Extract embeddings for CITE cells
    with torch.no_grad():
        for batch_data in test1_loader:
            m_values = batch_data[3]
            unique_m = m_values.unique()

            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, _, batch_tensor, idx = [x.to(device) for x in sub_batch]

                    # Forward pass
                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3, int(m_curr.item()), batch_tensor
                    )
            
            # Collect outputs
            z1_list.append(z)
            a1_list.append(a1)
            a3_list.append(a3)
            ae_list.append(ae)

    # Concatenate into a single tensor
    z1_tensor  = torch.cat(z1_list, dim=0)
    a1_tensor  = torch.cat(a1_list, dim=0)
    a3_tensor  = torch.cat(a3_list, dim=0)
    ae_tensor  = torch.cat(ae_list, dim=0)

    # Train MLPs
    def train_mlp(x_tensor, y_tensor, input_dim, output_dim, epochs=200):
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        mlp = M2L(input_dim, output_dim).to(device)
        optimizer = Adam(mlp.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        mlp.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                y_pred = mlp(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch+1) % 50 == 0:
                print(f"[MLP Epoch {epoch+1}/{epochs}] Loss: {running_loss/len(loader):.4f}")
        return mlp

    # Train MLP to map intermediate embeddings to joint latent
    mlp1 = train_mlp(a1_tensor, ae_tensor, a1_tensor.shape[1], ae_tensor.shape[1])  # RNA
    mlp2 = train_mlp(a3_tensor, ae_tensor, a3_tensor.shape[1], ae_tensor.shape[1])  # ADT

    def infer_mlp(mlp, x_tensor):
        batch_size = 512
        preds = []
        mlp.eval()
        with torch.no_grad():
            for i in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[i:i+batch_size]
                preds.append(mlp(batch))
        return torch.cat(preds, dim=0)

    # Prepare RNA and ADT data for inference
    gene_mask = adata.var['feature_types'] == 'GEX'
    rna_cells = adata.obs['Modality'] == 'rna'
    x_gene_np = adata[rna_cells, :][:, gene_mask].layers['counts']
    x_gene_tensor = torch.tensor(
        x_gene_np.toarray() if issparse(x_gene_np) else x_gene_np,
        dtype=torch.float32, device=device
    )
    z2_index = adata[rna_cells, :].obs_names.values

    protein_mask = adata.var['feature_types'] == 'ADT'
    adt_cells = adata.obs['Modality'] == 'adt'
    x_adt_np = adata[adt_cells, :][:, protein_mask].layers['counts']
    x_adt_tensor = torch.tensor(
        x_adt_np.toarray() if issparse(x_adt_np) else x_adt_np,
        dtype=torch.float32, device=device
    )
    z3_index = adata[adt_cells, :].obs_names.values

    # Map x1 → a1, x3 → a3
    def x_to_a1_a3(x_gene_tensor, x_adt_tensor, model):
        batch_size = 512
        a1_list = []
        a3_list = []

        I = torch.eye(model.kv_n, device=x_gene_tensor.device)
        scale = math.sqrt(model.q_dim)

        model.eval()
        with torch.no_grad():
            for i in range(0, x_gene_tensor.shape[0], batch_size):
                batch = x_gene_tensor[i:i+batch_size]
                e1 = model.encoder1(batch)
                q1 = model.q_net1(e1)
                ker = model.keys1(I)
                var = model.values1(I)
                attn_logits1 = (q1 @ ker.T) / scale
                attn_weights1 = torch.softmax(attn_logits1, dim=-1)
                a1_list.append(attn_weights1 @ var)

            for i in range(0, x_adt_tensor.shape[0], batch_size):
                batch = x_adt_tensor[i:i+batch_size]
                e3 = model.encoder3(batch)
                q3 = model.q_net3(e3)
                kea = model.keys3(I)
                vaa = model.values3(I)
                attn_logits3 = (q3 @ kea.T) / scale
                attn_weights3 = torch.softmax(attn_logits3, dim=-1)
                a3_list.append(attn_weights3 @ vaa)

        a1 = torch.cat(a1_list, dim=0)
        a3 = torch.cat(a3_list, dim=0)
        return a1, a3

    a1, a3 = x_to_a1_a3(x_gene_tensor, x_adt_tensor, model)
    e1_tensor = infer_mlp(mlp1, a1)
    e3_tensor = infer_mlp(mlp2, a3)

    # Map e → z
    def e_to_z(e_tensor):
        batch_size = 512
        z_list = []
        with torch.no_grad():
            for i in range(0, e_tensor.shape[0], batch_size):
                batch = e_tensor[i:i+batch_size]
                mu = model.m_net(batch)
                logvar = model.l_net(batch)
                var = torch.exp(logvar) + 1e-8
                qz = Normal(mu, var.sqrt())
                z_list.append(qz.rsample())
        return torch.cat(z_list, dim=0)

    z2_tensor = e_to_z(e1_tensor)
    z3_tensor = e_to_z(e3_tensor)

    latent_dim = z1_tensor.shape[1]
    latent_matrix = torch.full((adata.n_obs, latent_dim), float('nan'), device=device)

    adata_index = adata.obs_names.values
    adata_mapping = {cell: i for i, cell in enumerate(adata_index)}

    def fill_latent(tensor, tensor_index):
        idx = [adata_mapping[cell] for cell in tensor_index]
        latent_matrix[idx, :] = tensor

    # Fill latent matrix for different modalities
    fill_latent(z1_tensor, z1_index)  # Multiome
    fill_latent(z2_tensor, z2_index)  # RNA
    fill_latent(z3_tensor, z3_index)  # ADT

    # Save latent embeddings
    adata.obsm['latent'] = latent_matrix.cpu().numpy()
    adata.write_h5ad(output_path)

def rnaatacadtmappingandimputing(output_path, rna, atac, adt, *args, num_epochs=200):
    
    """
    Functionality:
    --------------
    Train a latent space using Multiome and CITE modalities, then map RNA, ATAC, and ADT data
    into this latent space. Additionally, extract predicted values for specified genes
    (genes_of_interest) and proteins (adts_of_interest). The results are stored in
    rna.obsm and written to an h5ad file.

    Parameters:
    -----------
    output_path : str
        Path to save the resulting h5ad file.
    rna : AnnData
        RNA AnnData object; also used as the container to store latent variables
        and predicted gene/protein values.
    atac : AnnData
        ATAC AnnData object.
    adt : AnnData
        ADT (CITE-seq) AnnData object.
    *args : tuple
        Model parameters, including:
        n_hidden : int
            Number of hidden layers for all neural network components (encoder & decoder)
        hidden : int
            Hidden dimension for all layers in the network
        z_dim : int
            Dimension of the latent space
        q_dim : int
            Dimension of the query vector (Q) in attention
        kv_n : int
            Number of key-value (K-V) pairs in attention
        genes_of_interest : list[str]
            List of genes to extract predicted values for
        adts_of_interest : list[str]
            List of ADT proteins to extract predicted values for
    num_epochs : int, optional
        Number of training epochs for the main model (default: 200)

    Returns:
    --------
    None
        Results are stored in rna.obsm and written to the specified h5ad file.
    """
    
    n_hidden, hidden, z_dim, q_dim, kv_n, genes_of_interest, adts_of_interest = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z1_index = rna.obs_names[rna.obs["Modality"] == "multiome"].tolist()
    z2_index = rna.obs_names[rna.obs["Modality"] == "cite"].tolist()

    all_index = z1_index + z2_index

    rna_t_d  = rna[all_index].layers['counts']
    atac_t_d = atac[all_index].layers['counts']
    adt_t_d  = adt[all_index].layers['counts']

    rna_dim, atac_dim, adt_dim = rna.shape[1], atac.shape[1], adt.shape[1]

    if issparse(rna_t_d): rna_t_d = rna_t_d.toarray()
    if issparse(atac_t_d): atac_t_d = atac_t_d.toarray()
    if issparse(adt_t_d): adt_t_d = adt_t_d.toarray() 

    modality_map = {
        'multiome': 12,
        'cite': 13,
    }

    modality_vector = rna[all_index].obs['Modality'].map(modality_map)
    modality_d = modality_vector.to_numpy().astype(float)

    batch_indices = torch.from_numpy(rna[all_index].obs['batch'].astype('category').cat.codes.values).long()
    batch_encoded = torch.nn.functional.one_hot(batch_indices)
    batch_dim = batch_encoded.shape[1]

    dataset_t = MultiOmicsDataset(rna_t_d, atac_t_d, adt_t_d, modality_d, batch_encoded)

    train_loader = DataLoader(dataset_t, batch_size=512, shuffle=True)
    test1_loader = DataLoader(dataset_t, batch_size=512, shuffle=False)

    model = multigai(rna_dim, atac_dim, adt_dim, n_hidden, hidden, z_dim, batch_encoded.shape[1], q_dim, kv_n).to(device)
    optimizer_main = Adam(model.parameters(), lr=0.001)
    scheduler_main = torch.optim.lr_scheduler.StepLR(optimizer_main, step_size=50, gamma=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        kl_weight = 0.0 if epoch < 100 else 0.1
        model.train()
        
        for batch_data in train_loader:
            optimizer_main.zero_grad()

            m_values = batch_data[3] 
            unique_m = m_values.unique().tolist()  
            random.shuffle(unique_m)  

            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, m_tensor, batch_tensor, idx = [x.to(device) for x in sub_batch]

                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3, int(m_curr), batch_tensor
                    )
                    loss, reconst_loss, kl_loss, cos_loss = model.loss_function(
                        m1, m2, m3, int(m_curr), p1, p2, p3, qz, pz, a1, a2, a3, kl_weight
                    )
                    loss.backward()
                    optimizer_main.step()

                    running_loss += loss.item()

        scheduler_main.step()
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, KL_weight: {kl_weight}")

    model.eval()

    z1_dict, a1_z1_dict, a2_dict, ae_z1_dict = {}, {}, {}, {}
    z2_dict, a1_z2_dict, a3_dict, ae_z2_dict = {}, {}, {}, {}

    with torch.no_grad():
        for batch_data in test1_loader:
            m1, m2, m3, m_tensor, batch_tensor, idx = [d.to(device) for d in batch_data]

            obs_names = [all_index[i] for i in idx.cpu().numpy()]

            unique_m = m_tensor.unique()
            for m_val in unique_m:
                mask = (m_tensor == m_val)
                if mask.sum() == 0:
                    continue

                m1_sub = m1[mask]
                m2_sub = m2[mask]
                m3_sub = m3[mask]
                batch_sub = batch_tensor[mask]
                obs_names_sub = [obs_names[i] for i, flag in enumerate(mask.cpu().numpy()) if flag]

                z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                    m1_sub, m2_sub, m3_sub, int(m_val.item()), batch_sub
                )

                for i, name in enumerate(obs_names_sub):
                    if int(m_val.item()) == 12:  # z1
                        z1_dict[name] = z[i].unsqueeze(0)
                        a1_z1_dict[name] = a1[i].unsqueeze(0)
                        a2_dict[name] = a2[i].unsqueeze(0)
                        ae_z1_dict[name] = ae[i].unsqueeze(0)
                    elif int(m_val.item()) == 13:  # z2
                        z2_dict[name] = z[i].unsqueeze(0)
                        a1_z2_dict[name] = a1[i].unsqueeze(0)
                        a3_dict[name] = a3[i].unsqueeze(0)
                        ae_z2_dict[name] = ae[i].unsqueeze(0)

    z1_tensor = torch.cat([z1_dict[name] for name in z1_index], dim=0)
    a1_z1_tensor = torch.cat([a1_z1_dict[name] for name in z1_index], dim=0)
    a2_tensor = torch.cat([a2_dict[name] for name in z1_index], dim=0)
    ae_z1_tensor = torch.cat([ae_z1_dict[name] for name in z1_index], dim=0)

    z2_tensor = torch.cat([z2_dict[name] for name in z2_index], dim=0)
    a1_z2_tensor = torch.cat([a1_z2_dict[name] for name in z2_index], dim=0)
    a3_tensor = torch.cat([a3_dict[name] for name in z2_index], dim=0)
    ae_z2_tensor = torch.cat([ae_z2_dict[name] for name in z2_index], dim=0)

    def train_mlp(x_tensor, y_tensor, input_dim, output_dim, epochs=200):
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        mlp = M2L(input_dim, output_dim).to(device)
        optimizer = Adam(mlp.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        mlp.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                y_pred = mlp(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch+1) % 50 == 0:
                print(f"[MLP Epoch {epoch+1}/{epochs}] Loss: {running_loss/len(loader):.4f}")
        return mlp

    a1_tensor_full = torch.cat([a1_z1_tensor, a1_z2_tensor], dim=0)
    ae_a1_full = torch.cat([ae_z1_tensor, ae_z2_tensor], dim=0)

    a2_tensor_full = a2_tensor
    ae_a2_full = ae_z1_tensor  
    a3_tensor_full = a3_tensor
    ae_a3_full = ae_z2_tensor  

    mlp1 = train_mlp(a1_tensor_full, ae_a1_full, a1_tensor_full.shape[1], ae_a1_full.shape[1])
    mlp2 = train_mlp(a2_tensor_full, ae_a2_full, a2_tensor_full.shape[1], ae_a2_full.shape[1])
    mlp3 = train_mlp(a3_tensor_full, ae_a3_full, a3_tensor_full.shape[1], ae_a3_full.shape[1])

    def infer_mlp(mlp, x_tensor):
        batch_size = 512
        preds = []
        mlp.eval()
        with torch.no_grad():
            for i in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[i:i+batch_size]
                preds.append(mlp(batch))
        return torch.cat(preds, dim=0)  

    x_gene_np = rna[rna.obs['Modality'] == 'rna', :].layers['counts']
    rna_cells = rna.obs['Modality'] == 'rna'
    x_gene_tensor = torch.tensor(
        x_gene_np.toarray() if issparse(x_gene_np) else x_gene_np,
        dtype=torch.float32, device=device
    )
    z3_index = rna[rna_cells, :].obs_names.tolist()

    x_peaks_np = atac[rna.obs['Modality'] == 'atac', :].layers['counts']
    atac_cells = rna.obs['Modality'] == 'atac'
    x_peaks_tensor = torch.tensor(
        x_peaks_np.toarray() if issparse(x_peaks_np) else x_peaks_np,
        dtype=torch.float32, device=device
    )
    z4_index = atac[atac_cells, :].obs_names.tolist()

    x_adt_np = adt[rna.obs['Modality'] == 'adt', :].layers['counts']
    adt_cells = rna.obs['Modality'] == 'adt'
    x_adt_tensor = torch.tensor(
        x_adt_np.toarray() if issparse(x_adt_np) else x_adt_np,
        dtype=torch.float32, device=device
    )
    z5_index = adt[adt_cells, :].obs_names.tolist()

    def x_to_a1_a2_a3(x_gene_tensor, x_peaks_tensor, x_adt_tensor, model):
        batch_size = 512
        a1_list = []
        a2_list = []
        a3_list = []

        I = torch.eye(model.kv_n, device=x_gene_tensor.device) 
        scale = math.sqrt(model.q_dim)

        model.eval()
        with torch.no_grad():

            for i in range(0, x_gene_tensor.shape[0], batch_size):
                batch = x_gene_tensor[i:i+batch_size]
                e1 = model.encoder1(batch)
                q1 = model.q_net1(e1)
                ker = model.keys1(I)
                var = model.values1(I)
                attn_logits1 = (q1 @ ker.T) / scale
                attn_weights1 = torch.softmax(attn_logits1, dim=-1)
                a1_list.append(attn_weights1 @ var)

            for i in range(0, x_peaks_tensor.shape[0], batch_size):
                batch = x_peaks_tensor[i:i+batch_size]
                e2 = model.encoder2(batch)
                q2 = model.q_net2(e2)
                kea = model.keys2(I)
                vaa = model.values2(I)
                attn_logits2 = (q2 @ kea.T) / scale
                attn_weights2 = torch.softmax(attn_logits2, dim=-1)
                a2_list.append(attn_weights2 @ vaa)

            for i in range(0, x_adt_tensor.shape[0], batch_size):
                batch = x_adt_tensor[i:i+batch_size]
                e3 = model.encoder3(batch)
                q3 = model.q_net3(e3)
                kea = model.keys3(I)
                vaa = model.values3(I)
                attn_logits3 = (q3 @ kea.T) / scale
                attn_weights3 = torch.softmax(attn_logits3, dim=-1)
                a3_list.append(attn_weights3 @ vaa)

        a1 = torch.cat(a1_list, dim=0)
        a2 = torch.cat(a2_list, dim=0)
        a3 = torch.cat(a3_list, dim=0)
        return a1, a2, a3

    a1, a2, a3 = x_to_a1_a2_a3(x_gene_tensor, x_peaks_tensor, x_adt_tensor, model)
    e1_tensor = infer_mlp(mlp1, a1)
    e2_tensor = infer_mlp(mlp2, a2)
    e3_tensor = infer_mlp(mlp3, a3)

    def e_to_z(e_tensor):
        batch_size = 512
        z_list = []
        with torch.no_grad():
            for i in range(0, e_tensor.shape[0], batch_size):
                batch = e_tensor[i:i+batch_size]
                mu = model.m_net(batch)
                logvar = model.l_net(batch)
                var = torch.exp(logvar) + 1e-8
                qz = Normal(mu, var.sqrt())
                z_list.append(qz.rsample())
        return torch.cat(z_list, dim=0)  # GPU Tensor

    z3_tensor = e_to_z(e1_tensor)
    z4_tensor = e_to_z(e2_tensor)
    z5_tensor = e_to_z(e3_tensor)

    latent_dim = z1_tensor.shape[1]

    latent_matrix = torch.full((rna.n_obs, latent_dim), float('nan'), device=device)

    adata_index = rna.obs_names.values
    adata_mapping = {cell: i for i, cell in enumerate(adata_index)}

    def fill_latent(tensor, tensor_index):

        idx = [adata_mapping[cell] for cell in tensor_index]
        latent_matrix[idx, :] = tensor

    fill_latent(z1_tensor, z1_index) # Multiome
    fill_latent(z2_tensor, z2_index) # RNA
    fill_latent(z3_tensor, z3_index) # ATAC
    fill_latent(z4_tensor, z4_index) # ATAC
    fill_latent(z5_tensor, z5_index) # ATAC

    rna.obsm['latent'] = latent_matrix.cpu().numpy()

    # Get the indices of genes of interest in rna.var.index
    gene_idx = [rna.var.index.get_loc(g) for g in genes_of_interest]

    # Get the indices of ADTs of interest in adt.var.index
    adt_idx  = [adt.var.index.get_loc(a) for a in adts_of_interest]

    # Combine genes and ADTs into a single 'interest' list
    interest = genes_of_interest + adts_of_interest
    interest_idx_map = {name: i for i, name in enumerate(interest)}  # Map name → column index

    # Initialize prediction matrix with NaNs (shape: n_cells × n_features of interest)
    pred_matrix = torch.full((rna.n_obs, len(interest)), float('nan'), device=device)

    # Retrieve latent variables z and batch one-hot encoding
    z = torch.tensor(rna.obsm['latent'], device=device)
    batch_indices = torch.from_numpy(rna.obs['batch'].astype('category').cat.codes.values).long()
    batch_encoded = F.one_hot(batch_indices).float().to(device)

    batch_size = 512
    model.eval()
    with torch.no_grad():
        # Process data in batches
        for i in range(0, z.shape[0], batch_size):
            dz = z[i:i+batch_size]  # Current batch of latent variables
            batch_tensor = batch_encoded[i:i+dz.shape[0]].to(dz.device)  # Corresponding batch encoding

            # ===== Decode latent variables to original modalities =====
            temp = dz
            for layer in model.decoder_base:
                temp = torch.cat([temp, batch_tensor], dim=1)  # Concatenate batch info
                temp = layer(temp)
            final = torch.cat([temp, batch_tensor], dim=1)  # Final input for modality-specific heads

            # RNA prediction (p1)
            scale1 = model.fc_scale1(final)
            dropout1 = model.fc_dropout1(final)
            expectation1 = (1 - torch.sigmoid(dropout1)) * scale1  # Corrected expectation for RNA

            # ADT prediction (p3)
            scale3 = model.fc_scale3(final)
            dropout3 = model.fc_dropout3(final)
            expectation3 = (1 - torch.sigmoid(dropout3)) * scale3  # Corrected expectation for ADT

            idx_range = list(range(i, i + dz.shape[0]))  # Indices for current batch in pred_matrix

            # Store RNA predictions in pred_matrix
            for j, g in enumerate(genes_of_interest):
                pred_matrix[idx_range, interest_idx_map[g]] = expectation1[:, gene_idx[j]]

            # Store ADT predictions in pred_matrix
            for j, a in enumerate(adts_of_interest):
                pred_matrix[idx_range, interest_idx_map[a]] = expectation3[:, adt_idx[j]]

    # Save each feature prediction as a separate entry in rna.obsm
    for j, name in enumerate(interest):
        rna.obsm[name] = pred_matrix[:, j].cpu().numpy()

    # Write the updated AnnData object to an h5ad file
    rna.write_h5ad(output_path)

def train_and_evaluate_model(
    output_path,
    train_loader,
    test_loader,
    adata,
    *args,
    num_epochs=200
):
    """
    Train the MultiGAI model and extract latent representations.

    The final latent embeddings are stored in adata.obsm['latent'].
    """

    # Select training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack model hyperparameters
    input_dim1, input_dim2, input_dim3, n_hidden, hidden, z_dim, batch_dim, q_dim, kv_n = args

    # Initialize model
    model = multigai(
        input_dim1, input_dim2, input_dim3,
        n_hidden, hidden, z_dim,
        batch_dim, q_dim, kv_n
    ).to(device)

    # Optimizer and learning rate scheduler
    optimizer_main = Adam(model.parameters(), lr=0.001)
    scheduler_main = torch.optim.lr_scheduler.StepLR(
        optimizer_main, step_size=50, gamma=0.9
    )

    # Progress bar
    tqdm_bar = tqdm(range(num_epochs), desc="Training Progress")

    # ===== Training loop =====
    for epoch in tqdm_bar:
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        running_cos = 0.0

        # KL annealing schedule
        kl_weight = 0.0 if epoch < 100 else 0.1

        model.train()

        for batch_data in train_loader:
            optimizer_main.zero_grad()

            # Extract modality indicators for the batch
            m_values = batch_data[3]
            unique_m = m_values.unique()

            # Randomize modality processing order
            perm = torch.randperm(len(unique_m))
            unique_m = unique_m[perm]

            # Process each modality combination separately
            for m_curr in unique_m:
                mask = (m_values == m_curr)

                if mask.any():
                    # Sub-batch corresponding to the current modality
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, m_tensor, batch_tensor, idx = [
                        x.to(device) for x in sub_batch
                    ]

                    # Forward pass
                    z, p1, p2, p3, qz, pz, a1, a2, a3, ae = model(
                        m1, m2, m3,
                        int(m_curr.item()),
                        batch_tensor
                    )

                    # Compute loss
                    loss, reconst_loss, kl_loss, cos_loss = model.loss_function(
                        m1, m2, m3,
                        int(m_curr.item()),
                        p1, p2, p3,
                        qz, pz,
                        a1, a2, a3,
                        kl_weight
                    )

                    # Backpropagation and optimization
                    loss.backward()
                    optimizer_main.step()

                    # Accumulate metrics
                    running_loss += loss.item()
                    running_recon += reconst_loss.item()
                    running_kl += kl_loss.item()
                    running_cos += cos_loss.item()

        # Update progress bar
        n_batches = len(train_loader)
        tqdm_bar.set_postfix({
            "loss": f"{running_loss / n_batches:.4f}",
            "recon": f"{running_recon / n_batches:.4f}",
            "kl": f"{running_kl / n_batches:.4f}",
            "cos": f"{running_cos / n_batches:.4f}",
            "w": f"{kl_weight:.3f}"
        })

        scheduler_main.step()

    # ===== Evaluation and latent extraction =====
    model.eval()
    z_all = torch.zeros((len(adata), z_dim), device=device)

    with torch.no_grad():
        for batch_data in test_loader:
            indices = batch_data[-1]
            m_values = batch_data[3]
            unique_m = m_values.unique()

            for m_curr in unique_m:
                mask = (m_values == m_curr)
                if mask.any():
                    sub_batch = [d[mask] for d in batch_data]
                    m1, m2, m3, m_tensor, batch_tensor, idx = [
                        x.to(device) for x in sub_batch
                    ]

                    # Encode to latent space
                    z, _, _, _, _, _, _, _, _, _ = model(
                        m1, m2, m3,
                        int(m_curr.item()),
                        batch_tensor
                    )

                    # Store latent embeddings at original indices
                    z_all[idx.long()] = z

    # Save latent representation to AnnData
    adata.obsm['latent'] = z_all.cpu().numpy()
    adata.write_h5ad(output_path)