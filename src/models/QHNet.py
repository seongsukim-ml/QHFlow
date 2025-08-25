import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from e3nn.o3 import Linear

from .layers import *
from .modules import *

"""
QHNet for QH9

The original QHNet model is from:
<https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9>
"""

class QHNet(nn.Module):
    """
    Quantum Hamiltonian Network (QHNet) for predicting molecular Hamiltonian matrices.
    
    This model uses equivariant graph neural networks to predict quantum mechanical
    properties of molecular systems by learning Hamiltonian matrix representations.
    
    Args:
        in_node_features (int): Number of input node features (default: 1)
        sh_lmax (int): Maximum spherical harmonics order (default: 4)
        hidden_size (int): Hidden layer size (default: 128)
        bottle_hidden_size (int): Bottleneck hidden layer size (default: 32)
        num_gnn_layers (int): Number of GNN layers (default: 5)
        max_radius (float): Maximum radius for graph construction (default: 12)
        num_nodes (int): Maximum number of nodes (default: 10)
        radius_embed_dim (int): Dimension for radius embeddings (default: 32)
        **kwargs: Additional keyword arguments
    """
    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,
        **kwargs,
    ):
        super(QHNet, self).__init__()
        
        # Core model parameters
        self.order = sh_lmax
        self.hidden_size = hidden_size
        self.bottle_hidden_size = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.start_layer = 2
        
        # Initialize irreducible representations
        self._init_irreps()
        
        # Initialize embeddings and transformations
        self._init_embeddings(num_nodes)
        
        # Initialize network components
        self._init_network_components()
        
        # Initialize output layers
        self._init_output_layers()
    
    def _init_irreps(self):
        """Initialize irreducible representations for E(3) equivariance."""
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        
        # Hidden representations with alternating parity
        self.hidden_irrep = o3.Irreps(
            f"{self.hidden_size}x0e + "
            f"{self.hidden_size}x1o + "
            f"{self.hidden_size}x2e + "
            f"{self.hidden_size}x3o + "
            f"{self.hidden_size}x4e"
        )
        
        # Bottleneck representations
        self.hidden_bottle_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1o + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3o + "
            f"{self.bottle_hidden_size}x4e"
        )
        
        # Base representations (all even parity)
        self.hidden_irrep_base = o3.Irreps(
            f"{self.hidden_size}x0e + "
            f"{self.hidden_size}x1e + "
            f"{self.hidden_size}x2e + "
            f"{self.hidden_size}x3e + "
            f"{self.hidden_size}x4e"
        )
        
        self.hidden_bottle_irrep_base = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3e + "
            f"{self.bottle_hidden_size}x4e"
        )
        
        # Input and output representations
        self.input_irrep = o3.Irreps(f"{self.hidden_size}x0e")
        self.final_out_irrep = o3.Irreps(
            f"{self.hidden_size * 3}x0e + "
            f"{self.hidden_size * 2}x1o + "
            f"{self.hidden_size}x2e"
        ).simplify()
    
    def _init_embeddings(self, num_nodes):
        """Initialize embedding layers."""
        self.node_embedding = nn.Embedding(num_nodes, self.hidden_size)
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim, self.max_radius
        )
        
        # One-body reduction for processing input matrices
        self.onebody_reduction = OneBody_Reduction()
    
    def _init_network_components(self):
        """Initialize main network components."""
        # Network configuration
        self.norm_layer = "layer"
        self.irreps_node_attr = "1x0e"
        self.irreps_head = o3.Irreps("32x0e+16x1o+8x2e")
        self.num_heads = 4
        self.irreps_pre_attn = None
        self.rescale_degree = False
        self.nonlinear_message = False
        
        # Dropout parameters
        self.alpha_drop = 0.0
        self.proj_drop = 0.0
        self.out_drop = 0.0
        self.drop_path_rate = 0.0
        
        # MLP configuration
        fc_neurons = [64, 64]
        self.fc_neurons = [self.radius_embed_dim] + fc_neurons
        self.irreps_mlp_mid = "128x0e+64x1e+32x2e"
        self.num_fc_layer = 1
        
        # Nonlinear activation configurations
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        
        # Initialize layer modules
        self._build_layers()

    def _build_layers(self):
        """Build the main network layers."""
        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.blocks_H = nn.ModuleList()
        self.blocks_S = nn.ModuleList()
        
        for i in range(self.num_gnn_layers):
            # Determine input irrep for first layer vs. subsequent layers
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            
            # Add convolution layer
            self.e3_gnn_layer.append(
                ConvNetLayer(
                    irrep_in_node=input_irrep,
                    irrep_hidden=self.hidden_irrep,
                    irrep_out=self.hidden_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hidden_size,
                    sh_irrep=self.sh_irrep,
                    resnet=True,
                    use_norm_gate=True if i != 0 else False,
                )
            )

            # Node embedding irrep for transformer blocks
            irreps_node_embedding = (
                o3.Irreps("16x0e+8x1o+4x2e") if i == 0 else self.hidden_irrep
            )

            # Add Hamiltonian transformer block
            self.blocks_H.append(
                self._create_trans_block(irreps_node_embedding)
            )

            # Add overlap transformer block
            self.blocks_S.append(
                self._create_trans_block(irreps_node_embedding)
            )

            # Add higher-order layers after start_layer
            if i > self.start_layer:
                self.e3_gnn_node_layer.append(
                    SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hidden_size,
                        resnet=True,
                    )
                )

                self.e3_gnn_node_pair_layer.append(
                    PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hidden_size,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hidden_size,
                        resnet=True,
                    )
                )
        
        # Add normalization layer
        self.norm = get_norm_layer(self.norm_layer)(self.hidden_irrep)
    
    def _create_trans_block(self, irreps_node_embedding):
        """Create a transformer block with standard configuration."""
        return TransBlock(
            irreps_node_input=irreps_node_embedding,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.sh_irrep,
            irreps_node_output=self.hidden_irrep,
            fc_neurons=self.fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=self.alpha_drop,
            proj_drop=self.proj_drop,
            drop_path_rate=self.drop_path_rate,
            irreps_mlp_mid=self.irreps_mlp_mid,
            norm_layer=self.norm_layer,
        )

    def _init_output_layers(self):
        """Initialize output layers for Hamiltonian prediction."""
        self.nonlinear_layer = get_nonlinear("ssp")
        
        # Initialize module dictionaries for different matrix components
        self.expand_ii = nn.ModuleDict()
        self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        self.fc_ij_bias = nn.ModuleDict()
        
        # Create expansion and FC layers for Hamiltonian prediction
        for matrix_type in {"hamiltonian"}:
            self._create_matrix_prediction_layers(matrix_type)
        
        # Linear transformations to bottleneck representation
        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
    
    def _create_matrix_prediction_layers(self, matrix_type):
        """Create layers for predicting matrix elements."""
        # Input irrep for expansion
        input_expand_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + {self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + {self.bottle_hidden_size}x3e + {self.bottle_hidden_size}x4e"
        )
        output_irrep = o3.Irreps("3x0e + 2x1e + 1x2e")
        
        # Diagonal elements (ii)
        self.expand_ii[matrix_type] = Expansion(
            input_expand_irrep, output_irrep, output_irrep
        )
        self.fc_ii[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_path_weight),
        )
        self.fc_ii_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_bias),
        )
        
        # Off-diagonal elements (ij)
        self.expand_ij[matrix_type] = Expansion(
            input_expand_irrep, output_irrep, output_irrep
        )
        self.fc_ij[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        )
        self.fc_ij_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        )

    def get_number_of_parameters(self):
        """Calculate the total number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def set(self, device):
        """Set the device and initialize orbital masks."""
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def injection(self, data):
        """Inject molecular graph information into data object.
        
        Args:
            data: Input data object containing molecular information
            
        Returns:
            tuple: (data, node_attr, edge_sh, rbf_new, transpose_edge_index)
        """
        # Build local graph with max_radius
        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(
            data, self.max_radius
        )
        node_attr = self.node_embedding(node_attr)
        
        # Update data with local graph information
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = (
            node_attr, edge_index, rbf_new, edge_sh
        )

        # Build full graph for complete molecular interaction
        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = (
            self.build_graph(data, 10000)  # Large radius for full connectivity
        )
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = (
            full_edge_index, full_edge_attr, full_edge_sh
        )
        
        return data, node_attr, edge_sh, rbf_new, transpose_edge_index

    def filter(
        self,
        H,
        data,
        node_attr,
        edge_sh,
        rbf_new,
        transpose_edge_index,
        keep_blocks=False,
    ):
        """Apply the neural network filter to predict Hamiltonian matrix.
        
        Args:
            H: Input Hamiltonian matrix
            data: Molecular data object
            node_attr: Node attributes
            edge_sh: Edge spherical harmonics
            rbf_new: Radial basis functions
            transpose_edge_index: Transpose indices for edge symmetry
            keep_blocks: Whether to return block matrices separately
            
        Returns:
            Predicted Hamiltonian matrix or block matrices
        """
        # Initialize node features and extract edge information
        node_attr_R = node_attr
        edge_dst, edge_src = data.edge_index
        full_dst, full_src = data.full_edge_index

        # Reduce input matrices to node features
        node_feats_H = self.onebody_reduction(data, H, keep_blocks)
        node_feats_S = self.onebody_reduction(data, data.overlap, keep_blocks)

        # Initialize features for higher-order interactions
        fii = None  # Self-interaction features
        fij = None  # Pair-interaction features
        
        # Process through GNN layers
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            # Update node representations through convolution
            node_attr_R = layer(data, node_attr_R)
            
            # Create node attributes for transformer blocks
            node_attr = torch.ones_like(node_feats_H.narrow(1, 0, 1))
            
            # Process Hamiltonian features
            node_feats_H = self.blocks_H[layer_idx](
                node_input=node_feats_H,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=rbf_new,
                batch=data.batch,
            )
            
            # Process overlap features
            node_feats_S = self.blocks_S[layer_idx](
                node_input=node_feats_S,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=rbf_new,
                batch=data.batch,
            )
            
            # Combine and normalize features
            node_attr = self.norm(
                node_attr_R + node_feats_H + node_feats_S, batch=data.batch
            )
            
            # Apply higher-order layers after start_layer
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx - self.start_layer - 1](
                    data, node_attr, fii
                )
                fij = self.e3_gnn_node_pair_layer[layer_idx - self.start_layer - 1](
                    data, node_attr, fij
                )

        # Transform to output representation
        fii = self.output_ii(fii)
        fij = self.output_ij(fij)
        
        # Generate diagonal matrix elements
        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii,
            self.fc_ii["hamiltonian"](data.node_attr),
            self.fc_ii_bias["hamiltonian"](data.node_attr),
        )
        
        # Generate off-diagonal matrix elements
        node_pair_embedding = torch.cat(
            [data.node_attr[full_dst], data.node_attr[full_src]], dim=-1
        )
        hamiltonian_non_diagonal_matrix = self.expand_ij["hamiltonian"](
            fij,
            self.fc_ij["hamiltonian"](node_pair_embedding),
            self.fc_ij_bias["hamiltonian"](node_pair_embedding),
        )

        if not keep_blocks:
            # Build complete Hamiltonian matrix
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix
            )
            # Ensure Hermitian symmetry
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)
            return hamiltonian_matrix
        else:
            # Return block matrices separately
            ret_hamiltonian_diagonal_matrix = (
                hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)
            )
            
            # Apply transpose considering edge symmetry
            ret_hamiltonian_non_diagonal_matrix = (
                hamiltonian_non_diagonal_matrix
                + hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)
            )
            return ret_hamiltonian_diagonal_matrix, ret_hamiltonian_non_diagonal_matrix

    def forward(self, data, H=None, keep_blocks=False):
        """Forward pass of the QHNet model.
        
        Args:
            data: Molecular data object
            H: Input Hamiltonian matrix (optional)
            keep_blocks: Whether to return block matrices separately
            
        Returns:
            dict: Dictionary containing predicted Hamiltonian matrix/blocks
        """
        # Initialize Hamiltonian if not provided
        if H is None:
            H = self._initialize_hamiltonian(data, keep_blocks)

        # Process molecular data and extract features
        data, node_attr, edge_sh, rbf_new, transpose_edge_index = self.injection(data)
        
        # Apply neural network filter to predict Hamiltonian
        H_pred = self.filter(
            H, data, node_attr, edge_sh, rbf_new, transpose_edge_index, keep_blocks
        )
        
        # Format results
        results = {}
        if keep_blocks:
            results["hamiltonian_diagonal_blocks"] = H_pred[0]
            results["hamiltonian_non_diagonal_blocks"] = H_pred[1]
        else:
            results["hamiltonian"] = H_pred
            
        return results
    
    def _initialize_hamiltonian(self, data, keep_blocks):
        """Initialize Hamiltonian matrix if not provided.
        
        Args:
            data: Molecular data object
            keep_blocks: Whether blocks are kept separate
            
        Returns:
            torch.Tensor: Initialized Hamiltonian matrix
        """
        if keep_blocks:
            batch_size = data.diagonal_hamiltonian_mask.size(0)
            mat_size = data.diagonal_hamiltonian_mask.size(-1)
            H = (
                data.diagonal_hamiltonian_mask
                * torch.eye(mat_size)
                .repeat(batch_size, 1, 1)
                .type(data.overlap.dtype)
                .to(data.overlap.device)
                .requires_grad_()
            )
        else:
            batch_size = data.overlap.size(0)
            mat_size = data.overlap.size(-1)
            H = (
                torch.eye(mat_size)
                .repeat(batch_size, 1, 1)
                .type(data.overlap.dtype)
                .to(data.overlap.device)
                .requires_grad_()
            )
        
        return H

    def build_graph(self, data, max_radius):
        """Build molecular graph with specified radius cutoff.
        
        Args:
            data: Molecular data object
            max_radius: Maximum radius for edge connections
            
        Returns:
            tuple: (node_attr, radius_edges, rbf, edge_sh, transpose_index)
        """
        # Extract node attributes (atomic numbers)
        node_attr = data.atoms.squeeze()
        
        # Build radius graph
        radius_edges = radius_graph(data.pos, max_radius, data.batch)
        dst, src = radius_edges
        
        # Calculate edge vectors and distances
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_distances = edge_vec.norm(dim=-1)
        
        # Apply radial basis function expansion
        rbf = (
            self.distance_expansion(edge_distances.unsqueeze(-1))
            .squeeze()
            .type(data.pos.type())
        )

        # Calculate spherical harmonics for edge attributes
        # Reorder coordinates for proper spherical harmonics computation
        edge_sh = o3.spherical_harmonics(
            self.sh_irrep,
            edge_vec[:, [1, 2, 0]],  # (y, z, x) ordering
            normalize=True,
            normalization="component",
        ).type(data.pos.type())

        # Calculate transpose indices for edge symmetry
        transpose_indices = self._calculate_transpose_indices(data, radius_edges)

        return node_attr, radius_edges, rbf, edge_sh, transpose_indices
    
    def _calculate_transpose_indices(self, data, radius_edges):
        """Calculate transpose indices for maintaining edge symmetry."""
        start_edge_index = 0
        all_transpose_index = []
        
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx + 1] - data.ptr[graph_idx]
            
            # Extract edges for current graph
            graph_edge_index = radius_edges[
                :, start_edge_index : start_edge_index + num_nodes * (num_nodes - 1)
            ]
            
            # Convert to local indices
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            
            # Calculate transpose mapping
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = (
                sub_graph_edge_index[0] * (num_nodes - 1)
                + sub_graph_edge_index[1]
                - bias
            )
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            
            start_edge_index = start_edge_index + num_nodes * (num_nodes - 1)

        return torch.cat(all_transpose_index, dim=-1)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        """Build the final Hamiltonian matrix from diagonal and off-diagonal blocks.
        
        Args:
            data: Molecular data object
            diagonal_matrix: Diagonal block matrices
            non_diagonal_matrix: Off-diagonal block matrices
            
        Returns:
            torch.Tensor: Complete Hamiltonian matrix
        """
        final_matrix = []
        dst, src = data.full_edge_index
        
        # Process each molecule in the batch
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            
            # Build matrix for current molecule
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    if src_idx == dst_idx:
                        # Diagonal block
                        orbital_mask_src = self.orbital_mask[data.atoms[src_idx].item()]
                        orbital_mask_dst = self.orbital_mask[data.atoms[dst_idx].item()]
                        
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(-2, orbital_mask_dst)
                            .index_select(-1, orbital_mask_src)
                        )
                    else:
                        # Off-diagonal block
                        mask1 = src == src_idx
                        mask2 = dst == dst_idx
                        index = torch.where(mask1 & mask2)[0].item()
                        
                        orbital_mask_src = self.orbital_mask[data.atoms[src_idx].item()]
                        orbital_mask_dst = self.orbital_mask[data.atoms[dst_idx].item()]
                        
                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(-2, orbital_mask_dst)
                            .index_select(-1, orbital_mask_src)
                        )
                
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        
        return torch.stack(final_matrix, dim=0)

    def get_orbital_mask(self):
        """Get orbital masks for different atomic numbers.
        
        Returns:
            dict: Mapping from atomic number to orbital indices
        """
        # Define orbital indices for different shell types
        idx_1s_2s = torch.tensor([0, 1])  # s orbitals
        idx_2p = torch.tensor([3, 4, 5])  # p orbitals
        
        # Combine s and p orbitals for light elements (H, He)
        orbital_mask_light = torch.cat([idx_1s_2s, idx_2p])
        
        # Full orbital set for heavier elements (includes d orbitals)
        orbital_mask_heavy = torch.arange(14)
        
        # Create mapping: atomic numbers 1,2 use light mask, others use heavy mask
        orbital_mask = {}
        for atomic_num in range(1, 11):
            orbital_mask[atomic_num] = (
                orbital_mask_light if atomic_num <= 2 else orbital_mask_heavy
            )
        
        return orbital_mask

    def split_matrix(self, data):
        """Split input matrix into diagonal and off-diagonal blocks.
        
        Args:
            data: Molecular data object containing the matrix to split
            
        Returns:
            tuple: (diagonal_matrix, non_diagonal_matrix) blocks
        """
        # Initialize output matrices
        diagonal_matrix = torch.zeros(
            data.atoms.shape[0], 14, 14, dtype=data.pos.dtype, device=self.device
        )
        non_diagonal_matrix = torch.zeros(
            data.edge_index.shape[1], 14, 14, dtype=data.pos.dtype, device=self.device
        )

        # Reshape input matrix for batch processing
        batch_size = len(data.ptr) - 1
        data.matrix = data.matrix.reshape(
            batch_size, data.matrix.shape[-1], data.matrix.shape[-1]
        )

        num_atoms = 0
        num_edges = 0
        
        # Process each molecule in the batch
        for graph_idx in range(batch_size):
            # Calculate orbital slices for current molecule
            slices = self._calculate_orbital_slices(data, graph_idx)
            
            # Extract diagonal blocks
            for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                local_node_idx = node_idx - num_atoms
                orb_mask = self.orbital_mask[data.atoms[node_idx].item()]
                
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = data.matrix[graph_idx][
                    slices[local_node_idx] : slices[local_node_idx + 1],
                    slices[local_node_idx] : slices[local_node_idx + 1],
                ]

            # Extract off-diagonal blocks
            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                
                # Check if edge belongs to current graph
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                    
                num_edges += 1
                orb_mask_dst = self.orbital_mask[data.atoms[dst].item()]
                orb_mask_src = self.orbital_mask[data.atoms[src].item()]
                
                graph_dst, graph_src = dst - num_atoms, src - num_atoms
                non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = (
                    data.matrix[graph_idx][
                        slices[graph_dst] : slices[graph_dst + 1],
                        slices[graph_src] : slices[graph_src + 1],
                    ]
                )

            num_atoms += data.ptr[graph_idx + 1] - data.ptr[graph_idx]
            
        return diagonal_matrix, non_diagonal_matrix
    
    def _calculate_orbital_slices(self, data, graph_idx):
        """Calculate orbital slicing indices for a specific graph."""
        slices = [0]
        for atom_idx in data.atoms[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
            orbital_count = len(self.orbital_mask[atom_idx.item()])
            slices.append(slices[-1] + orbital_count)
        return slices
