"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
def edges_to_coo(edge_list, num_nodes=None):
    """Convert a list of (src, dst) edge pairs (or an [E, 2] LongTensor)
    into COO-format src/dst tensors.

    Returns (src, dst, num_nodes), where src and dst are 1-D LongTensors
    of length E. If num_nodes is None, it is inferred as max index + 1
    (0 for an empty edge list).
    """
    if isinstance(edge_list, torch.Tensor):
        edges = edge_list.to(torch.long)
    else:
        edges = torch.tensor(list(edge_list), dtype=torch.long)

    # Empty input (e.g. [] or shape [0]) becomes a [0, 2] tensor.
    if edges.numel() == 0:
        edges = edges.reshape(0, 2)

    if edges.dim() != 2 or edges.size(1) != 2:
        raise ValueError(
            f"edge_list must have shape [E, 2], got {tuple(edges.shape)}"
        )

    src = edges[:, 0].contiguous()
    dst = edges[:, 1].contiguous()

    if edges.size(0) > 0:
        if int(edges.min()) < 0:
            raise ValueError("node indices must be non-negative")
        max_idx = int(edges.max())
    else:
        max_idx = -1

    if num_nodes is None:
        num_nodes = max_idx + 1
    else:
        num_nodes = int(num_nodes)
        if max_idx >= num_nodes:
            raise ValueError(
                f"num_nodes={num_nodes} is too small for max node index {max_idx}"
            )

    return src, dst, num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    loops = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    src_out = torch.cat([src, loops], dim=0)
    dst_out = torch.cat([dst, loops.to(dst.dtype).to(dst.device)], dim=0)
    return src_out, dst_out

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    if edge_weight is None:
        edge_weight = torch.ones(dst.size(0), dtype=torch.float, device=dst.device)
    else:
        edge_weight = edge_weight.to(torch.float)

    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=dst.device)
    deg.scatter_add_(0, dst, edge_weight)
    return deg

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    if edge_weight is None:
        edge_weight = torch.ones(src.size(0), dtype=torch.float32, device=src.device)
    else:
        edge_weight = edge_weight.to(torch.float32)

    # Weighted in-degree of each node (scatter edge weights onto destinations).
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=src.device)
    deg.scatter_add_(0, dst, edge_weight)

    # d^{-1/2}, with isolated nodes (degree 0) mapped to 0 instead of inf.
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0.0)

    return deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    """Return edge-aligned source feature rows.

    Args:
        node_features (Tensor): Node feature matrix of shape [N, F].
        src (LongTensor): Source node index for each edge, shape [E].

    Returns:
        Tensor: Shape [E, F], where row e is node_features[src[e]].
    """
    return node_features.index_select(0, src)

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    out = torch.zeros(
        num_nodes, edge_features.size(1),
        dtype=edge_features.dtype, device=edge_features.device,
    )
    return out.index_add(0, dst, edge_features)

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    """Scatter-mean edge features onto destination nodes (sum, then divide by in-degree).

    Args:
        edge_features: FloatTensor of shape (E, F), one feature row per edge.
        dst: LongTensor of shape (E,), destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the mean of edge features with dst == j.
        Nodes with no incoming edges get a zero row.
    """
    out = torch.zeros(
        num_nodes, edge_features.size(1),
        dtype=edge_features.dtype, device=edge_features.device,
    )
    out = out.index_add(0, dst, edge_features)

    counts = torch.zeros(num_nodes, dtype=edge_features.dtype, device=edge_features.device)
    counts = counts.index_add(0, dst, torch.ones_like(dst, dtype=edge_features.dtype))

    # Clamp so isolated nodes give 0 / 1 = 0 instead of 0 / 0 = NaN.
    return out / counts.clamp(min=1).unsqueeze(-1)

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    """Scatter-max edge features onto destination nodes (elementwise max).

    Args:
        edge_features: FloatTensor of shape (E, F), one feature row per edge.
        dst: LongTensor of shape (E,), destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the elementwise max of edge
        features with dst == j. Nodes with no incoming edges stay at -inf.
    """
    F = edge_features.size(1)
    out = torch.full(
        (num_nodes, F), float("-inf"),
        dtype=edge_features.dtype, device=edge_features.device,
    )
    index = dst.unsqueeze(-1).expand(-1, F)
    return out.scatter_reduce(0, index, edge_features, reduce="amax", include_self=True)

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    src_feats = node_features.index_select(0, src)  # (E, F)
    dst_feats = node_features.index_select(0, dst)  # (E, F)

    if edge_attr is None:
        return message_fn(src_feats, dst_feats)
    return message_fn(src_feats, dst_feats, edge_attr)

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    if aggr not in ('sum', 'mean', 'max'):
        raise ValueError(f"aggr must be one of 'sum', 'mean', 'max', got {aggr!r}")

    M = messages.size(1)

    if aggr == 'sum':
        out = torch.zeros(num_nodes, M, dtype=messages.dtype, device=messages.device)
        return out.index_add(0, dst, messages)

    if aggr == 'mean':
        out = torch.zeros(num_nodes, M, dtype=messages.dtype, device=messages.device)
        out = out.index_add(0, dst, messages)
        counts = torch.zeros(num_nodes, dtype=messages.dtype, device=messages.device)
        counts = counts.index_add(0, dst, torch.ones_like(dst, dtype=messages.dtype))
        return out / counts.clamp(min=1).unsqueeze(-1)

    # aggr == 'max'
    out = torch.full((num_nodes, M), float("-inf"),
                     dtype=messages.dtype, device=messages.device)
    index = dst.unsqueeze(-1).expand(-1, M)
    return out.scatter_reduce(0, index, messages, reduce="amax", include_self=True)

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    """Fuse each node's current state with its aggregated message.

    Args:
        node_features: FloatTensor of shape (N, F), current node states h_v^t.
        aggregated: FloatTensor of shape (N, M), aggregated messages m_v^{t+1}.
        update_fn: callable(node_features, aggregated) -> new node states,
            e.g. a GRUCell, an MLP over their concatenation, or a sum.

    Returns:
        FloatTensor of shape (N, F'), the updated node states h_v^{t+1}.
    """
    return update_fn(node_features, aggregated)

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    num_nodes = node_features.size(0)

    # 1) Message: gather endpoint features per edge and apply message_fn.
    src_feats = node_features.index_select(0, src)
    dst_feats = node_features.index_select(0, dst)
    if edge_attr is None:
        messages = message_fn(src_feats, dst_feats)
    else:
        messages = message_fn(src_feats, dst_feats, edge_attr)

    # 2) Aggregate: reduce messages onto destination nodes.
    M = messages.size(1)
    if aggr == 'sum':
        aggregated = torch.zeros(num_nodes, M, dtype=messages.dtype, device=messages.device)
        aggregated = aggregated.index_add(0, dst, messages)
    elif aggr == 'mean':
        total = torch.zeros(num_nodes, M, dtype=messages.dtype, device=messages.device)
        total = total.index_add(0, dst, messages)
        counts = torch.zeros(num_nodes, dtype=messages.dtype, device=messages.device)
        counts = counts.index_add(0, dst, torch.ones_like(dst, dtype=messages.dtype))
        aggregated = total / counts.clamp(min=1).unsqueeze(-1)
    elif aggr == 'max':
        aggregated = torch.full((num_nodes, M), float("-inf"),
                                dtype=messages.dtype, device=messages.device)
        index = dst.unsqueeze(-1).expand(-1, M)
        aggregated = aggregated.scatter_reduce(0, index, messages, reduce="amax", include_self=True)
    else:
        raise ValueError(f"aggr must be one of 'sum', 'mean', 'max', got {aggr!r}")

    # 3) Update: fuse each node's state with its aggregated message.
    return update_fn(node_features, aggregated)

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    h = node_features
    all_layer_outputs = []

    for layer in layers:
        h = layer(h, src, dst, edge_attr=edge_attr)
        all_layer_outputs.append(h)

    return h, all_layer_outputs

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # A_tilde = A + I: append (i, i) for every node.
    loops = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    src_hat = torch.cat([src, loops], dim=0)
    dst_hat = torch.cat([dst, loops.to(dst.dtype)], dim=0)

    # D_tilde: degree of A_tilde (counted on destinations, one unit per edge).
    edge_weight = torch.ones(src_hat.size(0), dtype=torch.float32, device=src.device)
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=src.device)
    deg.scatter_add_(0, dst_hat, edge_weight)

    # D_tilde^{-1/2}. Every node has a self-loop, so deg >= 1 and this is finite.
    deg_inv_sqrt = deg.pow(-0.5)

    # Weight of edge (i -> j) is 1 / sqrt(d_i * d_j).
    norm_weight = deg_inv_sqrt[src_hat] * edge_weight * deg_inv_sqrt[dst_hat]

    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    out = node_features @ weight
    if bias is not None:
        out = out + bias
    return out

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    if num_nodes is None:
        num_nodes = node_features.size(0)

    # 1) Renormalize: A_tilde = A + I, weights 1 / sqrt(d_i * d_j).
    loops = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    src_hat = torch.cat([src, loops], dim=0)
    dst_hat = torch.cat([dst, loops.to(dst.dtype)], dim=0)

    ones = torch.ones(src_hat.size(0), dtype=node_features.dtype, device=node_features.device)
    deg = torch.zeros(num_nodes, dtype=node_features.dtype, device=node_features.device)
    deg.scatter_add_(0, dst_hat, ones)
    deg_inv_sqrt = deg.pow(-0.5)  # deg >= 1 because of the self-loops
    norm = deg_inv_sqrt[src_hat] * deg_inv_sqrt[dst_hat]  # (E + N,)

    # 2) Transform: X W.
    h = node_features @ weight  # (N, Fout)

    # 3) Propagate: out_j = sum over edges (i -> j) of norm_ij * h_i.
    messages = h.index_select(0, src_hat) * norm.unsqueeze(-1)  # (E + N, Fout)
    out = torch.zeros(num_nodes, h.size(1), dtype=h.dtype, device=h.device)
    out = out.index_add(0, dst_hat, messages)

    # Bias after propagation, then optional nonlinearity.
    if bias is not None:
        out = out + bias
    if activation is not None:
        out = activation(out)
    return out

# Step 17 - init_gcn_parameters
import math
import torch
import torch.nn as nn


def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    """Initialize GCN parameters with Glorot (Xavier) uniform weights.

    Args:
        in_dim: int, input feature dimension.
        out_dim: int, output feature dimension.
        with_bias: bool, whether to create a bias vector.
        seed: optional int for reproducible initialization.

    Returns:
        dict with:
            'weight': nn.Parameter of shape (in_dim, out_dim), drawn from
                U(-a, a) with a = sqrt(6 / (in_dim + out_dim)).
            'bias': nn.Parameter of shape (out_dim,), zero-initialized.
                Only present when with_bias is True.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    limit = math.sqrt(6.0 / (in_dim + out_dim))
    weight = (torch.rand(in_dim, out_dim, generator=generator) * 2.0 - 1.0) * limit

    params = {'weight': nn.Parameter(weight)}
    if with_bias:
        params['bias'] = nn.Parameter(torch.zeros(out_dim))
    return params

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.size(0)

    if activations is None:
        activations = [None] * len(param_list)
    if len(activations) != len(param_list):
        raise ValueError(
            f"activations has length {len(activations)} but param_list has "
            f"length {len(param_list)}"
        )

    # Renormalize the adjacency once: A_tilde = A + I, weights 1 / sqrt(d_i * d_j).
    loops = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    src_hat = torch.cat([src, loops], dim=0)
    dst_hat = torch.cat([dst, loops.to(dst.dtype)], dim=0)

    ones = torch.ones(src_hat.size(0), dtype=node_features.dtype, device=node_features.device)
    deg = torch.zeros(num_nodes, dtype=node_features.dtype, device=node_features.device)
    deg.scatter_add_(0, dst_hat, ones)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = (deg_inv_sqrt[src_hat] * deg_inv_sqrt[dst_hat]).unsqueeze(-1)  # (E + N, 1)

    h = node_features
    all_layer_outputs = []

    for params, act in zip(param_list, activations):
        weight = params['weight']
        bias = params.get('bias')

        # Transform, then propagate with the shared normalized adjacency.
        z = h @ weight
        messages = z.index_select(0, src_hat) * norm
        out = torch.zeros(num_nodes, z.size(1), dtype=z.dtype, device=z.device)
        out = out.index_add(0, dst_hat, messages)

        if bias is not None:
            out = out + bias
        if act is not None:
            out = act(out)

        h = out
        all_layer_outputs.append(h)

    return h, all_layer_outputs

# Step 19 - gat_attention_logits
import torch.nn.functional as F


def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    transformed = node_features @ weight  # (N, Fout)

    # a^T [W h_i || W h_j] splits into a_src . W h_i + a_dst . W h_j,
    # so compute each term once per node, then gather per edge.
    score_src = transformed @ attn_src  # (N,)
    score_dst = transformed @ attn_dst  # (N,)

    logits = score_src.index_select(0, src) + score_dst.index_select(0, dst)  # (E,)
    logits = F.leaky_relu(logits, negative_slope=0.2)

    return logits, transformed

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # Per-destination max for numerical stability (isolated nodes keep -inf,
    # but they have no edges, so that value is never gathered).
    max_per_node = torch.full((num_nodes,), float("-inf"),
                              dtype=logits.dtype, device=logits.device)
    max_per_node = max_per_node.scatter_reduce(0, dst, logits, reduce="amax", include_self=True)

    exp = torch.exp(logits - max_per_node.index_select(0, dst))  # (E,)

    # Per-destination sum of exponentials.
    denom = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
    denom = denom.index_add(0, dst, exp)

    return exp / denom.index_select(0, dst)

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    if num_nodes is None:
        num_nodes = node_features.size(0)

    # 1) Shared linear transform: z = X W.
    z = node_features @ weight  # (N, Fout)

    # 2) Attention logits: LeakyReLU(a_src . z_i + a_dst . z_j) per edge i -> j.
    score_src = z @ attn_src  # (N,)
    score_dst = z @ attn_dst  # (N,)
    logits = F.leaky_relu(
        score_src.index_select(0, src) + score_dst.index_select(0, dst),
        negative_slope=0.2,
    )  # (E,)

    # 3) Softmax over each destination node's incoming edges (stable).
    max_per_node = torch.full((num_nodes,), float("-inf"),
                              dtype=logits.dtype, device=logits.device)
    max_per_node = max_per_node.scatter_reduce(0, dst, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - max_per_node.index_select(0, dst))
    denom = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
    denom = denom.index_add(0, dst, exp)
    attn_coeffs = exp / denom.index_select(0, dst)  # (E,)

    # 4) Aggregate: out_j = sum over edges (i -> j) of alpha_ij * z_i.
    messages = z.index_select(0, src) * attn_coeffs.unsqueeze(-1)  # (E, Fout)
    head_out = torch.zeros(num_nodes, z.size(1), dtype=z.dtype, device=z.device)
    head_out = head_out.index_add(0, dst, messages)

    if bias is not None:
        head_out = head_out + bias
    if activation is not None:
        head_out = activation(head_out)

    return head_out, attn_coeffs

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    """Merge multi-head GAT outputs into one node-feature tensor.

    Args:
        head_outputs: either a list of K FloatTensors of shape (N, Fout),
            or a single stacked FloatTensor of shape (K, N, Fout).
        mode: 'concat' to concatenate heads along the feature dimension
            (hidden layers), or 'mean' to average them (output layer).

    Returns:
        FloatTensor of shape (N, K * Fout) for 'concat',
        or (N, Fout) for 'mean'.
    """
    if isinstance(head_outputs, torch.Tensor):
        stacked = head_outputs  # (K, N, Fout)
    else:
        stacked = torch.stack(list(head_outputs), dim=0)  # (K, N, Fout)

    if mode == 'concat':
        K, N, Fout = stacked.shape
        # (K, N, Fout) -> (N, K, Fout) -> (N, K * Fout)
        return stacked.permute(1, 0, 2).reshape(N, K * Fout)
    if mode == 'mean':
        return stacked.mean(dim=0)
    raise ValueError(f"mode must be 'concat' or 'mean', got {mode!r}")

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    if merge_mode not in ('concat', 'mean'):
        raise ValueError(f"merge_mode must be 'concat' or 'mean', got {merge_mode!r}")
    if num_nodes is None:
        num_nodes = node_features.size(0)

    head_outs = []
    all_attn = []

    for params in head_params:
        weight = params['weight']
        attn_src = params['attn_src']
        attn_dst = params['attn_dst']
        bias = params.get('bias')

        # Shared linear transform.
        z = node_features @ weight  # (N, Fout)

        # Attention logits: LeakyReLU(a_src . z_i + a_dst . z_j) per edge i -> j.
        logits = F.leaky_relu(
            (z @ attn_src).index_select(0, src) + (z @ attn_dst).index_select(0, dst),
            negative_slope=0.2,
        )  # (E,)

        # Numerically stable softmax over each destination's incoming edges.
        max_per_node = torch.full((num_nodes,), float("-inf"),
                                  dtype=logits.dtype, device=logits.device)
        max_per_node = max_per_node.scatter_reduce(0, dst, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - max_per_node.index_select(0, dst))
        denom = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
        denom = denom.index_add(0, dst, exp)
        alpha = exp / denom.index_select(0, dst)  # (E,)

        # Weighted aggregation of source features onto destinations.
        messages = z.index_select(0, src) * alpha.unsqueeze(-1)  # (E, Fout)
        out_h = torch.zeros(num_nodes, z.size(1), dtype=z.dtype, device=z.device)
        out_h = out_h.index_add(0, dst, messages)

        if bias is not None:
            out_h = out_h + bias

        head_outs.append(out_h)
        all_attn.append(alpha)

    # Merge heads: concat along features, or average across heads.
    stacked = torch.stack(head_outs, dim=0)  # (K, N, Fout)
    if merge_mode == 'concat':
        K, N, Fout = stacked.shape
        out = stacked.permute(1, 0, 2).reshape(N, K * Fout)
    else:
        out = stacked.mean(dim=0)

    if activation is not None:
        out = activation(out)

    return out, all_attn

# Step 24 - init_gat_parameters
def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    """Initialize multi-head GAT parameters with Glorot (Xavier) uniform init.

    Args:
        in_dim: int, input feature dimension.
        out_dim: int, per-head output feature dimension.
        num_heads: int, number of attention heads.
        with_bias: bool, whether each head gets a bias vector.
        seed: optional int for reproducible initialization.

    Returns:
        list of num_heads dicts, each with:
            'weight':   nn.Parameter (in_dim, out_dim), U(-a, a), a = sqrt(6 / (in_dim + out_dim))
            'attn_src': nn.Parameter (out_dim,), U(-b, b), b = sqrt(6 / (out_dim + 1))
            'attn_dst': nn.Parameter (out_dim,), same distribution as attn_src
            'bias':     nn.Parameter (out_dim,), zeros (only if with_bias)
        This is the format gat_layer_forward expects for head_params.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def glorot(shape, fan_in, fan_out):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return nn.Parameter((torch.rand(*shape, generator=generator) * 2.0 - 1.0) * limit)

    head_params = []
    for _ in range(num_heads):
        params = {
            'weight': glorot((in_dim, out_dim), in_dim, out_dim),
            # Attention vectors are treated as (out_dim, 1) matrices for Glorot.
            'attn_src': glorot((out_dim,), out_dim, 1),
            'attn_dst': glorot((out_dim,), out_dim, 1),
        }
        if with_bias:
            params['bias'] = nn.Parameter(torch.zeros(out_dim))
        head_params.append(params)

    return head_params

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    L = len(layer_param_list)
    if num_nodes is None:
        num_nodes = node_features.size(0)
    if merge_modes is None:
        merge_modes = ['concat'] * L
    if activations is None:
        activations = [None] * L
    if len(merge_modes) != L:
        raise ValueError(f"merge_modes has length {len(merge_modes)} but there are {L} layers")
    if len(activations) != L:
        raise ValueError(f"activations has length {len(activations)} but there are {L} layers")

    def gat_layer(h, head_params, merge_mode, activation):
        if merge_mode not in ('concat', 'mean'):
            raise ValueError(f"merge_mode must be 'concat' or 'mean', got {merge_mode!r}")

        head_outs = []
        for params in head_params:
            z = h @ params['weight']  # (N, Fout)

            # Attention logits, then stable softmax over each destination's edges.
            logits = F.leaky_relu(
                (z @ params['attn_src']).index_select(0, src)
                + (z @ params['attn_dst']).index_select(0, dst),
                negative_slope=0.2,
            )
            max_per_node = torch.full((num_nodes,), float("-inf"),
                                      dtype=logits.dtype, device=logits.device)
            max_per_node = max_per_node.scatter_reduce(0, dst, logits, reduce="amax", include_self=True)
            exp = torch.exp(logits - max_per_node.index_select(0, dst))
            denom = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
            denom = denom.index_add(0, dst, exp)
            alpha = exp / denom.index_select(0, dst)

            # Attention-weighted aggregation onto destinations.
            messages = z.index_select(0, src) * alpha.unsqueeze(-1)
            out_h = torch.zeros(num_nodes, z.size(1), dtype=z.dtype, device=z.device)
            out_h = out_h.index_add(0, dst, messages)

            bias = params.get('bias')
            if bias is not None:
                out_h = out_h + bias
            head_outs.append(out_h)

        stacked = torch.stack(head_outs, dim=0)  # (K, N, Fout)
        if merge_mode == 'concat':
            K, N, Fout = stacked.shape
            out = stacked.permute(1, 0, 2).reshape(N, K * Fout)
        else:
            out = stacked.mean(dim=0)

        if activation is not None:
            out = activation(out)
        return out

    h = node_features
    all_layer_outputs = []
    for head_params, mode, act in zip(layer_param_list, merge_modes, activations):
        h = gat_layer(h, head_params, mode, act)
        all_layer_outputs.append(h)

    return h, all_layer_outputs

# Step 26 - global_mean_pool (not yet solved)
# TODO: implement

# Step 27 - global_sum_pool (not yet solved)
# TODO: implement

# Step 28 - global_max_pool (not yet solved)
# TODO: implement

# Step 29 - global_mean_max_pool (not yet solved)
# TODO: implement

# Step 30 - node_classification_head (not yet solved)
# TODO: implement

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph (not yet solved)
# TODO: implement

# Step 33 - build_node_classification_dataset (not yet solved)
# TODO: implement

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

