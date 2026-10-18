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

# Step 15 - gcn_linear_transform (not yet solved)
# TODO: implement

# Step 16 - gcn_layer_forward (not yet solved)
# TODO: implement

# Step 17 - init_gcn_parameters (not yet solved)
# TODO: implement

# Step 18 - gcn_stack_forward (not yet solved)
# TODO: implement

# Step 19 - gat_attention_logits (not yet solved)
# TODO: implement

# Step 20 - gat_masked_neighbor_softmax (not yet solved)
# TODO: implement

# Step 21 - gat_head_forward (not yet solved)
# TODO: implement

# Step 22 - merge_gat_heads (not yet solved)
# TODO: implement

# Step 23 - gat_layer_forward (not yet solved)
# TODO: implement

# Step 24 - init_gat_parameters (not yet solved)
# TODO: implement

# Step 25 - gat_stack_forward (not yet solved)
# TODO: implement

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

