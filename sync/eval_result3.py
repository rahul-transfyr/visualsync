import os
import re
import numpy as np
import pickle
import glob
import argparse
from collections import defaultdict
from itertools import combinations
from tabulate import tabulate
import networkx as nx
import random


def get_unique_video_names(folder_names):
    """
    Extracts unique video names from a list of folder paths.

    Args:
        folder_names (list of str): List of folder paths containing video names.

    Returns:
        set: A set of unique video names.
    """
    unique_videos = set()
    
    for folder in folder_names:
        # Get the basename of the folder path
        base_name = os.path.basename(folder)
        
        # Split by "__" to get individual video names
        video_names = base_name.split("__")
        
        # Add each video name to the set
        unique_videos.update(video_names)
    
    return unique_videos


def select_pairs_by_ratio(edges, result_offsets, video_names, pair_ratio=1.0, seed=42):
    """
    Select a subset of edge pairs based on a ratio, ensuring each video appears in at least one pair.
    
    Args:
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        video_names (list): List of video names.
        pair_ratio (float): Ratio of pairs to select (0 to 1).
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (selected_edges, selected_offsets) - Selected edge pairs and their offsets.
    """
    if pair_ratio >= 1.0:
        return edges, result_offsets
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create a list of valid edges (those in the result_offsets)
    valid_edges = []
    for edge in edges:
        i, j = edge
        if (i, j) in result_offsets or (j, i) in result_offsets:
            valid_edges.append(edge)
    
    # Track which videos are included in selected pairs
    n_videos = len(video_names)
    videos_included = set()
    
    # Calculate how many edges to select
    n_total_select = max(n_videos - 1, int(len(valid_edges) * pair_ratio))
    n_select = min(n_total_select, len(valid_edges))  # Can't select more than available
    
    # Step 1: First ensure each video is included in at least one pair
    # We'll prioritize edges that include videos not yet covered
    valid_edges_with_coverage = []
    
    for edge in valid_edges:
        i, j = edge
        # Calculate a score: 0 if both videos already included, 1 if one video included, 2 if neither included
        coverage_score = (i not in videos_included) + (j not in videos_included)
        valid_edges_with_coverage.append((edge, coverage_score))
    
    # Sort by coverage score, higher score first (more uncovered videos)
    valid_edges_with_coverage.sort(key=lambda x: x[1], reverse=True)
    
    selected_edges = []
    remaining_edges = []
    
    # First pass: Select edges until all videos are covered or we reach the target count
    for edge_info in valid_edges_with_coverage:
        edge, score = edge_info
        i, j = edge
        
        if score > 0 and len(selected_edges) < n_select:
            # This edge covers at least one new video
            selected_edges.append(edge)
            videos_included.add(i)
            videos_included.add(j)
        else:
            # This edge doesn't add coverage or we've already selected enough
            remaining_edges.append(edge)
    
    # If we haven't selected enough edges yet, add randomly from remaining edges
    if len(selected_edges) < n_select:
        # Shuffle remaining edges for random selection
        random.shuffle(remaining_edges)
        additional_edges = remaining_edges[:n_select - len(selected_edges)]
        selected_edges.extend(additional_edges)
    
    # Create a new dictionary with only the selected offsets
    selected_offsets = {}
    for i, j in selected_edges:
        if (i, j) in result_offsets:
            selected_offsets[(i, j)] = result_offsets[(i, j)]
        elif (j, i) in result_offsets:
            selected_offsets[(i, j)] = -result_offsets[(j, i)]
    
    # Check if all videos are included
    if len(videos_included) < n_videos:
        missing_videos = set(range(n_videos)) - videos_included
        missing_video_names = [video_names[i] for i in missing_videos]
        print(f"Warning: {len(missing_videos)} videos are not included in any pair: {missing_video_names}")
        print("Adding additional edges to ensure all videos are included...")
        
        # For each missing video, find a valid edge that includes it
        for video_idx in missing_videos:
            # Find edges that include this video
            video_edges = [edge for edge in valid_edges if video_idx in edge and edge not in selected_edges]
            
            if video_edges:
                # Pick a random edge
                selected_edge = random.choice(video_edges)
                selected_edges.append(selected_edge)
                
                # Add to offsets
                i, j = selected_edge
                if (i, j) in result_offsets:
                    selected_offsets[(i, j)] = result_offsets[(i, j)]
                elif (j, i) in result_offsets:
                    selected_offsets[(i, j)] = -result_offsets[(j, i)]
                
                # Mark videos as included
                videos_included.add(i)
                videos_included.add(j)
            else:
                print(f"Warning: Could not find any valid edge for video {video_names[video_idx]}")
    
    print(f"Selected {len(selected_edges)} edges ({pair_ratio:.2f} of total) from {len(valid_edges)} valid edges")
    return selected_edges, selected_offsets


def select_pairs_by_random_st(video_names, edges, result_offsets):
    """
    Select edge pairs forming a random spanning tree.
    
    Args:
        video_names (list): List of video names.
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        
    Returns:
        tuple: (rst_edges, rst_offsets) - Random spanning tree edge pairs and their offsets.
    """
    import networkx as nx
    import random
    
    # Create a graph
    G = nx.Graph()
    N = len(video_names)
    G.add_nodes_from(range(N))
    
    # Add edges with random weights to ensure randomness
    # We'll use the same edges as in the original function, but with random weights
    for i, j in edges:
        if (i, j) in result_offsets or (j, i) in result_offsets:
            # Use random weight between 0 and 1 for all edges
            weight = random.random()
            G.add_edge(i, j, weight=weight)
    
    # Check for connected components first
    components = list(nx.connected_components(G))
    
    if len(components) > 1:
        print(f"Warning: Graph has {len(components)} disconnected components")
        print("Adding cross-component edges to make the graph connected")
        
        # Create a list of potential edges that connect components
        cross_edges = []
        for comp1_idx, comp1 in enumerate(components):
            for comp2_idx, comp2 in enumerate(components):
                if comp1_idx >= comp2_idx:  # Avoid duplicate pairs
                    continue
                    
                # Find all potential edges between these components
                for i in comp1:
                    for j in comp2:
                        if (i, j) in result_offsets or (j, i) in result_offsets:
                            # Use random weight for connecting edges too
                            weight = random.random()
                            cross_edges.append((i, j, weight))
        
        # Shuffle the cross-component edges to introduce more randomness
        random.shuffle(cross_edges)
        
        # Build a forest connecting all components
        forest = nx.Graph()
        forest.add_nodes_from(range(N))
        
        # Add component internal edges with shuffled weights to introduce randomness
        for comp in components:
            subgraph = G.subgraph(comp)
            # Re-assign random weights to edges
            subgraph_edges = list(subgraph.edges())
            random.shuffle(subgraph_edges)
            for i, j in subgraph_edges:
                forest.add_edge(i, j, weight=random.random())
        
        # Track which super-nodes (components) are connected
        component_membership = {}
        for comp_idx, comp in enumerate(components):
            for node in comp:
                component_membership[node] = comp_idx
        
        # Initialize union-find data structure for components
        component_groups = {i: i for i in range(len(components))}
        
        def find(x):
            if component_groups[x] != x:
                component_groups[x] = find(component_groups[x])
            return component_groups[x]
        
        def union(x, y):
            component_groups[find(x)] = find(y)
        
        # Shuffle cross edges to add randomness
        random.shuffle(cross_edges)
        
        # Add cross-component edges in random order
        for i, j, weight in cross_edges:
            comp_i = component_membership[i]
            comp_j = component_membership[j]
            
            if find(comp_i) != find(comp_j):
                forest.add_edge(i, j, weight=weight)
                union(comp_i, comp_j)
                
                # Check if all components are now connected
                root_count = len(set(find(c) for c in range(len(components))))
                if root_count == 1:
                    break
    
        # Update G to the connected graph
        G = forest
    
    # Shuffle the edge weights to create randomness
    for i, j in G.edges():
        G[i][j]['weight'] = random.random()
    
    # Now compute spanning tree with randomized weights
    rst_edges = list(nx.minimum_spanning_edges(G, data=False))
    
    # Create a new dictionary with only the random spanning tree offsets
    rst_offsets = {}
    for i, j in rst_edges:
        if (i, j) in result_offsets:
            rst_offsets[(i, j)] = result_offsets[(i, j)]
        elif (j, i) in result_offsets:
            rst_offsets[(i, j)] = -result_offsets[(j, i)]
    
    print(f"Selected {len(rst_edges)} random spanning tree edges out of {len(edges)} total edges")
    return rst_edges, rst_offsets

def ensure_connected_graph(video_names, edges, result_offsets):
    """
    Ensure that the graph formed by edges is connected.
    If not, add minimal edges to make it connected.
    
    Args:
        video_names (list): List of video names.
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        
    Returns:
        tuple: (connected_edges, connected_offsets) - Connected edge pairs and their offsets.
    """
    # Create a graph
    G = nx.Graph()
    N = len(video_names)
    G.add_nodes_from(range(N))
    
    # Add existing edges
    for i, j in edges:
        if (i, j) in result_offsets:
            G.add_edge(i, j)
        elif (j, i) in result_offsets:
            G.add_edge(i, j)
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    if len(components) == 1:
        # Already connected
        return edges, result_offsets
    
    print(f"Graph has {len(components)} disconnected components, adding edges to connect")
    
    # Create a list of all possible edges that can connect components
    connecting_edges = []
    for comp1_idx, comp1 in enumerate(components):
        for comp2_idx, comp2 in enumerate(components):
            if comp1_idx >= comp2_idx:  # Avoid duplicate pairs
                continue
                
            # Find all potential edges between components
            for i in comp1:
                for j in comp2:
                    edge_key = None
                    weight = float('inf')
                    
                    if (i, j) in result_offsets:
                        edge_key = (i, j)
                        weight = abs(result_offsets[(i, j)])
                    elif (j, i) in result_offsets:
                        edge_key = (j, i)
                        weight = abs(result_offsets[(j, i)])
                    
                    if edge_key:
                        connecting_edges.append((edge_key[0], edge_key[1], weight))
    
    # Sort by weight (prefer smaller offsets)
    connecting_edges.sort(key=lambda x: x[2])
    
    # Use Kruskal's algorithm to connect components
    connected_edges = list(edges)
    connected_offsets = dict(result_offsets)
    
    # Create a union-find data structure for components
    component_map = {}
    for comp_idx, comp in enumerate(components):
        for node in comp:
            component_map[node] = comp_idx
    
    component_roots = {i: i for i in range(len(components))}
    
    def find(x):
        if component_roots[x] != x:
            component_roots[x] = find(component_roots[x])
        return component_roots[x]
    
    def union(x, y):
        component_roots[find(x)] = find(y)
    
    # Add minimal edges to connect components
    added_edges = []
    for i, j, _ in connecting_edges:
        comp_i = component_map[i]
        comp_j = component_map[j]
        
        if find(comp_i) != find(comp_j):
            # Add this edge to connect components
            if (i, j) not in connected_edges and (j, i) not in connected_edges:
                connected_edges.append((i, j))
                
                if (i, j) in result_offsets:
                    connected_offsets[(i, j)] = result_offsets[(i, j)]
                else:
                    connected_offsets[(i, j)] = -result_offsets[(j, i)]
                
                added_edges.append((i, j))
            
            union(comp_i, comp_j)
            
            # Check if all components are connected
            if len(set(find(comp) for comp in range(len(components)))) == 1:
                break
    
    print(f"Added {len(added_edges)} edges to ensure connectivity: {added_edges}")
    return connected_edges, connected_offsets


def estimate_global_offsets(video_names, edges, result_offsets):
    """
    Estimate global offsets using least squares.
    
    Args:
        video_names (list): List of video names.
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        
    Returns:
        np.ndarray: Array of predicted offsets.
    """
    # Create a graph for analysis
    G = nx.Graph()
    N = len(video_names)
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    
    # Find isolated videos
    isolated = list(nx.isolates(G))
    if isolated:
        print("Isolated videos (no pairwise data):",
            [video_names[i] for i in isolated])

    # Run global optimization with least squares
    print("RUNNING GLOBAL OPTIMIZATION")
    
    pred_offsets = np.full(N, np.nan)
    # Process each connected component separately
    num_comps = len(list(nx.connected_components(G)))
    print(f"Number of connected components: {num_comps}")
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            # singleton: skip, leave as nan
            continue

        # Subgraph of this component
        subG = G.subgraph(comp)

        # Pick reference node = highest degree in this component
        ref, _ = max(subG.degree, key=lambda x: x[1])
        # we'll enforce x_ref = 0
        pred_offsets[ref] = 0.0

        # Build list of unknown nodes (all except ref)
        unknowns = sorted(set(comp) - {ref})
        idx_map = {node: idx for idx, node in enumerate(unknowns)}
        m = len(unknowns)

        # Collect equations A x = b
        rows = []
        b = []
        for i, j in edges:
            # only use edges inside this component
            if i not in comp or j not in comp:
                continue
            d_ij = result_offsets.get((i, j))
            if d_ij is None:
                continue
            # row vector of length m
            row = np.zeros(m, dtype=np.float64)
            if i != ref:
                row[idx_map[i]] = 1
            # if i == ref, x_ref term moves to RHS (but since x_ref=0 it vanishes)
            if j != ref:
                row[idx_map[j]] -= 1
            # i==ref ⇒ row is +1 at j; j==ref ⇒ row is −1 at i
            rows.append(row)
            b.append(d_ij)

        A = np.vstack(rows)      # shape (E', m)
        b = np.array(b, dtype=np.float64)

        # Solve in least‐squares sense
        x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)

        # Write back into offsets
        for node, val in zip(unknowns, x_hat):
            pred_offsets[node] = val
    return pred_offsets


def huber_weight(r, delta):
    """
    Huber weight ψ(r)/r for residual r and threshold δ.
    
    Args:
        r (float): Residual.
        delta (float): Threshold.
        
    Returns:
        float: Weight.
    """
    return 1.0 if abs(r) <= delta else (delta / abs(r))


def estimate_global_offsets_robust(
    video_names, edges, result_offsets,
    loss='huber', delta=1.0, max_iter=10, tol=1e-6
):
    """
    Estimate global offsets using robust Huber least squares on each connected component.
    
    Args:
        video_names (list): List of video names.
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        loss (str): Loss function to use.
        delta (float): Huber threshold.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        
    Returns:
        np.ndarray: Array of predicted offsets.
    """
    N = len(video_names)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    # Report isolated videos
    isolated = list(nx.isolates(G))
    if isolated:
        print("Isolated videos (no pairwise data):",
              [video_names[i] for i in isolated])

    print("RUNNING ROBUST GLOBAL OPTIMIZATION")

    pred_offsets = np.full(N, np.nan)

    for comp in nx.connected_components(G):
        if len(comp) == 1:
            # no equations, leave as NaN
            continue

        # Pick reference = highest‐degree node
        subG = G.subgraph(comp)
        ref = max(subG.degree, key=lambda x: x[1])[0]
        pred_offsets[ref] = 0.0

        # Unknowns = all except ref
        unknowns = sorted(comp - {ref})
        idx_map = {node: idx for idx, node in enumerate(unknowns)}
        m = len(unknowns)

        # Build initial A, b over this component
        rows, b_vals = [], []
        for (i, j) in edges:
            if i not in comp or j not in comp:
                continue
            # Get measurement d_ij; if only (j,i) stored, flip sign
            if (i, j) in result_offsets:
                d_ij = result_offsets[(i, j)]
            elif (j, i) in result_offsets:
                d_ij = -result_offsets[(j, i)]
            else:
                continue

            # Assemble row: x_i - x_j = d_ij
            row = np.zeros(m, dtype=np.float64)
            if i != ref:
                row[idx_map[i]] = 1.0
            if j != ref:
                row[idx_map[j]] = -1.0

            rows.append(row)
            b_vals.append(d_ij)

        A = np.vstack(rows)        # shape (E', m)
        b = np.array(b_vals)       # shape (E',)

        # Initial (non‑robust) LS solve
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        # IRLS (Iteratively Reweighted Least Squares)
        for it in range(max_iter):
            # Residuals r = A x - b
            r = A.dot(x) - b
            # Weights w_k = huber_weight(r_k)
            w = np.array([huber_weight(ri, delta) for ri in r])
            # Form W^{1/2} A, W^{1/2} b
            W_sqrt = np.sqrt(np.diag(w))
            A_w = W_sqrt.dot(A)
            b_w = W_sqrt.dot(b)

            x_new = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        # Write back into pred_offsets
        for node, val in zip(unknowns, x):
            pred_offsets[node] = val

    return pred_offsets


def fill_missing_offsets(
    video_names, edges, result_offsets, pred_offsets_init,
    result_files, delta=1.0, max_iter=10, tol=1e-6
):
    """
    Fill missing entries in pred_offsets_init.
    
    Args:
        video_names (list): List of video names.
        edges (list): List of edge pairs (i, j).
        result_offsets (dict): Dictionary of offsets for each edge.
        pred_offsets_init (np.ndarray): Initial predicted offsets.
        result_files (list): List of result files.
        delta (float): Huber threshold.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        
    Returns:
        np.ndarray: Array of predicted offsets with missing values filled.
    """
    pred_offsets = pred_offsets_init.astype(np.float64).copy()
    
    if not np.isnan(pred_offsets).any():
        return pred_offsets  # No missing entries

    # Identify videos with missing offsets
    fill_indices = np.where(np.isnan(pred_offsets))[0]
    fill_edges = []
    fill_result_offsets = {}
    
    for idx in fill_indices:
        fill_video_name = video_names[idx]
        fill_result_files = [result_file for result_file in result_files if fill_video_name in result_file]
        fill_result_files.sort()
        
        for fill_result_file in fill_result_files:
            video1_name, video2_name = os.path.basename(os.path.dirname(fill_result_file)).split('__')
            i = video_names.index(video1_name)
            j = video_names.index(video2_name)
            
            if not (np.isnan(pred_offsets[i]) and np.isnan(pred_offsets[j])):
                fill_edges.append((i, j))
                with open(fill_result_file, 'rb') as f:
                    fill_result = pickle.load(f)
                
                offset_error_list = fill_result["offset_error_list"]
                if np.isnan(offset_error_list).all():
                    print(f"All offsets are nan for {video1_name} and {video2_name}, skipping")
                    continue
                    
                offsets = fill_result["offsets"]
                pred_offset = offsets[np.nanargmin(offset_error_list)]
                
                fill_result_offsets[(i, j)] = pred_offset
    
    # Fill missing offsets with optimization
    if fill_edges:
        fixed = ~np.isnan(pred_offsets)
        unknowns = np.where(np.isnan(pred_offsets))[0]
        idx_map = {node: idx for idx, node in enumerate(unknowns)}
        m = len(unknowns)

        rows = []
        b_vals = []
        
        for i, j in fill_edges:
            if (i, j) in fill_result_offsets:
                d_ij = fill_result_offsets[(i, j)]
            elif (j, i) in fill_result_offsets:
                d_ij = -fill_result_offsets[(j, i)]
            else:
                continue

            if not fixed[i] and not fixed[j]:
                row = np.zeros(m)
                row[idx_map[i]] = 1.0
                row[idx_map[j]] = -1.0
                b_val = d_ij
            elif not fixed[i] and fixed[j]:
                row = np.zeros(m)
                row[idx_map[i]] = 1.0
                b_val = d_ij + pred_offsets[j]
            elif fixed[i] and not fixed[j]:
                row = np.zeros(m)
                row[idx_map[j]] = -1.0
                b_val = d_ij - pred_offsets[i]
            else:
                continue

            rows.append(row)
            b_vals.append(b_val)

        if not rows:
            # No valid equations to solve — return as-is
            print("No valid equations to solve for missing offsets")
            pred_offsets[np.isnan(pred_offsets)] = 0
            return pred_offsets

        A = np.vstack(rows)
        b = np.array(b_vals)

        # Initial solve
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        # IRLS iterations
        for _ in range(max_iter):
            r = A @ x - b
            w = np.array([huber_weight(ri, delta) for ri in r])
            W_sqrt = np.sqrt(np.diag(w))
            A_w = W_sqrt @ A
            b_w = W_sqrt @ b
            x_new = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        # Update offsets
        for i, node in enumerate(unknowns):
            pred_offsets[node] = x[i]
    
    # If any offsets still missing, set to zero
    if np.isnan(pred_offsets).any():
        missing_count = np.sum(np.isnan(pred_offsets))
        print(f"{missing_count} offsets remain NaN after filling, setting to default value")
        pred_offsets[np.isnan(pred_offsets)] = 0
    
    return pred_offsets


def load_results(result_roots, pkl_name, group_name=None, exclusions=None, load_full=False, fill_type=None):
    """
    Load results from result roots. Handles duplicate sport names by creating unique keys.
    
    Args:
        result_roots (list): List of result root directories.
        pkl_name (str): Name of pickle file.
        group_name (list, optional): List of group names to filter by.
        exclusions (list, optional): List of video names to exclude.
        load_full (bool, optional): Whether to load full results. Default is False.
        fill_type (str, optional): Type of fill method to use when pred_offset is None.
                                  Options: "heuristic", "min". Default is None.
        
    Returns:
        dict: Dictionary of sports metrics with unique keys.
    """
    sport_metrics = {}
    sport_name_counts = {}  # To keep track of how many times each sport name has been seen
    
    for result_root in result_roots:
        sport_dirs = glob.glob(os.path.join(result_root, "*"))
        sport_dirs = [sport for sport in sport_dirs if os.path.isdir(sport)]
        
        for sport_dir in sport_dirs:
            sport_name = os.path.basename(sport_dir)
            
            # Filter by group name if specified
            if group_name is not None and sport_name not in group_name:
                continue
            
            # Create unique key for sport_name if it already exists
            if sport_name in sport_metrics:
                # Increment the counter for this sport name
                if sport_name not in sport_name_counts:
                    sport_name_counts[sport_name] = 1
                sport_name_counts[sport_name] += 1
                
                # Create a new unique key with a suffix
                unique_sport_name = f"{sport_name}{sport_name_counts[sport_name]}"
            else:
                # First time seeing this sport name
                sport_name_counts[sport_name] = 0
                unique_sport_name = sport_name
            
            # Initialize data structure for this sport
            sport_metrics[unique_sport_name] = {
                "absolute_errors": [], 
                "unregistered_videos": []
            }
            
            # Get result directories
            result_dirs = glob.glob(os.path.join(sport_dir, '*'))
            result_dirs = [result_dir for result_dir in result_dirs if os.path.isdir(result_dir)]
            
            # Apply exclusions if specified
            if exclusions:
                result_dirs = [
                    result_dir for result_dir in result_dirs 
                    if not any(exclusion in result_dir for exclusion in exclusions)
                ]
                
            result_dirs.sort()
            print(f"Number of result dirs in {sport_name}: {len(result_dirs)}")
            
            # Get video names
            video_dirs = set()
            for result_dir in result_dirs:
                result_name = os.path.basename(result_dir)
                video_dir1, video_dir2 = result_name.split("__")
                video_dirs.add(video_dir1)
                video_dirs.add(video_dir2)
            video_dirs = list(video_dirs)
            video_dirs.sort()
            
            video_names = list(get_unique_video_names(result_dirs))
            video_names.sort()
            print(f"Total {len(video_names)} videos within sequence: {video_names}")
            
            # Get result files
            result_files = glob.glob(os.path.join(sport_dir, '*', pkl_name))
            print(f"Number of candidate files: {len(result_files)}/{len(result_dirs)}")
            
            # Find edges and offsets
            edges = []
            result_offsets = {}
            gt_video_offsets = {}
            
            for result_dir in result_dirs:
                video1, video2 = os.path.basename(result_dir).split('__')
                
                # Extract frame ranges from video names
                # remove "i10" for pan171204_pose3_cam06_1188_1308_i10
                video1_start, video1_end = re.sub(r'_i\d+$', '', video1).split('_')[-2:] # video1.split('_')[-2:]
                video1_start, video1_end = int(video1_start), int(video1_end)
                video2_start, video2_end = re.sub(r'_i\d+$', '', video2).split('_')[-2:]# video2.split('_')[-2:]
                video2_start, video2_end = int(video2_start), int(video2_end)
                
                # Store GT offsets
                if video1 not in gt_video_offsets:
                    gt_video_offsets[video1] = {"start": video1_start, "end": video1_end}
                if video2 not in gt_video_offsets:
                    gt_video_offsets[video2] = {"start": video2_start, "end": video2_end}
                    
                # Load result
                result_path = os.path.join(result_dir, pkl_name)
                if not os.path.exists(result_path):
                    continue
                    
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                    pred_offset = result["pred_offset"]
                    
                    # If load_full is True, load additional data
                    if load_full:
                        offset_error_list = result.get("offset_error_list", None)
                        offsets = result.get("offsets", None)
                        overlap_len_list = result.get("overlap_len_list", None)
                    
                    # Try to fill pred_offset if it's None and fill_type is specified
                    if pred_offset is None and fill_type is not None:
                        if load_full and offsets is not None and offset_error_list is not None:
                            if fill_type == "heuristic" and overlap_len_list is not None:
                                pred_offset = offsets[np.nanargmin(offset_error_list / overlap_len_list)]
                            elif fill_type == "min":
                                pred_offset = offsets[np.nanargmin(offset_error_list)]
                        else:
                            continue
                    elif pred_offset is None:
                        continue
                    
                # Construct edges
                i = video_names.index(video1)
                j = video_names.index(video2)
                edges.append((i, j))
                result_offsets[(i, j)] = pred_offset
                
            # Store results in sport_metrics dictionary
            sport_metrics[unique_sport_name]["edges"] = edges
            sport_metrics[unique_sport_name]["result_offsets"] = result_offsets
            sport_metrics[unique_sport_name]["gt_video_offsets"] = gt_video_offsets
            sport_metrics[unique_sport_name]["video_names"] = video_names
            sport_metrics[unique_sport_name]["result_files"] = result_files
            
            # Store original sport name for reference
            sport_metrics[unique_sport_name]["original_sport_name"] = sport_name
    
    return sport_metrics


# Modify compute_global_offsets to run multiple iterations:
def compute_global_offsets(sport_metrics, est_choice="irls", max_iter=10, delta=1.0, 
                          fill_missing_flag=False, pair_ratio=1.0, use_rst=False, n_times=1, base_seed=42):
    """
    Compute global offsets for each sport.
    
    Args:
        sport_metrics (dict): Dictionary of sports metrics.
        est_choice (str): Estimation choice, either "irls" or "lsq".
        max_iter (int): Maximum number of iterations for robust optimization.
        delta (float): Huber threshold for robust optimization.
        fill_missing_flag (bool): Whether to fill missing offsets.
        pair_ratio (float): Ratio of pairs to use for global optimization (0 to 1).
        use_rst (bool): Whether to use random spanning tree for pair selection.
        n_times (int): Number of times to run the experiment with different seeds.
        base_seed (int): Base random seed for reproducibility.
        
    Returns:
        dict: Updated sports metrics with predicted offsets.
    """
    # Check if we need multiple runs
    run_multiple = (pair_ratio < 1.0 or use_rst) and n_times > 1
    
    for sport_name in sport_metrics:
        if sport_name == "overall":
            continue
            
        print(f"\nProcessing {sport_name}...")
        
        video_names = sport_metrics[sport_name]["video_names"]
        edges = sport_metrics[sport_name]["edges"]
        result_offsets = sport_metrics[sport_name]["result_offsets"]
        
        # Original edge and offset counts
        original_edge_count = len(edges)
        valid_edge_count = sum(1 for edge in edges if (edge[0], edge[1]) in result_offsets or (edge[1], edge[0]) in result_offsets)
        
        print(f"Original data: {valid_edge_count} valid edges out of {original_edge_count} total edges")
        
        # If running multiple times, store results for each run
        if run_multiple:
            sport_metrics[sport_name]["multi_run_offsets"] = []
            print(f"Running {n_times} iterations with different seeds")
            
            for i in range(n_times):
                current_seed = base_seed + i
                print(f"\nIteration {i+1}/{n_times} with seed {current_seed}")
                
                # Set seed for this iteration
                random.seed(current_seed)
                
                # Select pairs based on strategy
                if use_rst:
                    print(f"Using RST-based pair selection for {sport_name}")
                    selected_edges, selected_offsets = select_pairs_by_random_st(video_names, edges, result_offsets)
                    selection_method = f"RST-seed{current_seed}"
                elif pair_ratio < 1.0:
                    print(f"Using random pair selection with ratio {pair_ratio:.2f} for {sport_name}")
                    selected_edges, selected_offsets = select_pairs_by_ratio(edges, result_offsets, video_names, pair_ratio, seed=current_seed)
                    selection_method = f"Ratio-{pair_ratio:.2f}-seed{current_seed}"
                else:
                    print(f"Using all available pairs for {sport_name}")
                    selected_edges, selected_offsets = edges, result_offsets
                    selection_method = "All-Pairs"
                
                # Ensure graph is connected
                connected_edges, connected_offsets = ensure_connected_graph(
                    video_names, selected_edges, selected_offsets)
                
                if len(connected_edges) > len(selected_edges):
                    print(f"Added {len(connected_edges) - len(selected_edges)} edges to ensure connectivity")
                
                # Compute global offsets
                if est_choice == "lsq":
                    print(f"Using least squares optimization for {sport_name}")
                    pred_offsets = estimate_global_offsets(video_names, connected_edges, connected_offsets)
                else:
                    print(f"Using robust optimization (IRLS) for {sport_name}")
                    pred_offsets = estimate_global_offsets_robust(
                        video_names, connected_edges, connected_offsets, max_iter=max_iter, delta=delta)
                
                # Fill missing offsets if requested
                if fill_missing_flag and np.isnan(pred_offsets).any():
                    missing_count = np.sum(np.isnan(pred_offsets))
                    print(f"Filling {missing_count} missing offsets")
                    
                    result_files = sport_metrics[sport_name]["result_files"]
                    pred_offsets = fill_missing_offsets(
                        video_names, edges, result_offsets, pred_offsets, 
                        result_files, delta=delta
                    )
                
                # Store this iteration's predicted offsets
                sport_metrics[sport_name]["multi_run_offsets"].append({
                    "seed": current_seed,
                    "pred_offsets": pred_offsets.copy(),
                    "selection_method": selection_method,
                    "selected_edge_count": len(selected_edges),
                    "connected_edge_count": len(connected_edges)
                })
            
            # Use the first run's offsets as the "main" pred_offsets for backward compatibility
            sport_metrics[sport_name]["pred_offsets"] = sport_metrics[sport_name]["multi_run_offsets"][0]["pred_offsets"]
            sport_metrics[sport_name]["selection_method"] = sport_metrics[sport_name]["multi_run_offsets"][0]["selection_method"]
            sport_metrics[sport_name]["selected_edge_count"] = sport_metrics[sport_name]["multi_run_offsets"][0]["selected_edge_count"]
            sport_metrics[sport_name]["connected_edge_count"] = sport_metrics[sport_name]["multi_run_offsets"][0]["connected_edge_count"]
            
        else:
            # Original single-run code
            # Select pairs based on strategy
            if use_rst:
                print(f"Using RST-based pair selection for {sport_name}")
                selected_edges, selected_offsets = select_pairs_by_random_st(video_names, edges, result_offsets)
                selection_method = "RST"
            elif pair_ratio < 1.0:
                print(f"Using random pair selection with ratio {pair_ratio:.2f} for {sport_name}")
                selected_edges, selected_offsets = select_pairs_by_ratio(edges, result_offsets, video_names, pair_ratio)
                selection_method = f"Ratio-{pair_ratio:.2f}"
            else:
                print(f"Using all available pairs for {sport_name}")
                selected_edges, selected_offsets = edges, result_offsets
                selection_method = "All-Pairs"
            
            # Ensure graph is connected
            connected_edges, connected_offsets = ensure_connected_graph(
                video_names, selected_edges, selected_offsets)
            
            if len(connected_edges) > len(selected_edges):
                print(f"Added {len(connected_edges) - len(selected_edges)} edges to ensure connectivity")
            
            # Compute global offsets
            if est_choice == "lsq":
                print(f"Using least squares optimization for {sport_name}")
                pred_offsets = estimate_global_offsets(video_names, connected_edges, connected_offsets)
            else:
                print(f"Using robust optimization (IRLS) for {sport_name}")
                pred_offsets = estimate_global_offsets_robust(
                    video_names, connected_edges, connected_offsets, max_iter=max_iter, delta=delta)
            
            # Fill missing offsets if requested
            if fill_missing_flag and np.isnan(pred_offsets).any():
                missing_count = np.sum(np.isnan(pred_offsets))
                print(f"Filling {missing_count} missing offsets")
                
                result_files = sport_metrics[sport_name]["result_files"]
                pred_offsets = fill_missing_offsets(
                    video_names, edges, result_offsets, pred_offsets, 
                    result_files, delta=delta
                )
            
            # Store predicted offsets
            sport_metrics[sport_name]["pred_offsets"] = pred_offsets
            
            # Store statistics about pair selection for debugging/analysis
            sport_metrics[sport_name]["selection_method"] = selection_method
            sport_metrics[sport_name]["selected_edge_count"] = len(selected_edges)
            sport_metrics[sport_name]["connected_edge_count"] = len(connected_edges)
        
        # Common storage for both single and multi-run
        sport_metrics[sport_name]["original_edge_count"] = original_edge_count
        sport_metrics[sport_name]["valid_edge_count"] = valid_edge_count
        
        # Report success
        print(f"Successfully computed global offsets for {sport_name}")
        if "pred_offsets" in sport_metrics[sport_name] and np.isnan(sport_metrics[sport_name]["pred_offsets"]).any():
            nan_count = np.sum(np.isnan(sport_metrics[sport_name]["pred_offsets"]))
            print(f"Warning: {nan_count} videos have NaN offsets in the first run")
    
    return sport_metrics

# Modify evaluate_results to handle multiple runs:
def evaluate_results(sport_metrics, dataset_fps):
    """
    Evaluate results and compute error metrics.
    
    Args:
        sport_metrics (dict): Dictionary of sports metrics.
        dataset_fps (float): Frames per second of the dataset.
        
    Returns:
        dict: Updated sports metrics with evaluation results.
    """
    # Store dataset_fps in each sport's metrics for later use
    for sport_name in sport_metrics:
        if sport_name != "overall":
            sport_metrics[sport_name]["dataset_fps"] = dataset_fps
    
    # Initialize overall metrics
    sport_metrics['overall'] = {
        "absolute_errors": [], 
        "unregistered_videos": []
    }
    
    # Check if we're running with multiple iterations
    multi_run = any("multi_run_offsets" in sport_metrics[sport_name] 
                   for sport_name in sport_metrics if sport_name != "overall")
    
    if multi_run:
        sport_metrics['overall']["multi_run_metrics"] = []
    
    for sport_name in list(sport_metrics.keys()):
        if sport_name == "overall":
            continue
            
        video_names = sport_metrics[sport_name]["video_names"]
        gt_video_offsets = sport_metrics[sport_name]["gt_video_offsets"]
        
        # Handle multiple runs if present
        if "multi_run_offsets" in sport_metrics[sport_name]:
            sport_metrics[sport_name]["multi_run_metrics"] = []
            
            for run_idx, run_data in enumerate(sport_metrics[sport_name]["multi_run_offsets"]):
                pred_offsets = run_data["pred_offsets"]
                
                # Find reference video (the one with offset = 0)
                ref_video_idx = np.where(pred_offsets == 0)[0]
                if len(ref_video_idx) < 1:
                    print(f"No reference video found for {sport_name} run {run_idx}, skipping evaluation")
                    continue
                    
                if len(ref_video_idx) > 1:
                    print(f"Warning: Multiple reference videos found for {sport_name} run {run_idx}: {ref_video_idx}")
                    ref_video_idx = ref_video_idx[0:1]
                    
                ref_video_name = video_names[ref_video_idx[0]]
                
                # Compute ground truth absolute offsets
                gt_abs_offsets = np.empty(len(video_names), dtype=np.float32)
                for idx in range(len(video_names)):
                    gt_abs_offsets[idx] = gt_video_offsets[video_names[idx]]["start"] - gt_video_offsets[ref_video_name]["start"]
                
                # Evaluate absolute errors (per video)
                absolute_errors = []
                unregistered_videos = []
                
                for video_idx in range(len(video_names)):
                    if np.isnan(pred_offsets[video_idx]):
                        unregistered_videos.append(video_names[video_idx])
                        continue
                        
                    # Convert frame error to milliseconds
                    abs_err_frames = abs(pred_offsets[video_idx] - gt_abs_offsets[video_idx])
                    abs_err_ms = (abs_err_frames / dataset_fps) * 1000
                    absolute_errors.append(abs_err_ms)
                
                # Store metrics for this run
                run_metrics = {
                    "seed": run_data["seed"],
                    "absolute_errors": absolute_errors,
                    "unregistered_videos": unregistered_videos
                }
                sport_metrics[sport_name]["multi_run_metrics"].append(run_metrics)
                
                # For the first run, also store in the main metrics for backward compatibility
                if run_idx == 0:
                    sport_metrics[sport_name]["absolute_errors"] = absolute_errors
                    sport_metrics[sport_name]["unregistered_videos"] = unregistered_videos
                    
                    # Add to overall metrics
                    sport_metrics['overall']["absolute_errors"].extend(absolute_errors)
                    sport_metrics['overall']["unregistered_videos"].extend(unregistered_videos)
            
            # Calculate pairwise errors for each run
            sport_metrics = calculate_pairwise_errors_multi_run(sport_metrics, sport_name, dataset_fps)
            
        else:
            # Original single-run code
            pred_offsets = sport_metrics[sport_name]["pred_offsets"]
            
            # Find reference video (the one with offset = 0)
            ref_video_idx = np.where(pred_offsets == 0)[0]
            if len(ref_video_idx) < 1:
                print(f"No reference video found for {sport_name}, skipping evaluation")
                continue
                
            if len(ref_video_idx) > 1:
                print(f"Warning: Multiple reference videos found for {sport_name}: {ref_video_idx}")
                ref_video_idx = ref_video_idx[0:1]
                
            ref_video_name = video_names[ref_video_idx[0]]
            
            # Compute ground truth absolute offsets
            gt_abs_offsets = np.empty(len(video_names), dtype=np.float32)
            for idx in range(len(video_names)):
                gt_abs_offsets[idx] = gt_video_offsets[video_names[idx]]["start"] - gt_video_offsets[ref_video_name]["start"]
            
            # Evaluate absolute errors (per video)
            sport_metrics[sport_name]["absolute_errors"] = []
            sport_metrics[sport_name]["unregistered_videos"] = []
            
            for video_idx in range(len(video_names)):
                if np.isnan(pred_offsets[video_idx]):
                    sport_metrics[sport_name]["unregistered_videos"].append(video_names[video_idx])
                    continue
                    
                # Convert frame error to milliseconds
                abs_err_frames = abs(pred_offsets[video_idx] - gt_abs_offsets[video_idx])
                abs_err_ms = (abs_err_frames / dataset_fps) * 1000
                sport_metrics[sport_name]["absolute_errors"].append(abs_err_ms)
            
            # Add to overall metrics
            sport_metrics['overall']["absolute_errors"].extend(sport_metrics[sport_name]["absolute_errors"])
            sport_metrics['overall']["unregistered_videos"].extend(sport_metrics[sport_name]["unregistered_videos"])
    
    # Calculate pairwise errors for all sports (for single run or first run of multi-run)
    sport_metrics = calculate_pairwise_errors(sport_metrics)
    
    # Compile overall multi-run metrics if needed
    if multi_run:
        # Gather all run metrics across all sports
        all_runs_metrics = []
        for sport_name in sport_metrics:
            if sport_name == "overall" or "multi_run_metrics" not in sport_metrics[sport_name]:
                continue
            
            for run_metrics in sport_metrics[sport_name]["multi_run_metrics"]:
                if "pairwise_errors_ms" in run_metrics:
                    all_runs_metrics.append({
                        "sport": sport_name,
                        "seed": run_metrics["seed"],
                        "absolute_errors": run_metrics["absolute_errors"],
                        "unregistered_videos": run_metrics["unregistered_videos"],
                        "pairwise_errors_ms": run_metrics["pairwise_errors_ms"],
                        "unregistered_pairs": run_metrics["unregistered_pairs"],
                        "total_pairs": run_metrics["total_pairs"]
                    })
        
        # Group by seed to get overall metrics for each run
        seeds = sorted(set(run["seed"] for run in all_runs_metrics))
        for seed in seeds:
            seed_runs = [run for run in all_runs_metrics if run["seed"] == seed]
            
            # Combine metrics across sports for this seed
            overall_run = {
                "seed": seed,
                "absolute_errors": [],
                "unregistered_videos": [],
                "pairwise_errors_ms": [],
                "unregistered_pairs": [],
                "total_pairs": 0
            }
            
            for run in seed_runs:
                overall_run["absolute_errors"].extend(run["absolute_errors"])
                overall_run["unregistered_videos"].extend(run["unregistered_videos"])
                overall_run["pairwise_errors_ms"].extend(run["pairwise_errors_ms"])
                overall_run["unregistered_pairs"].extend(run["unregistered_pairs"])
                overall_run["total_pairs"] += run["total_pairs"]
            
            sport_metrics["overall"]["multi_run_metrics"].append(overall_run)
    
    return sport_metrics

# New function to calculate pairwise errors for multiple runs
def calculate_pairwise_errors_multi_run(sport_metrics, sport_name, dataset_fps):
    """
    Calculate pairwise errors from predicted offsets for each run in a sport.
    
    Args:
        sport_metrics (dict): Dictionary of sports metrics.
        sport_name (str): Name of the sport to process.
        dataset_fps (float): Frames per second of the dataset.
        
    Returns:
        dict: Updated sports metrics with pairwise errors for each run.
    """
    video_names = sport_metrics[sport_name]["video_names"]
    gt_video_offsets = sport_metrics[sport_name]["gt_video_offsets"]
    
    # Process each run
    for run_idx, run_metrics in enumerate(sport_metrics[sport_name]["multi_run_metrics"]):
        pred_offsets = sport_metrics[sport_name]["multi_run_offsets"][run_idx]["pred_offsets"]
        
        # Initialize pairwise error lists for this run
        run_metrics["pairwise_errors_ms"] = []
        run_metrics["unregistered_pairs"] = []
        run_metrics["total_pairs"] = 0
        
        # Calculate pairwise errors for all combinations
        for i, j in combinations(range(len(video_names)), 2):
            # Skip if either video has NaN offset
            if np.isnan(pred_offsets[i]) or np.isnan(pred_offsets[j]):
                run_metrics["unregistered_pairs"].append((video_names[i], video_names[j]))
                continue
            
            # Calculate predicted pairwise offset
            pred_pair_offset = pred_offsets[i] - pred_offsets[j]
            
            # Calculate ground truth pairwise offset
            gt_i_start = gt_video_offsets[video_names[i]]["start"]
            gt_j_start = gt_video_offsets[video_names[j]]["start"]
            gt_pair_offset = gt_i_start - gt_j_start
            
            # Calculate error in frames and convert to ms
            error_frames = abs(pred_pair_offset - gt_pair_offset)
            error_ms = (error_frames / dataset_fps) * 1000
            
            run_metrics["pairwise_errors_ms"].append(error_ms)
        
        # Count total pairs
        n_videos = len(video_names)
        run_metrics["total_pairs"] = n_videos * (n_videos - 1) // 2
        
        # For the first run, also ensure we have the main pairwise metrics for backward compatibility
        if run_idx == 0 and ("pairwise_errors_ms" not in sport_metrics[sport_name]):
            sport_metrics[sport_name]["pairwise_errors_ms"] = run_metrics["pairwise_errors_ms"].copy()
            sport_metrics[sport_name]["unregistered_pairs"] = run_metrics["unregistered_pairs"].copy()
            sport_metrics[sport_name]["total_pairs"] = run_metrics["total_pairs"]
    
    return sport_metrics


def calculate_auc_with_unregistered(errors_ms, threshold_ms, unregistered_count, total_pairs):
    """
    Calculate AUC@threshold: area under cumulative accuracy curve up to a threshold,
    including unregistered pairs as infinite errors
    
    Args:
        errors_ms: List or array of error values (in milliseconds) for registered pairs
        threshold_ms: Upper threshold for AUC (in milliseconds)
        unregistered_count: Number of unregistered pairs
        total_pairs: Total number of pairs (registered + unregistered)
    
    Returns:
        auc: Area under the curve (0–100%)
    """
    if total_pairs == 0:
        return 0.0
    
    # When there are only unregistered pairs
    if len(errors_ms) == 0:
        return 0.0
    
    # Sort errors and create the CDF for registered pairs
    sorted_errors = np.sort(errors_ms)
    
    # Create y values (cumulative proportion) accounting for unregistered pairs
    # Unregistered pairs are treated as having infinite error
    registered_proportion = len(errors_ms) / total_pairs
    y = np.arange(1, len(sorted_errors) + 1) / total_pairs
    
    # Find index where errors exceed the threshold
    idx = np.searchsorted(sorted_errors, threshold_ms)
    
    if idx == 0:
        # No errors below threshold
        return 0.0
    
    # Truncate at threshold
    x_truncated = np.append(sorted_errors[:idx], [threshold_ms])
    y_truncated = np.append(y[:idx], [y[idx - 1]])
    
    # Calculate AUC (area under curve) up to threshold
    auc = 100 * np.trapz(y_truncated, x_truncated) / threshold_ms
    
    return auc

def calculate_pairwise_errors(sport_metrics):
    """
    Calculate pairwise errors from predicted offsets for each sport.
    
    Args:
        sport_metrics (dict): Dictionary of sports metrics.
        
    Returns:
        dict: Updated sports metrics with pairwise errors.
    """
    for sport_name in list(sport_metrics.keys()):
        if sport_name == "overall":
            continue
            
        video_names = sport_metrics[sport_name]["video_names"]
        pred_offsets = sport_metrics[sport_name]["pred_offsets"]
        gt_video_offsets = sport_metrics[sport_name]["gt_video_offsets"]
        
        # Initialize pairwise error lists
        sport_metrics[sport_name]["pairwise_errors_ms"] = []
        sport_metrics[sport_name]["unregistered_pairs"] = []
        sport_metrics[sport_name]["total_pairs"] = 0
        
        # Calculate pairwise errors for all combinations
        for i, j in combinations(range(len(video_names)), 2):
            # Skip if either video has NaN offset
            if np.isnan(pred_offsets[i]) or np.isnan(pred_offsets[j]):
                sport_metrics[sport_name]["unregistered_pairs"].append((video_names[i], video_names[j]))
                continue
            
            # Calculate predicted pairwise offset
            pred_pair_offset = pred_offsets[i] - pred_offsets[j]
            
            # Calculate ground truth pairwise offset
            gt_i_start = gt_video_offsets[video_names[i]]["start"]
            gt_j_start = gt_video_offsets[video_names[j]]["start"]
            gt_pair_offset = gt_i_start - gt_j_start
            
            # Calculate error in frames and convert to ms
            error_frames = abs(pred_pair_offset - gt_pair_offset)
            error_ms = (error_frames / sport_metrics[sport_name].get("dataset_fps", 20.0)) * 1000
            
            sport_metrics[sport_name]["pairwise_errors_ms"].append(error_ms)
        
        # Count total pairs
        n_videos = len(video_names)
        sport_metrics[sport_name]["total_pairs"] = n_videos * (n_videos - 1) // 2
        
        # Add to overall metrics (create if needed)
        if "pairwise_errors_ms" not in sport_metrics.get("overall", {}):
            sport_metrics.setdefault("overall", {})["pairwise_errors_ms"] = []
            sport_metrics["overall"]["unregistered_pairs"] = []
            sport_metrics["overall"]["total_pairs"] = 0
        
        sport_metrics["overall"]["pairwise_errors_ms"].extend(sport_metrics[sport_name]["pairwise_errors_ms"])
        sport_metrics["overall"]["unregistered_pairs"].extend(sport_metrics[sport_name]["unregistered_pairs"])
        sport_metrics["overall"]["total_pairs"] += sport_metrics[sport_name]["total_pairs"]
    
    return sport_metrics



# Enhanced summary table with multi-run statistics
def generate_enhanced_summary_table(sport_metrics, dataset_fps):
    """
    Generate an enhanced summary table of results with AUC metrics.
    
    Args:
        sport_metrics (dict): Dictionary of sports metrics.
        dataset_fps (float): Frames per second of the dataset.
        
    Returns:
        str: Formatted table.
    """
    # For storing summary statistics
    metrics_data = []
    
    # Check if we have multiple runs
    multi_run = "multi_run_metrics" in sport_metrics.get("overall", {})

    # Process each sport (sort to ensure "overall" comes last)
    for sport_name in sorted(sport_metrics.keys(), key=lambda x: "zzz" if x == "overall" else x):
        # Skip if no data or if sport_name has no video_names (except for overall)
        if sport_name not in sport_metrics:
            continue
        
        if sport_name != "overall" and "video_names" not in sport_metrics[sport_name]:
            continue
        
        # For overall, we need to calculate total videos differently
        if sport_name == "overall":
            total_videos = sum(len(sport_metrics[s]["video_names"]) for s in sport_metrics if s != "overall" and "video_names" in sport_metrics[s])
        else:
            total_videos = len(sport_metrics[sport_name]["video_names"])
        
        # Handle multi-run case
        if multi_run and "multi_run_metrics" in sport_metrics[sport_name]:
            # Calculate statistics across runs
            n_runs = len(sport_metrics[sport_name]["multi_run_metrics"])
            
            # Error statistics across runs
            all_mean_errors = []
            all_median_errors = []
            all_auc_100ms = []
            all_auc_500ms = []
            all_unreg_videos_pct = []
            all_unreg_pairs_pct = []
            
            for run_metrics in sport_metrics[sport_name]["multi_run_metrics"]:
                # Video-level metrics
                run_abs_errors = run_metrics["absolute_errors"]
                num_unreg_videos = len(run_metrics["unregistered_videos"])
                
                # Calculate statistics for individual videos
                if run_abs_errors:
                    mean_abs_error = np.mean(run_abs_errors)
                    median_abs_error = np.median(run_abs_errors)
                    all_mean_errors.append(mean_abs_error)
                    all_median_errors.append(median_abs_error)
                    all_unreg_videos_pct.append(100.0 * num_unreg_videos / total_videos)
                
                # Pairwise metrics
                if "pairwise_errors_ms" in run_metrics:
                    pairwise_errors = run_metrics["pairwise_errors_ms"]
                    unreg_pairs = len(run_metrics["unregistered_pairs"])
                    total_pairs = run_metrics["total_pairs"]
                    
                    # Calculate AUC at 100ms and 500ms
                    auc_100ms = calculate_auc_with_unregistered(pairwise_errors, 100, unreg_pairs, total_pairs)
                    auc_500ms = calculate_auc_with_unregistered(pairwise_errors, 500, unreg_pairs, total_pairs)
                    
                    all_auc_100ms.append(auc_100ms)
                    all_auc_500ms.append(auc_500ms)
                    all_unreg_pairs_pct.append(100.0 * unreg_pairs / total_pairs)
            
            # Calculate average statistics across runs
            if all_mean_errors:
                avg_mean_error = np.mean(all_mean_errors)
                avg_median_error = np.mean(all_median_errors)
                avg_unreg_videos_pct = np.mean(all_unreg_videos_pct)
                std_mean_error = np.std(all_mean_errors)
                std_median_error = np.std(all_median_errors)
            else:
                avg_mean_error = float('nan')
                avg_median_error = float('nan')
                avg_unreg_videos_pct = float('nan')
                std_mean_error = float('nan')
                std_median_error = float('nan')
                
            if all_auc_100ms:
                avg_auc_100ms = np.mean(all_auc_100ms)
                avg_auc_500ms = np.mean(all_auc_500ms)
                avg_unreg_pairs_pct = np.mean(all_unreg_pairs_pct)
                std_auc_100ms = np.std(all_auc_100ms)
                std_auc_500ms = np.std(all_auc_500ms)
            else:
                avg_auc_100ms = float('nan')
                avg_auc_500ms = float('nan')
                avg_unreg_pairs_pct = float('nan')
                std_auc_100ms = float('nan')
                std_auc_500ms = float('nan')
            
            # Add to metrics data with standard deviations
            if sport_name == "overall":
                sport_display = "OVERALL"
            else:
                sport_display = sport_name
                
            metrics_data.append([
                sport_display, 
                total_videos,
                f"{avg_unreg_videos_pct:.1f}%",
                f"{avg_mean_error:.2f}±{std_mean_error:.2f}" if not np.isnan(avg_mean_error) else "NaN",
                f"{avg_median_error:.2f}±{std_median_error:.2f}" if not np.isnan(avg_median_error) else "NaN",
                f"{avg_unreg_pairs_pct:.1f}%",
                f"{avg_auc_100ms:.2f}±{std_auc_100ms:.2f}",
                f"{avg_auc_500ms:.2f}±{std_auc_500ms:.2f}",
                n_runs
            ])
        else:
            # Original single-run metrics
            absolute_errors = sport_metrics[sport_name].get("absolute_errors", [])
            num_unregistered_videos = len(sport_metrics[sport_name].get("unregistered_videos", []))
            
            # Calculate statistics for individual videos
            mean_abs_error = np.mean(absolute_errors) if absolute_errors else float('nan')
            median_abs_error = np.median(absolute_errors) if absolute_errors else float('nan')
            
            # Get pairwise errors
            errors_ms = sport_metrics[sport_name].get("pairwise_errors_ms", [])
            unregistered_count = len(sport_metrics[sport_name].get("unregistered_pairs", []))
            total_pairs = sport_metrics[sport_name].get("total_pairs", 0)
            
            # Calculate AUC at 100ms and 500ms with unregistered pairs treated as infinite errors
            auc_100ms = calculate_auc_with_unregistered(errors_ms, 100, unregistered_count, total_pairs)
            auc_500ms = calculate_auc_with_unregistered(errors_ms, 500, unregistered_count, total_pairs)
            
            # Add to metrics data
            if sport_name == "overall":
                sport_display = "OVERALL"
            else:
                sport_display = sport_name
                
            metrics_data.append([
                sport_display, 
                total_videos,
                f"{100.0 * num_unregistered_videos / total_videos:.1f}%" if total_videos > 0 else "N/A",
                f"{mean_abs_error:.2f}" if not np.isnan(mean_abs_error) else "NaN",
                f"{median_abs_error:.2f}" if not np.isnan(median_abs_error) else "NaN",
                f"{100.0 * unregistered_count / total_pairs:.1f}%" if total_pairs > 0 else "N/A",
                f"{auc_100ms:.2f}",
                f"{auc_500ms:.2f}",
                1
            ])
    
    # Create a formatted summary table
    if multi_run:
        headers = [
            "Sport", 
            "Total Videos", 
            "Unreg. Videos(%)", 
            "Mean Abs Error±std (ms)", 
            "Median Abs Error±std (ms)",
            "Unreg. Pairs(%)",
            "AUC@100ms±std",
            "AUC@500ms±std",
            "# Runs"
        ]
    else:
        headers = [
            "Sport", 
            "Total Videos", 
            "Unreg. Videos(%)", 
            "Mean Abs Error (ms)", 
            "Median Abs Error (ms)",
            "Unreg. Pairs(%)",
            "AUC@100ms",
            "AUC@500ms",
            "# Runs"
        ]

    # Format the table with tabulate
    table = tabulate(metrics_data, headers=headers, tablefmt="fancy_grid")
    return table

# Modify main() to pass the new parameters:
def main():
    """Main function for video synchronization evaluation."""
    parser = argparse.ArgumentParser(description="Video synchronization error evaluation")
    parser.add_argument("--result_roots", type=str, nargs="+", 
                        default=["/data11/shaowei3/datasets/egohumans/data_preprocessed_results_v2_improved", 
                                "/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected_results_v2_improved"], 
                        help="result roots")
    parser.add_argument("--dataset_fps", type=float, default=20.0,
                        help="frames per second of the dataset")
    parser.add_argument("--max_iter", type=int, default=10,
                        help="maximum iterations for robust optimization")
    parser.add_argument("--delta", type=float, default=1.0, 
                        help="delta for robust global optimization")
    parser.add_argument("--use_vggt", action='store_true', 
                        help="use vggt for camera pose")
    parser.add_argument("--vggt_choice", default="", type=str, 
                        choices=["", "samplev1", "samplev2", "full"], 
                        help="vgg sample choice")
    parser.add_argument("--use_hloc", action='store_true', 
                        help="use hloc camera pose")
    parser.add_argument("--use_gt_intrinsic", action='store_true', 
                        help="use gt intrinsic")
    parser.add_argument("--use_gt_sample", action='store_true', 
                        help="use gt sample")
    parser.add_argument("--use_mast3r", action='store_true', 
                        help="use mast3r (baseline)")
    parser.add_argument("--use_F_list", action='store_true', 
                        help="use F_list")
    parser.add_argument("--use_cosine", action='store_true', 
                        help="use cosine distance")
    parser.add_argument("--use_algebraic", action='store_true',
                        help="use algebraic distance")
    parser.add_argument("--err_thresh", type=float, default=50, 
                        help="error threshold")
    parser.add_argument("--est_choice", type=str, choices=["irls", "lsq"], 
                        default="irls", help="estimation choice")
    parser.add_argument("--fill_missing", action='store_true', 
                        help="fill missing pred offsets")
    parser.add_argument("--use_epipolar", action='store_true', 
                        help="use epipolar distance")
    parser.add_argument("--use_gt", action='store_true', 
                        help="use gt")
    parser.add_argument("--group_name", type=str, nargs="+", 
                        default=None, help="group name")
    parser.add_argument('--use_mvus', action='store_true', 
                        help="use mvus baseline")
    parser.add_argument('--use_random', action='store_true', 
                        help="use random baseline")
    parser.add_argument("--use_uni4d", action='store_true', 
                        help="use uni4d")
    # Add new arguments
    parser.add_argument("--pair_ratio", type=float, default=1.0,
                        help="ratio of pairs to use for global optimization (0.0 to 1.0)")
    parser.add_argument("--use_rst", action='store_true',
                        help="use minimum spanning tree for pair selection")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed for pair selection")
    parser.add_argument("--n_times", type=int, default=1,
                        help="number of times to run the experiment with different seeds")
    
    parser.add_argument('--fill_type', type=str, default="heuristic",
                      choices=["heuristic", "min"], 
                      help="method to fill missing pred offsets")
    parser.add_argument('--load_full', action='store_true', 
                      help="load full result files (ablation only on spurious pairs)")
    
    parser.add_argument("--save_result", action='store_true', help="save result")
    parser.add_argument("--result_root", type=str, default="../results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pair_ratio <= 0.0 or args.pair_ratio > 1.0:
        print(f"Warning: Invalid pair_ratio {args.pair_ratio}. Setting to default value 1.0")
        args.pair_ratio = 1.0
    
    if args.use_rst and args.pair_ratio < 1.0:
        print("Warning: Both use_rst and pair_ratio < 1.0 specified. RST will take precedence.")
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Determine pickle name based on args
    if args.use_vggt:
        if args.vggt_choice != "":
            pkl_name = f"result_candidates_v3_v2_vggt_{args.vggt_choice}.pkl"
        else:
            pkl_name = "result_candidates_v3_v2_vggt.pkl"
    elif args.use_hloc:
        pkl_name = "result_candidates_v3_v2_hloc.pkl"
    elif args.use_gt_sample:
        pkl_name = "result_candidates_v3_v2_gt_sample.pkl"
    elif args.use_F_list:
        pkl_name = "result_candidates_v3_v2_F_list.pkl"
    elif args.use_mast3r:
        pkl_name = "result_candidates_v3_mast3r.pkl"
    elif args.use_mvus:
        pkl_name = "result_candidates_mvus.pkl"
    elif args.use_random:
        pkl_name = "result_candidates_random.pkl"
    elif args.use_uni4d:
        pkl_name = "result_candidates_v3_v2_uni4d.pkl"
    elif args.use_gt:
        pkl_name = "result_gt.pkl"
    else:
        pkl_name = "result_candidates_v3_v2.pkl"
    
    # Add suffixes based on additional flags
    if args.use_epipolar:
        pkl_name = pkl_name.replace(".pkl", "_epipolar.pkl")
    if args.use_cosine:
        pkl_name = pkl_name.replace(".pkl", "_cosine.pkl")
    if args.use_algebraic:
        pkl_name = pkl_name.replace(".pkl", "_algebraic.pkl")
    if args.use_gt_intrinsic:
        pkl_name = pkl_name.replace(".pkl", "_gt_intrinsic.pkl")
    
    print(f"Using pickle file: {pkl_name}")
    
    # Define exclusions
    exclusions = ["pan171204_pose3_cam01_1152_1399", "volleyball_cam11"]
    
    # Print summary of settings
    print("\nVideo Synchronization Evaluation Settings:")
    print(f"- Optimization Method: {args.est_choice}")
    print(f"- Fill Missing Offsets: {args.fill_missing}")
    if args.use_rst:
        print("- Pair Selection: Minimum Spanning Tree")
    elif args.pair_ratio < 1.0:
        print(f"- Pair Selection: Random sampling with ratio {args.pair_ratio:.2f}")
    else:
        print("- Pair Selection: Using all available pairs")
    print(f"- Random Seed: {args.random_seed}")
    
   # Check if multi-run is needed
    need_multi_run = (args.use_rst or args.pair_ratio < 1.0) and args.n_times > 1
    if need_multi_run:
        print(f"Running with multiple iterations: {args.n_times} runs with different seeds")

    # Set random seed for reproducibility of first run
    random.seed(args.random_seed)

    # Determine pickle name based on args
    if args.use_vggt:
        if args.vggt_choice != "":
            pkl_name = f"result_candidates_v3_v2_vggt_{args.vggt_choice}.pkl"
        else:
            pkl_name = "result_candidates_v3_v2_vggt.pkl"
    elif args.use_hloc:
        pkl_name = "result_candidates_v3_v2_hloc.pkl"
    elif args.use_gt_sample:
        pkl_name = "result_candidates_v3_v2_gt_sample.pkl"
    elif args.use_F_list:
        pkl_name = "result_candidates_v3_v2_F_list.pkl"
    elif args.use_mast3r:
        pkl_name = "result_candidates_v3_mast3r.pkl"
    elif args.use_mvus:
        pkl_name = "result_candidates_mvus.pkl"
    elif args.use_random:
        pkl_name = "result_candidates_random.pkl"
    elif args.use_uni4d:
        pkl_name = "result_candidates_v3_v2_uni4d.pkl"
    elif args.use_gt:
        pkl_name = "result_gt.pkl"
    else:
        pkl_name = "result_candidates_v3_v2.pkl"
    
    # Add suffixes based on additional flags
    if args.use_epipolar:
        pkl_name = pkl_name.replace(".pkl", "_epipolar.pkl")
    if args.use_cosine:
        pkl_name = pkl_name.replace(".pkl", "_cosine.pkl")
    if args.use_algebraic:
        pkl_name = pkl_name.replace(".pkl", "_algebraic.pkl")
    if args.use_gt_intrinsic:
        pkl_name = pkl_name.replace(".pkl", "_gt_intrinsic.pkl")
    
    print(f"Using pickle file: {pkl_name}")
    
    # Define exclusions
    exclusions = ["pan171204_pose3_cam01_1152_1399", "volleyball_cam11"]
    
    # Print summary of settings
    print("\nVideo Synchronization Evaluation Settings:")
    print(f"- Optimization Method: {args.est_choice}")
    print(f"- Fill Missing Offsets: {args.fill_missing}")
    if args.use_rst:
        print("- Pair Selection: Minimum Spanning Tree")
    elif args.pair_ratio < 1.0:
        print(f"- Pair Selection: Random sampling with ratio {args.pair_ratio:.2f}")
    else:
        print("- Pair Selection: Using all available pairs")
    print(f"- Base Random Seed: {args.random_seed}")
    if need_multi_run:
        print(f"- Number of Runs: {args.n_times}")
        print(f"- Seeds: {[args.random_seed + i for i in range(args.n_times)]}")

    # Step 1: Load results
    sport_metrics = load_results(
        args.result_roots, 
        pkl_name, 
        group_name=args.group_name,
        exclusions=exclusions, 
        fill_type=args.fill_type,
        load_full=args.load_full
    )
    
    # Step 2: Compute global offsets with new parameters
    sport_metrics = compute_global_offsets(
        sport_metrics,
        est_choice=args.est_choice,
        max_iter=args.max_iter,
        delta=args.delta,
        fill_missing_flag=args.fill_missing,
        pair_ratio=args.pair_ratio,
        use_rst=args.use_rst,
        n_times=args.n_times,
        base_seed=args.random_seed
    )
    
    if args.save_result:
        dataset_roots = ["/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected", 
                         "/data11/shaowei3/datasets/egohumans/data_preprocessed",
                        "/data02/dyyao2/data_preprocessed_3dpop", 
                        "/data02/dyyao2/data_preprocessed_panoptic_corrected", 
                        "/data11/shaowei3/datasets/egohumans/data_preprocessed_blender_corrected"]
        for sport_name in sport_metrics:
            if sport_name == "overall":
                continue
            
            if len(args.result_roots) > 1 and sport_name.endswith("1"):
                result_root = args.result_roots[1]
            else:
                result_root = args.result_roots[0]
            dataset_root = None
            max_len = -1
            result_root_name = result_root.split("/")[-1]
            for root in dataset_roots:
                root_name = root.split("/")[-1]
                if root_name in result_root_name and len(root_name) > max_len:
                    dataset_root = root
                    max_len = len(root_name)
            
            if dataset_root is None:
                raise ValueError(f"Dataset root not found for result root: {result_root}")
            
            video_names = sport_metrics[sport_name]["video_names"]
            pred_offsets = sport_metrics[sport_name]["pred_offsets"]
            gt_video_offsets = sport_metrics[sport_name]["gt_video_offsets"]
            pred_video_offsets = {video_names[i]: pred_offsets[i] for i in range(len(video_names))}
            
            dataset_name = dataset_root.split("/")[-1]
            if args.use_mast3r:
                method_name = "mast3r"
            elif args.use_uni4d:
                method_name = "uni4d"
            else:
                method_name = "ours"
            save_dir = os.path.join(args.result_root, dataset_name, method_name)
            os.makedirs(save_dir, exist_ok=True)
            
            results = {
                "dataset_root": dataset_root,
                "video_names": video_names,
                "pred_video_offsets": pred_video_offsets,
                "gt_video_offsets": gt_video_offsets,
                "sport_name": sport_name,
                "dataset_fps": args.dataset_fps
            }
            
            save_path = os.path.join(save_dir, f"results_{sport_name}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            print("Saved results to", save_path)
            
    # Step 3: Evaluate results including pairwise errors and AUC metrics
    sport_metrics = evaluate_results(
        sport_metrics,
        dataset_fps=args.dataset_fps
    )
    
    # Step 4: Generate and print enhanced summary table with new metrics
    table = generate_enhanced_summary_table(sport_metrics, args.dataset_fps)
    print("\nVIDEO SYNCHRONIZATION ERROR SUMMARY (in milliseconds)")
    print(table)


if __name__ == "__main__":
    main()
