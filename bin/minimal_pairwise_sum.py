import dendropy
from ete3 import Tree
import math
import itertools

def give_depth(tree):
    # First assign depth 0 to leaves.
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf():
            node.depth = 0
    # Then assign depth for internal nodes: depth = max(child.depth + child.dist)
    for node in tree.traverse(strategy="postorder"):
        if not node.is_leaf():
            node.depth = max(child.depth + child.dist for child in node.get_children())


def min_pairwise_sum(tree, k, take_root=True):
    """Outputs the set of k leaves such that the resulting tree has minimal sum
    of pairwise distances. The tree is assumed to be ultrametric.
    If take_root is True, we constrain the root to be in the selected set."""


    give_depth(tree)
    
    # dp will be a dictionary {node: {nb_spld: (min_cost, leaves_list)}}
    dp = {}
    
    def compute_dp(node):
        # Initialize dp table for node with keys 0..k.
        dp_node = {n_spld: (math.inf, []) for n_spld in range(k+1)}
        
        # Base case: if node is a leaf.
        if node.is_leaf():
            dp_node[0] = (0, [])
            dp_node[1] = (0, [node])
            dp[node] = dp_node
            return dp_node
        
        children = node.get_children()
        left, right = children[0], children[1]
        dp_left = compute_dp(left)
        dp_right = compute_dp(right)
        

        node_depth = left.dist + left.depth
        
        if node.is_root() and take_root:
            valid_j = lambda i, j: (i >= 2 and j >= 1 and (i - j) >= 1)
        else:
            valid_j = lambda i, j: True
        
        # Merge the two DP tables.
        for i in range(k+1):
            best_cost = math.inf
            best_leaves = []
            # Consider all splits j + (i-j) = i.
            for j in range(i+1):
                if not valid_j(i, j):
                    continue
                cost_left, leaves_left = dp_left.get(j, (math.inf, []))
                cost_right, leaves_right = dp_right.get(i - j, (math.inf, []))
                if cost_left == math.inf or cost_right == math.inf:
                    continue

                left_to_right_paths = j * (i - j) * (2 * node_depth)
                total_cost = cost_left + cost_right + left_to_right_paths
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_leaves = leaves_left + leaves_right
            dp_node[i] = (best_cost, best_leaves)
        dp[node] = dp_node
        return dp_node
    

    dp_table = compute_dp(tree)
    min_cost, leaves_list = dp_table.get(k, (math.inf, []))
    return min_cost, leaves_list

def enumerate_subsets_dp_sanity(tree, k):
    """
    For the sanity check: enumerate over all subsets of k leaves (from the tree's inherent root)
    such that at least one leaf is taken from each of the two children of the root,
    and return the minimum sum of pairwise distances and the subset achieving that.
    """
    # The tree's inherent root.
    root = tree
    children = root.get_children()
    if len(children) != 2:
        raise ValueError("Sanity check expects a binary tree with exactly two children at the root.")
    left_leaves = children[0].get_leaves()
    right_leaves = children[1].get_leaves()
    
    best_cost = math.inf
    best_subset = None
    # For each possible split: at least one from left, at least one from right.
    for i in range(1, k):
        if i > len(left_leaves) or (k - i) > len(right_leaves):
            continue  # not enough leaves on one side.
        for left_subset in itertools.combinations(left_leaves, i):
            for right_subset in itertools.combinations(right_leaves, k - i):
                subset = list(left_subset) + list(right_subset)
                # Compute the sum of pairwise distances.
                cost = 0
                for u, v in itertools.combinations(subset, 2):
                    cost += tree.get_distance(u, v)
                if cost < best_cost:
                    best_cost = cost
                    best_subset = subset
    return best_cost, best_subset

n_trials = 100
total_leaves = 12
sampled_leaved = 6
for i in range(n_trials):
    tree = dendropy.model.birthdeath.birth_death_tree(birth_rate=1, death_rate=0, num_extant_tips=total_leaves)
    tree.write(path="tree.nwk", schema="newick", suppress_rooting=True)
    # Example Newick string of an ultrametric tree (binary tree).
    # In an ultrametric tree, all leaves are equidistant from the root.
    tree = Tree("/home/enzo/Documents/git/WP1/DeepGhosts/bin/tree.nwk", format=1)
    k = sampled_leaved  # number of leaves to select
    
    # Compute the DP solution using our function.
    dp_cost, dp_leaves = min_pairwise_sum(tree, k, take_root=True)
    # Sanity check: enumerate over all valid subsets.
    sanity_cost, sanity_subset = enumerate_subsets_dp_sanity(tree, k)
    print(f"DP solution: {dp_cost}, Sanity check: {sanity_cost}")
    
    # Compare the two.
    if abs(dp_cost - sanity_cost) > 1e-9:
        raise ValueError("Sanity check FAILED!")
    print("--------------------------------------------------")