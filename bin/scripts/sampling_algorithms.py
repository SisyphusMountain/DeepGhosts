from ete3 import Tree
import random
import math

def give_depth(tree):
    # First assign depth 0 to leaves.
    for node in tree.get_leaves():
        node.depth = 0
    # Then assign depth for internal nodes: depth = max(child.depth + child.dist)
    for node in tree.traverse(strategy="postorder"):
        if not node.is_leaf():
            node.depth = max(child.depth + child.dist for child in node.get_children())

def uniform_sample_leaves(tree, k, seed):
    random.seed(seed)
    leaves = [leaf.name for leaf in tree.iter_leaves()]
    sampled_leaves = random.sample(leaves, k)
    return sampled_leaves

def diversified_sample_leaves(tree, k, seed):
    random.seed(seed)
    give_depth(tree)
    nodes_with_depths = [(node, node.depth) for node in tree.traverse() if not node.is_leaf()]
    nodes_with_depths.sort(key=lambda x: x[1], reverse=False)
    internal_nodes_to_keep = {node[0] for node in nodes_with_depths[-k+1:]} # Start by taking the root

    chosen_dict = dict()
    sampled_leaves = []
    for node in internal_nodes_to_keep:
        children = node.get_children()
        flags = tuple(child in internal_nodes_to_keep for child in children)
        chosen_dict[node] = flags

    for node, flags in chosen_dict.items():
        children = node.get_children()
        for flag, child in zip(flags, children):
            if not flag:
                sampled_leaves.append(random.choice(child.get_leaves()))
    return [leaf.name for leaf in sampled_leaves]

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
    return min_cost, [leaf.name for leaf in leaves_list]

def sampling(tree, k, sampling_mode, seed, take_root=True):
    if sampling_mode == "uniform":
        return uniform_sample_leaves(tree, k, seed)
    elif sampling_mode == "diversified":
        return diversified_sample_leaves(tree, k, seed)
    elif sampling_mode == "min_pairwise_sum":
        _, leaves_list = min_pairwise_sum(tree, k, take_root)
        return leaves_list
    else:
        raise ValueError("Invalid sampling mode")


# Version of cluster sampling like Sebastian Höhna's 
# def compute_ltk_added(shallowest_node, internal_nodes_to_keep, root_left_children, root_right_children, 
#                         root_has_left_children, root_has_right_children):
#     """Helper function for the clustered sampling algorithm."""
#     ltk_added = 2  # at the maximum, adding an internal node will require adding two leaves
    
#     # Check if children are already in the set of internal nodes to keep
#     has_left_child = shallowest_node.get_children()[0] in internal_nodes_to_keep
#     has_right_child = shallowest_node.get_children()[1] in internal_nodes_to_keep
    
#     if has_left_child:
#         ltk_added -= 1
#     if has_right_child:
#         ltk_added -= 1
        
#     # Check if this node contributes to root's left or right children
#     if not root_has_left_children and shallowest_node in root_left_children:
#         ltk_added -= 1
#         root_has_left_children = True
#     if not root_has_right_children and shallowest_node in root_right_children:
#         ltk_added -= 1
#         root_has_right_children = True
        
#     return ltk_added, has_left_child, has_right_child, root_has_left_children, root_has_right_children

# def cluster_sample_leaves(tree, nb_leaves_to_keep, seed):
#     random.seed(seed)
#     give_depth(tree)
#     nodes_with_depths = [(node, node.depth) for node in tree.traverse() if not node.is_leaf()]
#     nodes_with_depths.sort(key=lambda x: x[1], reverse=True)
#     internal_nodes_to_keep = {nodes_with_depths.pop(0)[0]: (False, False)} # Always keep the root
#     nb_sampled_leaves = 2 # If we keep the root, we need two leaves.
#     # Now choose the shallowest internal nodes in the tree, while keeping count of the current number of sampled leaves
#     # Once we reach the target number of sampled leaves -1, we have two possibilities:
#     # Either adding the shallowest nonsampling internal node would add a single leaf to the sample, in which case we add it
#     # Or it would add more than one leaf to the sample, in which case we don't add it, and instead find the internal node in the tree
#     # whose addition would add a branch with the smallest length to the constructed tree.
#     root_left_children = {node for node in (tree.get_children()[0]).iter_descendants()}
#     root_right_children = {node for node in (tree.get_children()[1]).iter_descendants()}

#     while nb_sampled_leaves < nb_leaves_to_keep:
#         ltk_added, has_left_child, has_right_child, root_has_left_children, root_has_right_children = compute_ltk_added(
#             nodes_with_depths[-1][0], internal_nodes_to_keep, root_left_children, root_right_children, internal_nodes_to_keep[tree][0], internal_nodes_to_keep[tree][1])
#         if ltk_added + nb_sampled_leaves <= nb_leaves_to_keep:
#             internal_nodes_to_keep[nodes_with_depths.pop()[0]] = (has_left_child, has_right_child)
#             nb_sampled_leaves += ltk_added
#             internal_nodes_to_keep[tree] = (root_has_left_children, root_has_right_children)
#         else:
#             # remove the shallowest node
#             nodes_with_depths.pop()
#     sampled_leaves = set()
#     # Now sample one leaf for each side of internal node that doesn't already have a leaf
#     for internal_node, (has_left_child, has_right_child) in internal_nodes_to_keep.items():
#         if not has_left_child:
#             sampled_leaves.add(random.choice((internal_node.get_children()[0]).get_leaves()))
#         if not has_right_child:
#             sampled_leaves.add(random.choice((internal_node.get_children()[1]).get_leaves()))
#     return sampled_leaves
# tree_string = "((((1:1.0, 2:1.0)9:1.0, 3:2.0)10:1.0,4:3.0)11:1.0,((7:1.2,8:1.2)12:1.8,(5:1.5,6:1.5)13:1.5)14:1.0)15:1.0;"
# t = Tree(tree_string, format=1)
# result = cluster_sample_leaves(t, 4, 5336)
# result_2 = diversified_sample_leaves(t, 4, 5336)
# print([node.name for node in result])
# print([node.name for node in result_2])