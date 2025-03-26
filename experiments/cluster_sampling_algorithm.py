from ete3 import Tree
tree_string = "((((1:1.0, 2:1.0)9:1.0, 3:2.0)10:1.0,4:3.0)11:1.0,((7:1.2,8:1.2)12:1.8,(5:1.5,6:1.5)13:1.5)14:1.0)15:1.0;"
t = Tree(tree_string, format=1)


def give_depth(tree):
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf():
            node.add_features(depth=0)
        else:
            node.add_features(depth=max(child.depth + child.dist for child in node.children))

def compute_ltk_added(shallowest_node, internal_nodes_to_keep, root_left_children, root_right_children, 
                        root_has_left_children, root_has_right_children):
    ltk_added = 2  # at the maximum, adding an internal node will require adding two leaves
    
    # Check if children are already in the set of internal nodes to keep
    has_left_child = shallowest_node.get_children()[0] in internal_nodes_to_keep
    has_right_child = shallowest_node.get_children()[1] in internal_nodes_to_keep
    
    if has_left_child:
        ltk_added -= 1
    if has_right_child:
        ltk_added -= 1
        
    # Check if this node contributes to root's left or right children
    if not root_has_left_children and shallowest_node in root_left_children:
        ltk_added -= 1
        root_has_left_children = True
    if not root_has_right_children and shallowest_node in root_right_children:
        ltk_added -= 1
        root_has_right_children = True
        
    return ltk_added, has_left_child, has_right_child, root_has_left_children, root_has_right_children

def cluster_sample_leaves(tree, nb_leaves_to_keep):
    give_depth(tree)
    nodes_with_depths = [(node, node.depth) for node in tree.traverse() if not node.is_leaf()]
    nodes_with_depths.sort(key=lambda x: x[1], reverse=True)
    internal_nodes_to_keep = {nodes_with_depths.pop(0)[0]: (False, False)} # Always keep the root
    nb_sampled_leaves = 2 # If we keep the root, we need two leaves.
    # Now choose the shallowest internal nodes in the tree, while keeping count of the current number of sampled leaves
    # Once we reach the target number of sampled leaves -1, we have two possibilities:
    # Either adding the shallowest nonsampling internal node would add a single leaf to the sample, in which case we add it
    # Or it would add more than one leaf to the sample, in which case we don't add it, and instead find the internal node in the tree
    # whose addition would add a branch with the smallest length to the constructed tree.
    root_left_children = {node for node in (tree.get_children()[0]).iter_descendants()}
    root_right_children = {node for node in (tree.get_children()[1]).iter_descendants()}

    while nb_sampled_leaves < nb_leaves_to_keep:
        ltk_added, has_left_child, has_right_child, root_has_left_children, root_has_right_children = compute_ltk_added(
            nodes_with_depths[-1][0], internal_nodes_to_keep, root_left_children, root_right_children, internal_nodes_to_keep[tree][0], internal_nodes_to_keep[tree][1])
        if ltk_added + nb_sampled_leaves <= nb_leaves_to_keep:
            internal_nodes_to_keep[nodes_with_depths.pop()[0]] = (has_left_child, has_right_child)
            nb_sampled_leaves += ltk_added
            internal_nodes_to_keep[tree] = (root_has_left_children, root_has_right_children)
        else:
            # remove the shallowest node
            nodes_with_depths.pop()
    sampled_leaves = set()
    # Now sample one leaf for each side of internal node that doesn't already have a leaf
    for internal_node, (has_left_child, has_right_child) in internal_nodes_to_keep.items():
        if not has_left_child:
            sampled_leaves.add(internal_node.get_children()[0])
        if not has_right_child:
            sampled_leaves.add(internal_node.get_children()[1])
    return sampled_leaves
result = cluster_sample_leaves(t, 5)
print([node.name for node in result])