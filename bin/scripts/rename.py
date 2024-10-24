import sys
from ete3 import Tree

def rename_internal_nodes(newick_path, output_path):
    # Load the Newick tree
    tree = Tree(newick_path, format=1)

    count = len(tree) # start at the number of leaves to make sure there is no redundancy between internal names and leaf names
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = str(count)
            count += 1
    
    # Save the tree with renamed internal nodes
    tree.write(outfile=output_path, format=1, format_root_node=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename.py <input_newick_file> <output_newick_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    rename_internal_nodes(input_path, output_path)