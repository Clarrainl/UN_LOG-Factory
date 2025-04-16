import os

def get_directory_tree(start_path='.', prefix=''):
    tree_lines = []
    files = os.listdir(start_path)
    files.sort()
    for index, file in enumerate(files):
        path = os.path.join(start_path, file)
        connector = '└── ' if index == len(files) - 1 else '├── '
        tree_lines.append(prefix + connector + file)
        if os.path.isdir(path):
            extension = '    ' if index == len(files) - 1 else '│   '
            subtree = get_directory_tree(path, prefix + extension)
            tree_lines.extend(subtree)
    return tree_lines

if __name__ == "__main__":
    output_file = "tree_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Project Directory Tree:\n\n")
        tree = get_directory_tree()
        for line in tree:
            f.write(line + "\n")
    print(f"Directory tree saved to '{output_file}'")
