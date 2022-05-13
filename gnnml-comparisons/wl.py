from collections import defaultdict


def bijective_mapping(a, b):
    mapping = defaultdict(set)
    for i, j in zip(a, b):
        mapping[i].add(j)
    a_values = set(a)
    b_values = set(b)
    matching_cardinality = len(a_values) == len(b_values)
    mappings_are_one_to_one = all([len(l) == 1 for l in mapping.values()])
    return matching_cardinality and mappings_are_one_to_one


def weisfeiler_lehman(G, verbose=False, hash_function="reindex", include_self=False):
    n = len(G.vs)
    coloring = G.degree()
    if hash_function == "concatenate":
        coloring = [(c,) for c in coloring]
    new_coloring = []
    while not new_coloring or not bijective_mapping(coloring, new_coloring):
        coloring = new_coloring if new_coloring else coloring
        new_offset = max(coloring)
        new_coloring = []
        if verbose:
            print("Before loop", coloring, new_coloring)
        # For each node, compute its color fingerprint
        for node in range(n):
            neighs = G.neighbors(node)
            if include_self:
                neighs.insert(node, 0)
            colors = tuple(sorted([coloring[nb] for nb in neighs]))
            new_coloring.append(colors)
        if verbose:
            print("Looped", coloring, new_coloring)
        # Rehash the color if needed
        if hash_function == "reindex":
            # Compute the equivalence classes (equally valued colors) and hashes
            equivalence_classes = sorted(list(set(new_coloring)))
            equivalence_classes = {
                fingerprint: new_offset + i 
                for i, fingerprint in enumerate(equivalence_classes, 1)
            }
            new_coloring = [equivalence_classes[fingerprint] for fingerprint in new_coloring]
        elif hash_function == "concatenate":
            new_coloring = [tuple(list(old) + [new]) for old, new in zip(coloring, new_coloring)]
        if verbose:
            print("Hashed", coloring, new_coloring)
    # Pretty-print the concatenated encodings for readability
    if hash_function == "concatenate":
        new_coloring = [{i: c for i, c in enumerate(color)} for color in new_coloring]
    return new_coloring
