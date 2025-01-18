import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'col1': [1, 2, 4, 5, 7],
    'col2': [2, 3, 5, 6, 8]
})

# Union-Find class to handle grouping
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

# Function to group coordinates
def group_coordinates_from_df(df):
    uf = UnionFind()

    # Union the pairs from the DataFrame
    for x, y in zip(df['col1'], df['col2']):
        uf.union(x, y)

    # Group the connected components
    groups = {}
    for key in uf.parent:
        root = uf.find(key)
        groups.setdefault(root, set()).add(key)

    # Convert to sorted lists
    return [sorted(group) for group in groups.values()]

# Apply the function
result = group_coordinates_from_df(df)
print(result)