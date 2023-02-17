# Convert a 1D index b into a 2D grid position (i, j), such that
# 0 <= i < n and max(k + 1, i) <= j < n. This is useful when writing
# parallel code, because it is easier to parallelize over b than over
# rows and columns of varying sizes.
#
# For example, k = 2 and n = 5:
#
# >>> print_trapezoid(2, 5)
# ..###
# ..###
# ..###
# ...##
# ....#

def trapezoid(k, n):
    for b in range(k * (n - k) + (n - k) * (n - k + 1) // 2):
        # Map 1D indices to 2D grid positions.
        i = b // (n - k)
        j = n - 1 - b % (n - k)
        if i > j:
            i = n - (i - (k + 1)) - 1
            j = n - (j - k) - 1
        yield (i, j)


# Used for testing: python -i trapezoid_indexing.py
def print_trapezoid(k, n):
    grid = [[False for j in range(n)] for i in range(n)]
    for i, j in trapezoid(k, n):
        assert 0 <= i < n and 0 <= j < n and not grid[i][j]
        grid[i][j] = True
    for row in grid:
        print("".join(".#"[b] for b in row))
