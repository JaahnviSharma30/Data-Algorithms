# Graph Algorithms and Advanced Data Structures Guide

## Graph Algorithms

### Representation of Graphs
- **Adjacency Matrix**: 2D array where matrix[i][j] = 1 indicates an edge from vertex i to j
- **Adjacency List**: Array of lists where each list contains the neighbors of a vertex
- **Edge List**: List of all edges in the graph as pairs (u, v)
- **Incidence Matrix**: Matrix where rows represent vertices and columns represent edges

### Breadth-First Search (BFS)
```
BFS(graph, startVertex):
    Create a queue Q
    Mark startVertex as visited and enqueue it
    
    while Q is not empty:
        v = Q.dequeue()
        Process v
        
        for each unvisited neighbor w of v:
            Mark w as visited
            Enqueue w to Q
```

**Time Complexity**: O(V + E) where V is the number of vertices and E is the number of edges
**Space Complexity**: O(V)

### Depth-First Search (DFS)
```
DFS(graph, startVertex):
    Mark startVertex as visited
    Process startVertex
    
    for each unvisited neighbor w of startVertex:
        DFS(graph, w)
```

**Time Complexity**: O(V + E)
**Space Complexity**: O(V) for the recursion stack

### Topological Sort
```
TopologicalSort(graph):
    Create a stack S
    Mark all vertices as unvisited
    
    for each unvisited vertex v:
        TopologicalSortUtil(v, S)
    
    while S is not empty:
        Print S.pop()

TopologicalSortUtil(v, S):
    Mark v as visited
    
    for each unvisited neighbor w of v:
        TopologicalSortUtil(w, S)
    
    Push v to stack S
```

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)

### Difference between BFS and DFS

| BFS | DFS |
|-----|-----|
| Uses queue data structure | Uses stack or recursion |
| Explores vertices level by level | Explores as far as possible along each branch |
| Finds shortest path in unweighted graphs | May not find the shortest path |
| Better for finding nodes closer to the start | Better for exploring all possible paths |
| More memory-intensive for wide graphs | More memory-efficient for deep graphs |
| Optimal for searching vertices at a given distance | Good for solving mazes or puzzles |

## Disjoint Sets (Union-Find)

### Basic Operations
- **MakeSet(x)**: Creates a new set with a single element x
- **Find(x)**: Returns the representative (root) of the set containing x
- **Union(x, y)**: Merges the sets containing x and y

### Implementation with Path Compression and Union by Rank
```
// Initialize
function MakeSet(x):
    parent[x] = x
    rank[x] = 0

// Find with path compression
function Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])  // Path compression
    return parent[x]

// Union by rank
function Union(x, y):
    rootX = Find(x)
    rootY = Find(y)
    
    if rootX == rootY:
        return
    
    if rank[rootX] < rank[rootY]:
        parent[rootX] = rootY
    else if rank[rootX] > rank[rootY]:
        parent[rootY] = rootX
    else:
        parent[rootY] = rootX
        rank[rootX]++
```

**Time Complexity**: Nearly O(1) amortized time per operation with path compression and union by rank
**Space Complexity**: O(n)

### Finding Cycle in a Graph
```
// For undirected graphs using Union-Find
function HasCycle(graph):
    Initialize disjoint set for each vertex
    
    for each edge (u,v) in graph:
        u_root = Find(u)
        v_root = Find(v)
        
        if u_root == v_root:
            return true  // Cycle found
        
        Union(u_root, v_root)
    
    return false  // No cycle
```

**Time Complexity**: O(E α(V)) where α is the inverse Ackermann function (nearly constant)

### Finding Strongly Connected Components (Kosaraju's Algorithm)
```
function FindSCCs(graph):
    // Step 1: DFS and store vertices in order of finishing time
    stack = empty stack
    visited = set()
    
    for each vertex v in graph:
        if v not in visited:
            FillOrder(v, visited, stack)
    
    // Step 2: Transpose the graph
    transposeGraph = reverse all edges in graph
    
    // Step 3: DFS on transpose graph using stack order
    visited.clear()
    
    while stack is not empty:
        v = stack.pop()
        if v not in visited:
            DFS(transposeGraph, v, visited)
            print "SCC completed"

function FillOrder(v, visited, stack):
    visited.add(v)
    
    for each neighbor w of v:
        if w not in visited:
            FillOrder(w, visited, stack)
    
    stack.push(v)
```

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)

## Minimum Spanning Trees

### Kruskal's Algorithm
```
function Kruskal(graph):
    result = []  // Will store the MST
    
    Sort all edges in non-decreasing order of weight
    Initialize disjoint sets for all vertices
    
    for each edge (u,v) in sorted edges:
        u_root = Find(u)
        v_root = Find(v)
        
        if u_root != v_root:  // Including this edge doesn't form a cycle
            result.add(edge)
            Union(u_root, v_root)
    
    return result
```

**Time Complexity**: O(E log E) or O(E log V)
**Space Complexity**: O(V + E)

### Prim's Algorithm
```
function Prim(graph, startVertex):
    result = []
    priority_queue = min heap of vertices based on key values
    
    // Initialize all vertices with infinite key value except startVertex
    for each vertex v in graph:
        key[v] = INFINITE
        inMST[v] = false
    
    key[startVertex] = 0
    Insert all vertices into priority_queue
    
    while priority_queue is not empty:
        u = Extract-Min from priority_queue
        inMST[u] = true
        
        for each adjacent vertex v of u:
            if inMST[v] == false and weight(u,v) < key[v]:
                key[v] = weight(u,v)
                parent[v] = u
                Decrease-Key(priority_queue, v, key[v])
    
    // Construct MST from parent array
    for v from 1 to V-1:
        result.add(edge from parent[v] to v)
    
    return result
```

**Time Complexity**: O(E log V) with binary heap
**Space Complexity**: O(V)

## Single Source Shortest Paths

### Dijkstra's Algorithm
```
function Dijkstra(graph, startVertex):
    // Initialize distances
    for each vertex v in graph:
        distance[v] = INFINITE
        visited[v] = false
    
    distance[startVertex] = 0
    priority_queue = min heap based on distance values
    Insert all vertices into priority_queue
    
    while priority_queue is not empty:
        u = Extract-Min from priority_queue
        visited[u] = true
        
        for each adjacent vertex v of u:
            if visited[v] == false and distance[u] + weight(u,v) < distance[v]:
                distance[v] = distance[u] + weight(u,v)
                parent[v] = u
                Decrease-Key(priority_queue, v, distance[v])
    
    return distance
```

**Time Complexity**: O(E log V) with binary heap
**Space Complexity**: O(V)
**Limitation**: Works only for graphs with non-negative weights

### Bellman-Ford Algorithm
```
function BellmanFord(graph, startVertex):
    // Initialize distances
    for each vertex v in graph:
        distance[v] = INFINITE
    
    distance[startVertex] = 0
    
    // Relax all edges V-1 times
    for i from 1 to |V|-1:
        for each edge (u,v) with weight w in graph:
            if distance[u] != INFINITE and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    
    // Check for negative weight cycles
    for each edge (u,v) with weight w in graph:
        if distance[u] != INFINITE and distance[u] + w < distance[v]:
            return "Graph contains negative weight cycle"
    
    return distance
```

**Time Complexity**: O(V·E)
**Space Complexity**: O(V)
**Advantage**: Works with negative edge weights (as long as there's no negative cycle)

## All Pair Shortest Paths

### Floyd-Warshall Algorithm
```
function FloydWarshall(graph):
    // Initialize distance matrix
    dist = copy of adjacency matrix
    
    for vertex k from 0 to |V|-1:
        for vertex i from 0 to |V|-1:
            for vertex j from 0 to |V|-1:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist
```

**Time Complexity**: O(V³)
**Space Complexity**: O(V²)
**Advantage**: Simple implementation and works with negative edge weights (as long as there's no negative cycle)

## Greedy Algorithms

### Activity Selection Problem
```
function ActivitySelection(start[], finish[]):
    // Sort activities by finish time
    Sort activities based on finish time
    
    result = [first activity]
    last_selected = 0
    
    for i from 1 to n-1:
        if start[i] >= finish[last_selected]:
            Add activity i to result
            last_selected = i
    
    return result
```

**Time Complexity**: O(n log n) for sorting
**Space Complexity**: O(n) for the result

### Job Sequencing Problem
```
function JobSequencing(jobs[], deadlines[], profits[]):
    // Sort jobs by profit in non-increasing order
    Sort jobs based on profit
    
    maxDeadline = maximum deadline among all jobs
    slot = array of size maxDeadline initialized with false
    result = array to store job IDs
    
    for i from 0 to n-1:
        // Find free slot before deadline
        for j from min(deadlines[i]-1, maxDeadline-1) down to 0:
            if slot[j] == false:
                result[j] = i  // Schedule job i at slot j
                slot[j] = true
                break
    
    return result
```

**Time Complexity**: O(n²) where n is the number of jobs
**Space Complexity**: O(maxDeadline)

### Huffman Coding
```
function HuffmanCoding(characters[], frequencies[]):
    Create a leaf node for each character and add to min heap
    
    while heap.size() > 1:
        left = heap.extractMin()
        right = heap.extractMin()
        
        top = new Node('$', left.freq + right.freq)
        top.left = left
        top.right = right
        
        heap.insert(top)
    
    // Print codes by traversing the Huffman tree
    PrintCodes(heap.extractMin(), "")
```

**Time Complexity**: O(n log n) where n is the number of unique characters
**Space Complexity**: O(n)

### Fractional Knapsack Problem
```
function FractionalKnapsack(weights[], values[], capacity):
    // Create array of value/weight ratios
    for i from 0 to n-1:
        items[i] = (values[i]/weights[i], weights[i], values[i])
    
    // Sort items by value/weight ratio in non-increasing order
    Sort items
    
    totalValue = 0
    
    for i from 0 to n-1:
        if capacity >= items[i].weight:
            // Take the whole item
            capacity -= items[i].weight
            totalValue += items[i].value
        else:
            // Take a fraction of the item
            totalValue += items[i].ratio * capacity
            break
    
    return totalValue
```

**Time Complexity**: O(n log n) for sorting
**Space Complexity**: O(n)

## Dynamic Programming

### Overlapping Substructure Property
A problem has overlapping substructure if the solution involves solving the same subproblem multiple times, making it a good candidate for dynamic programming.

### Optimal Substructure Property
A problem has optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.

### Tabulation vs Memoization

| Tabulation (Bottom-Up) | Memoization (Top-Down) |
|------------------------|------------------------|
| Iterative approach | Recursive approach |
| Starts from base cases | Starts from the original problem |
| Usually uses arrays/tables | Uses hash map or array for caching |
| Handles all subproblems | Only handles necessary subproblems |
| Better space complexity | More intuitive for some problems |
| No recursion overhead | Has recursion overhead |

### Fibonacci Numbers

**Memoization (Top-Down)**:
```
memo = {}

function fibonacci(n):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n-1) + fibonacci(n-2)
    return memo[n]
```

**Tabulation (Bottom-Up)**:
```
function fibonacci(n):
    if n <= 1:
        return n
        
    dp = new array of size n+1
    dp[0] = 0
    dp[1] = 1
    
    for i from 2 to n:
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

**Space-Optimized**:
```
function fibonacci(n):
    if n <= 1:
        return n
        
    a = 0
    b = 1
    
    for i from 2 to n:
        c = a + b
        a = b
        b = c
    
    return b
```

**Time Complexity**: O(n)
**Space Complexity**: O(n) for memoization/tabulation, O(1) for space-optimized

### 0/1 Knapsack Problem
```
function knapsack(weights[], values[], capacity, n):
    // Create a 2D DP table
    dp = new 2D array of size (n+1) x (capacity+1)
    
    for i from 0 to n:
        for w from 0 to capacity:
            if i == 0 or w == 0:
                dp[i][w] = 0
            else if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

**Time Complexity**: O(n·W) where n is the number of items and W is the capacity
**Space Complexity**: O(n·W)

### Longest Common Subsequence
```
function LCS(X, Y, m, n):
    // Create a 2D DP table
    dp = new 2D array of size (m+1) x (n+1)
    
    for i from 0 to m:
        for j from 0 to n:
            if i == 0 or j == 0:
                dp[i][j] = 0
            else if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**Time Complexity**: O(m·n) where m and n are the lengths of the sequences
**Space Complexity**: O(m·n)

### Matrix Chain Multiplication
```
function MatrixChainOrder(p[], n):
    // p[] is array of dimensions where matrix i has dimensions p[i-1] x p[i]
    // n is the number of matrices + 1
    
    // Create a 2D DP table
    dp = new 2D array of size n x n
    
    // Initialize diagonal elements as 0
    for i from 1 to n-1:
        dp[i][i] = 0
    
    // Fill the DP table
    for L from 2 to n-1:  // L is chain length
        for i from 1 to n-L:
            j = i + L - 1
            dp[i][j] = INFINITE
            
            for k from i to j-1:
                cost = dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j]
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
    
    return dp[1][n-1]
```

**Time Complexity**: O(n³)
**Space Complexity**: O(n²)

## Hashing and String Matching

### Hashing Data Structure

**Hash Function**: A function that converts keys into array indices
  - Good hash functions:
    - Distribute keys uniformly
    - Minimize collisions
    - Are easy to compute
  - Common methods: Division, Multiplication, Universal Hashing

**Collision Handling**:
1. **Chaining**: Keep a linked list of all elements that hash to the same slot
   ```
   Insert(key, value):
       index = hash(key)
       Add (key, value) to the list at table[index]
   
   Search(key):
       index = hash(key)
       Search for key in the list at table[index]
       Return associated value if found
   
   Delete(key):
       index = hash(key)
       Remove the node with key from the list at table[index]
   ```

2. **Open Addressing**: All elements are stored in the hash table itself
   - Linear Probing: probe linearly until an empty slot is found
   - Quadratic Probing: probe with quadratic function
   - Double Hashing: use a second hash function for probe distance
   ```
   Insert(key, value):
       index = hash(key)
       while table[index] is occupied:
           index = (index + probe(key, i)) % tableSize
           i++
       table[index] = (key, value)
   
   Search(key):
       index = hash(key)
       while table[index] is not empty:
           if table[index].key == key:
               return table[index].value
           index = (index + probe(key, i)) % tableSize
           i++
       return null  // Not found
   
   Delete(key):
       // Special care needed - typically mark as "deleted" rather than empty
   ```

### String Matching Algorithms

#### Naive String Matching
```
function NaiveSearch(text, pattern):
    n = text.length
    m = pattern.length
    
    for i from 0 to n-m:
        j = 0
        while j < m and text[i+j] == pattern[j]:
            j++
        
        if j == m:
            print "Pattern found at index", i
```

**Time Complexity**: O((n-m+1)·m) in worst case
**Space Complexity**: O(1)

#### Rabin-Karp Algorithm
```
function RabinKarp(text, pattern, q):
    n = text.length
    m = pattern.length
    d = number of characters in the alphabet
    h = d^(m-1) % q
    
    p = 0  // hash value for pattern
    t = 0  // hash value for text
    
    // Calculate initial hash values
    for i from 0 to m-1:
        p = (d * p + pattern[i]) % q
        t = (d * t + text[i]) % q
    
    for i from 0 to n-m:
        if p == t:  // Check character by character
            if text[i...i+m-1] == pattern:
                print "Pattern found at index", i
        
        if i < n-m:
            t = (d * (t - text[i] * h) + text[i+m]) % q
            if t < 0:
                t += q
```

**Time Complexity**: O(n+m) average, O((n-m+1)·m) worst case
**Space Complexity**: O(1)

#### Knuth-Morris-Pratt (KMP) Algorithm
```
function KMP(text, pattern):
    n = text.length
    m = pattern.length
    
    // Preprocess: Compute LPS (Longest Prefix Suffix) array
    lps = ComputeLPSArray(pattern)
    
    i = 0  // index for text
    j = 0  // index for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i++
            j++
        
        if j == m:
            print "Pattern found at index", i-j
            j = lps[j-1]
        else if i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i++

function ComputeLPSArray(pattern):
    m = pattern.length
    lps = new array of size m
    
    length = 0
    lps[0] = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length++
            lps[i] = length
            i++
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i++
    
    return lps
```

**Time Complexity**: O(n+m)
**Space Complexity**: O(m)

## NP-Completeness

NP-complete problems are a class of decision problems for which:
1. Solutions can be verified in polynomial time (NP)
2. Every problem in NP can be reduced to it in polynomial time (NP-hard)

Examples of NP-complete problems:
- Boolean Satisfiability Problem (SAT)
- Traveling Salesman Problem (decision version)
- Vertex Cover
- Subset Sum
- Graph Coloring
- Hamiltonian Path/Cycle
- Clique Problem

**Approaches to NP-complete problems**:
1. Exact algorithms that may take exponential time
2. Approximation algorithms
3. Heuristic-based approaches
4. Parameterized algorithms
5. Randomized algorithms
