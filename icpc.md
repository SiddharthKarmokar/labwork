```c++ - 
vector<int> dijkstra(int V, vector<vector<pair<int, int>>> &adj, int src) {
    vector<int> dist(V, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    dist[src] = 0;
    pq.push({0, src}); // {distance, node}

    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > dist[u]) continue;

        for (auto &edge : adj[u]) {
            int v = edge.first;
            int w = edge.second;

            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}


def dijkstra(V, adj, src):
    dist = [float('inf')] * V
    dist[src] = 0
    pq = [(0, src)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist
```

```python
vector<int> kahn(int V, vector<vector<int>>& adj) {
    vector<int> indegree(V, 0);
    for (int u = 0; u < V; u++) {
        for (int v : adj[u])
            indegree[v]++;
    }

    queue<int> q;
    for (int i = 0; i < V; i++)
        if (indegree[i] == 0)
            q.push(i);

    vector<int> topo;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        topo.push_back(u);

        for (int v : adj[u]) {
            indegree[v]--;
            if (indegree[v] == 0)
                q.push(v);
        }
    }

    if (topo.size() != V)
        cout << "Cycle detected! No valid topological order.\n";
    return topo;
}

def kahn(V, adj):
    indegree = [0] * V
    for u in range(V):
        for v in adj[u]:
            indegree[v] += 1

    q = deque([i for i in range(V) if indegree[i] == 0])
    topo = []

    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)def kahn(V, adj):
    indegree = [0] * V
    for u in range(V):
        for v in adj[u]:
            indegree[v] += 1

    q = deque([i for i in range(V) if indegree[i] == 0])
    topo = []

    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

```
```c++
void dfs(int node, vector<vector<int>> &adj, vector<int> &visited) {
    visited[node] = 1;
    cout << node << " ";

    for (int neighbor : adj[node]) {
        if (!visited[neighbor])
            dfs(neighbor, adj, visited);
    }
}

def dfs(node, adj, visited):
    visited[node] = True
    print(node, end=" ")

    for neighbor in adj[node]:
        if not visited[neighbor]:
            dfs(neighbor, adj, visited)
```