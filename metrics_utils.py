import networkx as nx

class ClusterAnalysis:
   
        

    def determine_cluster_memberships(G, u, threshold=.1):
        cluster_memberships = {node: [] for node in G.nodes()}  # Initialize for all nodes in G
        num_clusters = u.shape[0]

        for i in range(num_clusters):
            for j, node in enumerate(G.nodes()):
                if u[i][j] > threshold:
                    cluster_memberships[node].append(i)


        print(cluster_memberships)
        return cluster_memberships

    def calculate_structural_modularity(G, cluster_memberships, num_clusters):
        entities = {i: 0 for i in range(num_clusters)}
        internal_edges = {i: 0 for i in range(num_clusters)}
        external_edges = {i: {j: 0 for j in range(num_clusters)} for i in range(num_clusters)}

        for node in G.nodes():
            node_clusters = cluster_memberships.get(node, [])
            for cluster in node_clusters:
                entities[cluster] += 1
                for neighbor in G[node]:
                    neighbor_clusters = cluster_memberships.get(neighbor, [])
                    for neighbor_cluster in neighbor_clusters:
                        if cluster == neighbor_cluster:
                            internal_edges[cluster] += 1
                        else:
                            external_edges[cluster][neighbor_cluster] += 1

        for i in range(num_clusters):
            internal_edges[i] //= 2
            for j in range(num_clusters):
                external_edges[i][j] //= 2

        scohi_sum = sum((internal_edges[i] / (entities[i]**2) if entities[i] > 0 else 0) for i in range(num_clusters))
        scopij_sum = sum((external_edges[i][j] / (2 * (entities[i] * entities[j])) if entities[i] > 0 and entities[j] > 0 else 0) for i in range(num_clusters) for j in range(num_clusters) if i != j)

        sm = (1/num_clusters) * scohi_sum - (1 / (2 * (num_clusters-1))) * scopij_sum if num_clusters > 1 else 0
        return sm
    
    def calculate_ifn(G, cluster_memberships, num_clusters):
        # Initialize calls matrix
        calls_between_clusters = [[0] * num_clusters for _ in range(num_clusters)]

        # Populate calls matrix based on cluster memberships and graph edges
        for node in G.nodes():
            node_clusters = cluster_memberships[node]
            for neighbor in G.neighbors(node):
                neighbor_clusters = cluster_memberships[neighbor]
                for node_cluster in node_clusters:
                    for neighbor_cluster in neighbor_clusters:
                        if node_cluster != neighbor_cluster:
                            calls_between_clusters[node_cluster][neighbor_cluster] += 1

        # Calculate IFN for each cluster and overall
        ifn_values = []
        for i in range(num_clusters):
            ifn_i = sum(1 for j in range(num_clusters) if i != j and calls_between_clusters[i][j] > 0)
            ifn_values.append(ifn_i)

        ifn = sum(ifn_values) / num_clusters if num_clusters > 0 else 0
        return ifn
    def get_cluster_sizes(cluster_memberships):
        cluster_sizes = {}
        for node, clusters in cluster_memberships.items():
            for cluster in clusters:
                if cluster in cluster_sizes:
                    cluster_sizes[cluster] += 1
                else:
                    cluster_sizes[cluster] = 1
        return cluster_sizes
    
    
    
    def calculate_ned(cluster_memberships, G):
        cluster_sizes = get_cluster_sizes(cluster_memberships)
        total_classes = sum(cluster_sizes.values())
        non_extreme_classes = sum(size for size in cluster_sizes.values() if 5 <= size <= 20)

        if total_classes > 0:
            ned = 1 - (non_extreme_classes / total_classes)
        else:
            ned = 1  # Default to 1 if there are no classes at all

        return ned
    def calculate_icp(G,cluster_memberships, num_clusters):
        calls_between_clusters = get_calls_matrix(G, cluster_memberships, num_clusters)
        total_calls = sum(sum(calls_between_clusters[i][j] for j in range(num_clusters) if i != j) for i in range(num_clusters))

        icp_values = {}
        if total_calls > 0:
            for i in range(num_clusters):
                for j in range(num_clusters):
                    if i != j:
                        icp_i_j = calls_between_clusters[i][j] / total_calls
                        icp_values[(i, j)] = icp_i_j
        else:
            for i in range(num_clusters):
                for j in range(num_clusters):
                    if i != j:
                        icp_values[(i, j)] = 0

        return icp_values

def get_calls_matrix(G, cluster_memberships, num_clusters):
    calls_between_clusters =[[0 for _ in range(num_clusters)] for _ in range(num_clusters)]
    for node in G.nodes:
        node_cluster = cluster_memberships[node]
        for neighbor in G.neighbors(node):
            neighbor_cluster = cluster_memberships[neighbor]
            if node_cluster != neighbor_cluster:
                # Increment the integer value, ensuring node_cluster and neighbor_cluster are indices
                calls_between_clusters[node_cluster[0]][neighbor_cluster[0]] += 1

    return calls_between_clusters

    

def get_cluster_sizes(cluster_memberships):
            cluster_sizes = {}
            for node, clusters in cluster_memberships.items():
                for cluster in clusters:
                    if cluster in cluster_sizes:
                        cluster_sizes[cluster] += 1
                    else:
                        cluster_sizes[cluster] = 1
            return cluster_sizes



    
    




    


