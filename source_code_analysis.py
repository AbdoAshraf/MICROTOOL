import os
import javalang
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import community as community_louvain
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig
import json
import math
from networkx.algorithms.community import kernighan_lin_bisection

class SourceCodeAnalysis:
    def __init__(self, java_directory, excel_filepath, db_clusters):
        self.java_directory = java_directory
        self.excel_filepath = excel_filepath
        self.db_clusters = db_clusters
        # self.bcs_per_class_filepath = bcs_per_class_filepath
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def start_source_code(self):
        df = pd.read_excel(self.excel_filepath)
        project_classes = self.extract_project_classes(self.java_directory)
        G, unmatched_classes = self.build_call_graph(df, project_classes)

        print("Unmatched Caller Classes:", unmatched_classes['caller'])
        print("Unmatched Called Classes:", unmatched_classes['called'])
        print("Unmatched Project Classes:", unmatched_classes['project'])

        features = self.extract_features(G)
        initial_communities = self.compute_initial_communities(features, self.db_clusters)
        clusters = self.apply_community_detection(G, initial_communities)
        clusters = self.apply_leiden_with_initial_communities(G, clusters)
        
        self.visualize_clustered_graph(G, clusters)
        
        services, internal_edges, inter_edges = self.collect_service_data(G, clusters)
        self.visualize_clustered_graph(G, clusters)
        services, internal_edges, inter_edges = self.collect_service_data(G, clusters)
        
        sm_score = self.compute_sm(services, internal_edges, inter_edges)
        print("SM Score:", sm_score)
        ifn_score = self.calculate_ifn(G, clusters, len(services))
        print("IFN Score:", ifn_score)
        ned_score = self.calculate_ned(services)
        print("NED Score:", ned_score)
        icp_values = self.calculate_icp(G, clusters)
        print("ICP values:", icp_values)
        
        # # bcs_per_class = self.load_bcs_per_class(self.bcs_per_class_filepath)
        # partition_class_bcs_assignment = self.transform_clusters_to_partition(clusters, bcs_per_class)
        
        results_filepath = 'results.xlsx'
        self.save_results_to_excel(clusters, services, sm_score, ifn_score, ned_score, icp_values, results_filepath)
        
    def extract_project_classes(self, java_directory):
        project_classes = set()
        for root, _, files in os.walk(java_directory):
            for file in files:
                if file.endswith(".java"):
                    class_name = os.path.splitext(file)[0]
                    project_classes.add(class_name)
        return project_classes
    
    def build_call_graph(self, df, project_classes):
        G = nx.DiGraph()
        unmatched_classes = {'caller': set(), 'called': set(), 'project': set()}
        
        for _, row in df.iterrows():
            caller_class = row['CallingClass'].split('.')[-1].split(':')[0].strip()
            called_class_raw = row['CalledClass'].strip()
            call_type = called_class_raw[1]
            called_class = called_class_raw[3:].split('.')[-1].split(':')[0].strip()
            
            if caller_class not in project_classes:
                unmatched_classes['caller'].add(caller_class)
            if called_class not in project_classes:
                unmatched_classes['called'].add(called_class)
                
            if caller_class in project_classes and called_class in project_classes and caller_class != called_class:
                if G.has_edge(caller_class, called_class):
                    G[caller_class][called_class]['weight'] += 1
                else:
                    G.add_edge(caller_class, called_class, weight=1, type=call_type)
                    
        excel_classes = set(df['CallingClass'].apply(lambda x: x.split('.')[-1].split(':')[0].strip())) \
                        .union(set(df['CalledClass'].apply(lambda x: x[3:].split('.')[-1].split(':')[0].strip())))
        unmatched_classes['project'] = project_classes - excel_classes
        
        return G, unmatched_classes
    
    def extract_features(self, G):
        class_name_embeddings = self.get_class_name_embeddings(G)
        pagerank = nx.pagerank(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        semantic_weight = 2
        features = {}

        for node in G.nodes():
            structural_features = [
                G.in_degree(node),
                G.out_degree(node),
                pagerank.get(node, 0),
                betweenness_centrality.get(node, 0),
                closeness_centrality.get(node, 0),
            ]
            weighted_semantic_features = (class_name_embeddings[node] * semantic_weight).tolist()
            combined_features = structural_features + weighted_semantic_features
            features[node] = combined_features

        return features

    def get_class_name_embeddings(self, G):
        embeddings = {}
        for node in G.nodes():
            embeddings[node] = self.model.encode(node)
        return embeddings

    def compute_initial_communities(self, features, db_clusters, threshold=50):
        initial_communities = {node: i for i, node in enumerate(features.keys())}
        assigned_communities = set()

        feature_keys = list(features.keys())
        normalized_keys = {node.lower(): node for node in feature_keys}

        for node, cluster_ids in db_clusters.items():
            node_key = node.lower()
            matches = process.extract(node_key, normalized_keys.keys(), limit=None, scorer=process.fuzz.token_sort_ratio)

            for match, score in matches:
                actual_key = normalized_keys[match]
                if score >= threshold and actual_key not in assigned_communities:
                    initial_communities[actual_key] = cluster_ids[0]
                    assigned_communities.add(actual_key)

        for node in features.keys():
            if node not in assigned_communities:
                print(f"Warning: No valid community assignment found for '{node}'.")

        return initial_communities

    def convert_to_undirected_and_combine_weights(self, G):
        H = nx.Graph()
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            if H.has_edge(u, v):
                H[u][v]['weight'] += weight
            else:
                H.add_edge(u, v, weight=weight)
        return H

    def apply_community_detection(self, G, initial_communities):
        H = self.convert_to_undirected_and_combine_weights(G)
        partition = community_louvain.best_partition(H, partition=initial_communities, randomize=False)
        return partition

    def apply_leiden_with_initial_communities(self, nx_graph, initial_communities=None):
        g = ig.Graph(directed=True)
        g.add_vertices(list(nx_graph.nodes()))

        edges = [(u, v) for u, v in nx_graph.edges()]
        weights = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
        g.add_edges(edges)
        g.es['weight'] = weights

        g.vs['label'] = list(nx_graph.nodes())

        if initial_communities:
            membership = [initial_communities.get(node) for node in nx_graph.nodes()]
            partition = la.RBConfigurationVertexPartition(g, initial_membership=membership, weights='weight')
        else:
            partition = la.RBConfigurationVertexPartition(g, weights='weight')

        optimiser = la.Optimiser()
        optimiser.optimise_partition(partition, n_iterations=-1)

        community_dict = {node: partition.membership[i] for i, node in enumerate(nx_graph.nodes())}
        return community_dict

    def visualize_clustered_graph(self, G, clusters):
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        colors = [clusters.get(node, 0) for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='#FF5733', node_size=2000, font_size=10, font_weight='bold', cmap=plt.cm.jet)
        plt.title('Java Application Dependency Graph with Clustering')
        plt.show()

    def collect_service_data(self, G: nx.Graph, clusters: dict) -> tuple:
        """
        Collects service data from the graph G based on the provided clusters.

        Args:
            G (nx.Graph): The input graph.
            clusters (dict): A dictionary mapping nodes to their respective cluster IDs.

        Returns:
            tuple: A tuple containing services, internal_edges, and inter_edges.
        """
        services = {}
        internal_edges = {}
        inter_edges = {}
        node_to_service = {node: clusters[node] for node in G.nodes() if node in clusters}

        for u, v, data in G.edges(data=True):
            service_u = node_to_service.get(u)
            service_v = node_to_service.get(v)
            weight = data.get('weight', 1)

            if service_u is not None and service_v is not None:
                if service_u == service_v:
                    internal_edges[service_u] = internal_edges.get(service_u, 0) + weight
                else:
                    edge_key = tuple(sorted((service_u, service_v)))
                    inter_edges[edge_key] = inter_edges.get(edge_key, 0) + weight

                if service_u not in services:
                    services[service_u] = {'entities': set(), 'nodes': set()}
                if service_v not in services:
                    services[service_v] = {'entities': set(), 'nodes': set()}

                services[service_u]['entities'].add(u)
                services[service_u]['nodes'].add(u)
                if u != v:
                    services[service_v]['entities'].add(v)
                    services[service_v]['nodes'].add(v)

        for service in services:
            services[service]['entities'] = len(services[service]['entities'])
            services[service]['nodes'] = list(services[service]['nodes'])

        return services, internal_edges, inter_edges


    def compute_sm(self, services, internal_edges, inter_edges):
        N = len(services)
        scohi_sum = 0
        scopi_sum = 0

        for i in services:
            Ni = services[i]['entities']
            ui = internal_edges.get(i, 0)
            scohi = ui / (Ni * Ni) if Ni > 0 else 0
            scohi_sum += scohi

            for j in services:
                if i != j:
                    Nj = services[j]['entities']
                    segmaij = inter_edges.get((i, j), 0) + inter_edges.get((j, i), 0)
                    scopi_j = segmaij / (Ni * Nj) if Ni > 0 and Nj > 0 else 0
                    scopi_sum += scopi_j

        average_scohi = scohi_sum / N if N > 0 else 0
        average_scopi = scopi_sum / ((N * (N - 1)) / 2) if N > 1 else 0

        sm = average_scohi - average_scopi
        return sm


    def collect_interface_data(self, G, clusters):
        cluster_interfaces = defaultdict(set)

        for u, v in G.edges():
            cluster_u = clusters.get(u)
            cluster_v = clusters.get(v)
            if cluster_u and cluster_v and cluster_u != cluster_v:
                cluster_interfaces[cluster_u].add((u, v))

        return cluster_interfaces

    def calculate_ifn(self, G, clusters, num_services):
        cluster_interfaces = self.collect_interface_data(G, clusters)
        total_interfaces = sum(len(interfaces) for interfaces in cluster_interfaces.values())
        ifn = total_interfaces / num_services if num_services > 0 else 0
        return ifn

    def calculate_ned(self, partition_class_bcs_assignment):
        total_classes = 0
        non_extreme_sum = 0

        for cluster_data in partition_class_bcs_assignment.values():
            size = cluster_data['entities']
            total_classes += size
            if 5 <= size <= 20:
                non_extreme_sum += size

        ned = 1 - (non_extreme_sum / total_classes) if total_classes > 0 else 1
        return round(ned, 3)

    def calculate_icp(self, G, clusters):
        K = len(set(clusters.values()))
        cluster_classes = {i: [] for i in range(K)}

        for cls, cluster_id in clusters.items():
            cluster_classes[cluster_id].append(cls)

        numerator = 0
        denominator = 0

        for i in range(K):
            for j in range(K):
                if i != j:
                    for c_k in cluster_classes[i]:
                        for c_l in cluster_classes[j]:
                            if G.has_edge(c_k, c_l):
                                edge_weight = G[c_k][c_l]['weight']
                                numerator += edge_weight
                                denominator += edge_weight
                else:
                    for c_k in cluster_classes[i]:
                        for c_l in cluster_classes[j]:
                            if G.has_edge(c_k, c_l):
                                edge_weight = G[c_k][c_l]['weight']
                                denominator += edge_weight

        icp = numerator / denominator if denominator != 0 else 0
        return icp

    def duplicate_important_nodes(self, G, clusters, important_nodes):
        new_edges = []
        print(important_nodes)
        for node in important_nodes:
            connected_communities = set(clusters[neighbor] for neighbor in G.neighbors(node) if clusters[neighbor] != clusters[node])
            original_community = clusters[node]

            for community in connected_communities:
                new_node = f"{node}_dup_{community}"
                clusters[new_node] = community

                for neighbor in G.neighbors(node):
                    if clusters[neighbor] == community:
                        new_edges.append((new_node, neighbor, G[node][neighbor]))
                for neighbor in G.predecessors(node):
                    if clusters[neighbor] == community:
                        new_edges.append((neighbor, new_node, G[neighbor][node]))

        for new_node, neighbor, edge_data in new_edges:
            G.add_edge(new_node, neighbor, **edge_data)
    
    def load_bcs_per_class(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def transform_clusters_to_partition(self, clusters, bcs_per_class):
        cluster_to_classes = defaultdict(list)
        for cls, cluster_id in clusters.items():
            cluster_to_classes[cluster_id].append(cls)
        
        partition_class_bcs_assignment = {}
        for cluster_id, class_ids in cluster_to_classes.items():
            for class_id in class_ids:
                if class_id in bcs_per_class:
                    partition_class_bcs_assignment[class_id] = {
                        'business_context': bcs_per_class[class_id]
                    }
        
        return partition_class_bcs_assignment

    def save_results_to_excel(self, clusters, services, sm_score, ifn_score, ned_score, icp_values, filename):
        clusters_data = {'Class': list(clusters.keys()), 'Cluster': list(clusters.values())}
        clusters_df = pd.DataFrame(clusters_data)
        
        services_data = {
            'Service': [],
            'Entities': [],
            'Nodes': []
        }
        
        for service_id, service_info in services.items():
            services_data['Service'].append(service_id)
            services_data['Entities'].append(service_info['entities'])
            services_data['Nodes'].append(', '.join(service_info['nodes']))
        
        services_df = pd.DataFrame(services_data)
        
        measurements_data = {
            'SM Score': [sm_score],
            'IFN Score': [ifn_score],
            'NED Score': [ned_score],
            'ICP Values': [icp_values]
        }
        measurements_df = pd.DataFrame(measurements_data)
        
        with pd.ExcelWriter(filename) as writer:
            clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
            services_df.to_excel(writer, sheet_name='Services', index=False)
            measurements_df.to_excel(writer, sheet_name='Measurements', index=False)
