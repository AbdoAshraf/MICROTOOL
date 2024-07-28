import os
import json
import mysql.connector
import networkx as nx
from transformers import BertModel, BertTokenizer
import torch
from openai import OpenAI
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_curve
import numpy as np
import community as community_louvain
import skfuzzy as fuzz
import plotly.express as px
import matplotlib.pyplot as plt

class DatabaseAnalysis:
    def __init__(self, db_config, api_key, descriptions_file):
        self.db_config = db_config
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.descriptions_file = descriptions_file
        self.connection = None
        self.db_clusters = None  # Initialize db_clusters as None

    def connect_to_database(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("Successfully connected to the database.")
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")

    def fetch_schema_data(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s;", (self.connection.database,))
        tables_columns = {}
        for table, column in cursor.fetchall():
            if table not in tables_columns:
                tables_columns[table] = []
            tables_columns[table].append(column)

        cursor.execute("SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL;", (self.connection.database,))
        foreign_keys = cursor.fetchall()
        cursor.close()
        return tables_columns, foreign_keys

    def generate_description_with_gpt(self, table_name, columns):
        chat_completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Describe the database table '{table_name}' which includes columns {', '.join(columns)}. explain its use cases."
                }
            ]
        )
        description = chat_completion.choices[0].message.content
        return description

    def load_descriptions(self):
        try:
            with open(self.descriptions_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_descriptions(self, descriptions):
        try:
            with open(self.descriptions_file, 'w') as f:
                json.dump(descriptions, f)
        except Exception as e:
            print(f"Error saving descriptions: {str(e)}")

    def ensure_descriptions(self, tables_columns):
        descriptions = self.load_descriptions()
        need_save = False

        for table, columns in tables_columns.items():
            if table not in descriptions or not descriptions[table]:
                print(f"Generating description for: {table}")
                description = self.generate_description_with_gpt(table, columns)
                if description:
                    descriptions[table] = description
                    need_save = True
                else:
                    descriptions[table] = "Description not available."

        if need_save:
            self.save_descriptions(descriptions)

        return descriptions

    def get_bert_embeddings(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            output = self.model(**encoded_input)
        embeddings = output.last_hidden_state[:, 0, :].squeeze()
        return embeddings

    def create_graph_with_descriptions(self, tables_columns, foreign_keys, descriptions):
        G = nx.DiGraph()
        embeddings = {table: self.get_bert_embeddings(desc) for table, desc in descriptions.items()}
        similarity_scores = []

        for src, src_emb in embeddings.items():
            for dest, dest_emb in embeddings.items():
                if src != dest and src_emb is not None and dest_emb is not None:
                    sim = util.pytorch_cos_sim(src_emb, dest_emb).item()
                    if sim is not None:
                        similarity_scores.append(sim)
                        G.add_edge(src, dest, weight=sim, label=f'Similarity: {sim:.2f}')

        for src_table, src_column, dest_table, dest_column in foreign_keys:
            if src_table in G.nodes and dest_table in G.nodes:
                foreign_key_weight = 1.0
                G.add_edge(src_table, dest_table, weight=foreign_key_weight, label=f"{src_column} -> {dest_column}", type='foreign_key')

        return G, similarity_scores

    def compute_elbow_threshold(self, similarity_scores):
        sorted_scores = np.sort(similarity_scores)
        gradients = np.diff(sorted_scores)
        elbow_index = np.argmax(gradients)
        threshold = sorted_scores[elbow_index]
        return threshold

    def apply_similarity_threshold(self, G, similarity_scores):
        if similarity_scores:
            threshold = self.compute_elbow_threshold(similarity_scores)
            print(f"Applying similarity threshold: {threshold}")

            for u, v, data in list(G.edges(data=True)):
                if data.get('weight', 0) < 0.9:
                    G.remove_edge(u, v)

        return G

    def apply_louvain_community_detection(self, G):
        if G.is_directed():
            G = G.to_undirected()
        partition = community_louvain.best_partition(G)
        return partition

    def plot_graph_with_communities(self, G, partition):
        if G.number_of_edges() == 0:
            print("No edges in the graph.")
            return

        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='MidnightBlue'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
                                text=[f"{node} (Community {partition[node]})" for node in G.nodes()],
                                marker=dict(showscale=True, colorscale='YlOrRd', size=10,
                                            color=[partition[node] for node in G.nodes()],
                                            colorbar=dict(title='Community',
                                                          tickvals=list(set(partition.values())),
                                                          ticktext=[f"Community {i}" for i in set(partition.values())],
                                                          xanchor='left', titleside='right'),
                                            line=dict(width=2, color='DarkSlateGrey')))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='Network Graph with Community Detection', showlegend=False,
                                         hovermode='closest', margin=dict(b=0, l=0, r=0, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        fig.show()

    def prepare_data_for_fcm(self, G, partition):
        nodes = list(G.nodes())
        communities = list(set(partition.values()))
        data = np.zeros((len(nodes), len(communities)))

        for i, node in enumerate(nodes):
            data[i, communities.index(partition[node])] = 1
        return data

    def apply_fcm(self, data, num_clusters):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, c=num_clusters, m=2, error=0.005, maxiter=1000, init=None)
        return u

    def build_enhanced_graph(self, G, u, threshold=0.1):
        F = nx.Graph()
        node_cluster_map = {}

        for j, node in enumerate(G.nodes()):
            node_cluster_map[node] = []
            for i in range(u.shape[0]):
                if u[i][j] > threshold:
                    new_node = f"{node}_C{i + 1}"
                    F.add_node(new_node, cluster=i + 1, original=node)
                    node_cluster_map[node].append(new_node)

        for (n1, n2) in G.edges():
            if n1 in node_cluster_map and n2 in node_cluster_map:
                for new_n1 in node_cluster_map[n1]:
                    for new_n2 in node_cluster_map[n2]:
                        if new_n1.split('_')[1] == new_n2.split('_')[1]:
                            F.add_edge(new_n1, new_n2, type='intra-cluster')
                        else:
                            F.add_edge(new_n1, new_n2, type='inter-cluster', style='dashed')

        return F

    def visualize_clustered_graph(self, F):
        pos = nx.spring_layout(F)
        edge_x = []
        edge_y = []
        for edge in F.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='Grey', dash='solid' if edge[2]['type'] == 'intra-cluster' else 'dash'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        text = []
        for node in F.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(f"{F.nodes[node]['original']} (Cluster {F.nodes[node]['cluster']})")

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=text,
                                marker=dict(showscale=True, colorscale='Viridis', size=10,
                                            color=[F.nodes[node]['cluster'] for node in F.nodes()], line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='Network Graph with FCM Clustering', showlegend=False,
                                         hovermode='closest', margin=dict(b=0, l=0, r=0, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        fig.show()
    def determine_cluster_memberships(self, G, u, threshold=.3):
        cluster_memberships = {node: [] for node in G.nodes()}  # Initialize for all nodes in G
        num_clusters = u.shape[0]

        for i in range(num_clusters):
            for j, node in enumerate(G.nodes()):
                if u[i][j] > threshold:
                    cluster_memberships[node].append(i)


        print(cluster_memberships)
        return cluster_memberships

    def start_database_analysis(self):
        self.connect_to_database()
        if self.connection:
            tables_columns, foreign_keys = self.fetch_schema_data()
            complete_descriptions = self.ensure_descriptions(tables_columns)
            G, similarity_scores = self.create_graph_with_descriptions(tables_columns, foreign_keys, complete_descriptions)
            G = self.apply_similarity_threshold(G, similarity_scores)
            partition = self.apply_louvain_community_detection(G)
            self.plot_graph_with_communities(G, partition)
            data = self.prepare_data_for_fcm(G, partition)
            u = self.apply_fcm(data, num_clusters=len(set(partition.values())))
            F = self.build_enhanced_graph(G, u)
            self.visualize_clustered_graph(F)
            self.db_clusters = self.determine_cluster_memberships(G, u) # Store db_clusters as an instance attribute
        else:
            print("Failed to connect to the database.")
