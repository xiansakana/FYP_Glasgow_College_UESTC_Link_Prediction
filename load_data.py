import numpy as np
import networkx as nx
from scipy.io import loadmat

def load_data(dataset):
    if dataset == "0":
        G = nx.Graph()
        path = './dataset/Celegans.txt'
        edge_list = []
        weight_edge_list = []
        node_set = set() 
        weight_set = set()
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                y3=int(cols[2])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
                weight_set.add(y3)
                weight_edge = (y1,y2,y3)
                weight_edge_list.append(weight_edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_weighted_edges_from(weight_edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network

    if dataset == "1":
        G = nx.Graph()
        path = './dataset/Contact.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network


    if dataset == "2":
        G = nx.Graph()
        path = './dataset/Dolphin.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network

    if dataset == "3":
        G = nx.Graph()
        path = './dataset/Email.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
 
    if dataset == "4":
            G = nx.Graph()
            path = './dataset/FWF.txt'
            edge_list = []
            node_set = set() 
            with open(path, 'r') as f:
                for line in f:
                    cols = line.strip().split('\t')
                    y1=int(cols[0])
                    y2=int(cols[1])
                    node_set.add(y1)
                    node_set.add(y2)
                    edge = (y1,y2) 
                    edge_list.append(edge)
            G.add_nodes_from(range(1,len(node_set))) 
            G.add_edges_from(edge_list)
            Original_Network = nx.adjacency_matrix(G).toarray()
            return Original_Network

    if dataset == "5":
            G = nx.Graph()
            path = './dataset/FWM.txt'
            edge_list = []
            node_set = set() 
            with open(path, 'r') as f:
                for line in f:
                    cols = line.strip().split('\t')
                    y1=int(cols[0])
                    y2=int(cols[1])
                    node_set.add(y1)
                    node_set.add(y2)
                    edge = (y1,y2) 
                    edge_list.append(edge)
            G.add_nodes_from(range(1,len(node_set))) 
            G.add_edges_from(edge_list)
            Original_Network = nx.adjacency_matrix(G).toarray()
            return Original_Network
   
    if dataset == "6":
        G = nx.Graph()
        path = './dataset/Jazz.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
    
    if dataset == "7":
        G = nx.Graph()
        path = './dataset/Karate.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network

    if dataset == "8":
        G = nx.Graph()
        path = './dataset/Macaca.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network  
       
    if dataset == "9":
        G = nx.Graph()
        path = './dataset/Metabolic.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
    
    if dataset == "10":
        G = nx.Graph()
        path = './dataset/Political_Blog.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
    
    if dataset == "11":
        G = nx.Graph()
        path = './dataset/USAir.txt'
        edge_list = []
        weight_edge_list = []
        node_set = set() 
        weight_set = set()
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                y3=float(cols[2])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
                weight_set.add(y3)
                weight_edge = (y1,y2,y3)
                weight_edge_list.append(weight_edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_weighted_edges_from(weight_edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
    
    if dataset == "12":
        G = nx.Graph()
        path = './dataset/World_Trade.txt'
        edge_list = []
        node_set = set() 
        with open(path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                y1=int(cols[0])
                y2=int(cols[1])
                node_set.add(y1)
                node_set.add(y2)
                edge = (y1,y2) 
                edge_list.append(edge)
        G.add_nodes_from(range(1,len(node_set))) 
        G.add_edges_from(edge_list)
        Original_Network = nx.adjacency_matrix(G).toarray()
        return Original_Network
    