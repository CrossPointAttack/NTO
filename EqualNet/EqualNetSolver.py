# This is a code that establishes the key logic of EqualNet [NDSS 2022]
# Only uesed for learn and experiments.
# We have tested the results and find it is consistent with the thesis.

import geopy.distance
import networkx as nx
from networkx.algorithms.operators.unary import reverse
from networkx.exception import NetworkXError
from networkx.readwrite.json_graph import node_link
import os
import random
import time
import pandas as pd
import itertools
import multiprocessing
import json

from Util.config import DEBUG
from Util.config import Leakage_reduction_factor,Number_virtual_nodes
from Util.common import atf,edg,atf_set_init


def _atf_set_init(phy_topo):
    _atf_set = _attack_flows(phy_topo)
    #print(_atf_set)
    atf_set = []
    for att_flow in _atf_set:
        #print(att_flow)
        tmp = atf(src=att_flow[0],dst=att_flow[1])
        atf_set.append(tmp)
    for atf_flow in atf_set:
     # In some topology, where an atf have multiple same-weight routes, 
     # the shortest_path function randomly choose one.
     # So, the shortest_path should be use only once in each experiment.
        route_all = nx.all_shortest_paths(phy_topo,source=atf_flow.src,target=atf_flow.dst)
        route_all.sort()
        atf_flow = route_all[0]
        # atf_flow.route = nx.shortest_path(phy_topo,source=atf_flow.src,target=atf_flow.dst)
        #atf_flow.delay = get_path_delay(atf_flow.route,topo)
        atf_flow.weight = 0
    return atf_set

def _attack_flows(phy_topo): 
    #TODO
    #This generation inputs all possible src-dst pairs into defenders.
    atk_node_list = []
    for node in phy_topo.nodes:
        if phy_topo.nodes[node]["Internal"] == 1:
            atk_node_list.append(node)
    p_tmp = nx.Graph()
    p_tmp.add_nodes_from(atk_node_list)
    P_CompleteGraph = nx.complete_graph(p_tmp.nodes,create_using = nx.DiGraph)
    FlowSet = [edge for edge in P_CompleteGraph.edges] 
    #print(FlowSet)
    return FlowSet         

def assign_split_time(logical_topo):
    for node in logical_topo.nodes:
        logical_topo.nodes[node]["split_time"] = 1
        logical_topo.nodes[node]["virtual_fd"] = \
        logical_topo.nodes[node]['fd']/logical_topo.nodes[node]["split_time"]
    return logical_topo

# def assign_ip_address(phy_topo):
#     #not use
#     with open('ipaddr.json','r') as f:
#         ip_addr_list = json.load(f)
#         f.close()
#     ip_addr = random.sample(ip_addr_list,len(t.nodes))
#     for ip,node in zip(ip_addr,t.nodes):
#         t.nodes[node]["IP"] = ip
#         print(t.nodes[node])
#     return phy_topo

def flow_density_calculate(topo,atf_set):    
    def _get_edges(path_list)->list: # return list of edges
        edge_list = []
        idx = 0
        while idx +1 < len(path_list):
            edge_list.append((path_list[idx],path_list[idx+1]))
            idx = idx + 1
        return edge_list
   # Initilize 
    for edge in topo.edges:
        topo.edges[edge]['fd'] = 0
        topo.edges[edge]['atf_set'] = [] 
    for node in topo.nodes:
        topo.nodes[node]['fd'] = 0
        topo.nodes[node]['atf_set'] = []
    # Start calculate
    for atf in atf_set:
        route = atf.route
        edges = _get_edges(route)
        for _edge in edges:
            topo.edges[_edge]["fd"] += 1
            topo.edges[_edge]["atf_set"].append((atf.src,atf.dst))
        for node in route:
            topo.nodes[node]['fd'] += 1
            topo.nodes[node]['atf_set'].append((atf.src,atf.dst))
    for atf in atf_set:
        route = atf.route
        simi_p = 0
        for node in route:
            if topo.nodes[node]['fd'] > simi_p:
                simi_p = topo.nodes[node]['fd']
            atf.similarity_p = simi_p
    return topo,atf_set

# def _get_node_prev(di_topo,node)->list:
#     # This function needs the topology class to be a Digraph.
#     # We use the get_node_neighbor to get the successors and predecessors together.
#     # We calculate the flow_density of node using the successor only, in line with EqualNet alg.2
#     return prev_node

# def _get_node_next(di_topo,node)->list:
#     # This function needs the topology class to be a Digraph.
#     # We use the get_node_neighbor to get the successors and predecessors together.
#     # We calculate the flow_density of node using the successor only, in line with EqualNet alg.2
#     return next_node

# def _get_node_neighbor(any_topo,node)->list:
#     neighbor = list(any_topo.neighbor(node))
#     return neighbor_node

def get_max_fd_node(logical_topo):
    max_fd = 0
    node_number = 0
    for node in logical_topo.nodes:
        node_fd = logical_topo.nodes[node]['fd'] / logical_topo.nodes[node]["split_time"]
        #print(node_fd)
        if node_fd > max_fd:
            node_number = node
            max_fd = node_fd
    return node_number

def get_min_fd_node(logical_topo):
    min_fd = 1000000000000
    for node in logical_topo.nodes:
        if logical_topo.nodes[node]['fd'] < min_fd:
            node_number = node
            min_fd = logical_topo.nodes[node]['fd']
    return node_number

def get_cur_leakage(any_topo):
    id_max_node_fd = get_max_fd_node(any_topo)
    max_node_fd = any_topo.nodes[id_max_node_fd]['fd']/any_topo.nodes[id_max_node_fd]["split_time"]
    min_node_fd = any_topo.nodes[get_min_fd_node(any_topo)]['fd']
    leak_cur = max_node_fd - min_node_fd
    if DEBUG: 
        pass
        # print(f"max_nodeid = {id_max_node_fd} max_fd ={max_node_fd} min_fd = {min_node_fd} leak_cur = {leak_cur}")
    return leak_cur

def get_alw_leakage(o_threshold,leak_cur):
    leak_alw = (1 - o_threshold) * leak_cur
    if DEBUG:
        pass
        # print(f"leak_alw={leak_alw}")
    return leak_alw

# def get_logical_topo(phy_topo):
#     # alg 1
#     # In this paper, we use the logical_topo = phy_topo to be consistent with the Nethide configurations.
#     # Such configuration makes sense, representing a general situation that:
#     # Between any the routers, there are at least one Layer-2 switch.
#     # It can be configured with more IP addresses, for which we recommend using the ITDK dataset.
#     # https://publicdata.caida.org/datasets/topology/ark/ipv4/itdk/2010-01/
#     return logical_topo

def gen_virtual_topo(logical_topo,o_threshold):
    # alg 2
    # we do not use fixed number of virtual nodes as input, instead we use the alg to calculate the value
    cur_leak = get_cur_leakage(logical_topo)
    alw_leak = get_alw_leakage(o_threshold,cur_leak)
    while cur_leak > alw_leak:
        node_curr = get_max_fd_node(logical_topo)
        # print(f"get_curr_node :{node_curr} fd:{logical_topo.nodes[node_curr]['fd']}")
        logical_topo.nodes[node_curr]["split_time"] *= 2
        # print(f"split node {node_curr} to {logical_topo.nodes[node_curr]['split_time']}")
        cur_leak = get_cur_leakage(logical_topo)
    # print(f"cur_leak = {cur_leak} and alw_leak = {alw_leak}, process ends")
    for node in logical_topo.nodes:
        logical_topo.nodes[node]['virtual_fd'] = \
        logical_topo.nodes[node]['fd']/logical_topo.nodes[node]["split_time"]
    return logical_topo

def get_virtual_ip_of_node(virtual_topo,node_id):
    # This function change the virtual node ip addresses 
    split_time = virtual_topo.nodes[node_id]

def assign_ip_addr_randomly(phy_topo,addr):
    df = pd.read_csv(addr)
    random_node = random.sample(range(0,len(df)),len(phy_topo.nodes))
    for x_id,node in zip(random_node,phy_topo.nodes):
        ip_list = df.loc[x_id,'ip_addr']
        ip_list = ip_list[1:-1].split(', ')
        ip_list = list(set(ip_list))
        ip_list.remove('nan')
        ip_list_formal = [ip[1:-1] for ip in ip_list]
        xx_id = random.randint(0,len(ip_list)-1)
        phy_topo.nodes[node]["ip"] = ip_list_formal[xx_id]
        if DEBUG:
            phy_topo.nodes[node]["ip_list_debug"] = ip_list_formal
    return phy_topo

def assign_virtual_ip(any_topo):
    cur_ip = []
    # Initilize physical ip_address
    for node in any_topo.nodes:
        cur_ip.append(any_topo.nodes[node]["ip"])
    
    for node in any_topo.nodes:
        if any_topo.nodes[node]["split_time"] > 1:
            any_topo.nodes[node]["ip_virtual"] = []
            any_topo.nodes[node]["ip_virtual"].append(any_topo.nodes[node]["ip"])
            count = 1
            ip_prefix = any_topo.nodes[node]["ip"].split('.')[:-1]
            while count < any_topo.nodes[node]["split_time"]:
                ip_suffix = str(random.randint(1,254))
                ip_addr_tmp = ip_prefix + [ip_suffix]
                ip_addr = '.'.join(ip_addr_tmp)
                while ip_addr in cur_ip:
                    ip_suffix = str(random.randint(1,254))
                    ip_addr_tmp = ip_prefix + [ip_suffix]
                    ip_addr = '.'.join(ip_addr_tmp)
                # end while, find virtual ip until no conflict
                any_topo.nodes[node]["ip_virtual"].append(ip_addr)
                #print(f"virtual topo {node} add a virtual_ip {ip_addr}")
                cur_ip.append(ip_addr)
                count += 1
            #end while, assign split_time - 1 virtual ip address to each node
    #print(any_topo.nodes[0])
    return any_topo

def atf_eqinit(atf_set,virtual_topo):
    atf_iter = iter(atf_set)
    for ele in atf_iter:
        for node in ele.route:
            if virtual_topo.nodes[node]["split_time"] == 1:
                ip_addr = virtual_topo.nodes[node]['ip']
            else:
                ip_list = virtual_topo.nodes[node]['ip_virtual']
                ele_tuple = (ele.src,ele.dst)
                index_atf = virtual_topo.nodes[node]['atf_set'].index(ele_tuple)
                index_ip = index_atf % len(ip_list)
                ip_addr = ip_list[index_ip]
            ele.route_eq.append(ip_addr)
            ele.route_ip.append(virtual_topo.nodes[node]["ip"])
    return atf_set

def print_fd_debug(virtual_topo,node_list):
    my_v = []
    my_p = []
    for node in virtual_topo.nodes:
        v_fd = int(virtual_topo.nodes[node]['virtual_fd'])
        p_fd = int(virtual_topo.nodes[node]['fd'])
        my_v.append(v_fd)
        my_p.append(p_fd)
    my_v.sort()
    my_p.sort()
    #print('equalnet_virtual_fd:',my_v)
    print('equalnet_physical_fd:',my_p)
    atf_v = list(node_list.values())
    atf_v.sort()
    print("adversary probe fd:",atf_v)
    
def atf_flow_density(atf_eq_set):
    node_list = {}
    cur_node = []
    for atf in atf_eq_set:
        for ip in atf.route_eq:
            if ip in cur_node:
                node_list[ip] += 1
            else:
                cur_node.append(ip)
                node_list[ip] = 1
    return node_list

def sim_ip_density(atf_eq_set):
    node_list = {}
    cur_node = []
    for atf in atf_eq_set:
        for ip in atf.route_eq:
            ip_prefix_tmp = ip.split('.')[:-1]
            ip_prefix = '.'.join(ip_prefix_tmp)
            if ip_prefix in cur_node:
                node_list[ip_prefix] += 1
            else:
                cur_node.append(ip_prefix)
                node_list[ip_prefix] = 1
    return node_list

def update_ip_similarity_v(atf_eq_set,node_list):
    for atf in atf_eq_set:
        simi_max_v = 0
        for ip in atf.route_eq:
            ip_prefix_tmp = ip.split('.')[:-1]
            ip_prefix = '.'.join(ip_prefix_tmp)
            if node_list[ip_prefix] > simi_max_v:
                simi_max_v = node_list[ip_prefix]
        atf.similarity_v = simi_max_v
    return atf_eq_set

def EqualNet_a_topo(topo,ip_addr_dataset): 
    # ip_addr_dataset = './CAIDA dataset/as_id_node/ipaddr_13576.csv'
    # topo = './archive/UsCarrier.gml'
    # ======Start initialize===== # 
    start_time = time.time()
    # test_topo = topo
    # phy_topo = nx.read_gml(test_topo,label='id')
    phy_topo = topo
    atf_set = atf_set_init(phy_topo)
    phy_topo = assign_ip_addr_randomly(phy_topo,ip_addr_dataset)
    # ======Run EqualNet===== # 
    phy_topo,atf_set = flow_density_calculate(phy_topo,atf_set)
    l_topo = assign_split_time(phy_topo)
    virtual_topo = gen_virtual_topo(l_topo,Leakage_reduction_factor)
    virtual_topo = assign_virtual_ip(virtual_topo)
    gen_virtual_topo_time = time.time() 
    gen_virtual_topo_duration = gen_virtual_topo_time - start_time
    # ======Gen virtual links===== # 
    atf_eq_set = atf_eqinit(atf_set,virtual_topo)
    init_eq_set_time = time.time()
    init_eq_set_duration = init_eq_set_time - gen_virtual_topo_time
    # ======Adversary Probes===== # 
    node_list_crossfire = atf_flow_density(atf_eq_set)
    node_list_crosspoint = sim_ip_density(atf_eq_set)
    atf_eq_set = update_ip_similarity_v(atf_eq_set,node_list_crosspoint)
    # print_fd_debug(virtual_topo,node_list_crossfire)
    # print_fd_debug(virtual_topo,node_list_crosspoint)
    probe_time = time.time()
    probe_duration = probe_time - init_eq_set_time
    end_time = time.time()
    whole_duration = end_time - start_time
    if DEBUG:
        print("virtual topo gen:", gen_virtual_topo_duration)
        print("init set gen:",init_eq_set_duration)
        print("probe_duration:",probe_duration)
        print("whole_duration",whole_duration)
    return atf_eq_set