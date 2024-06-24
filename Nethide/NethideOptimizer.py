#!/usr/bin/env python

import sys
import random
from networkx.algorithms.flow.capacityscaling import capacity_scaling 
import numpy as np
import networkx as nx 
import gurobipy as gp

from gurobipy import GRB
import Levenshtein as LD
import matplotlib.pyplot as plt
import time 
import geopy.distance
from numpy.lib.function_base import append

from Util.config import DEBUG,ITERATION,LINKCAPACITY_PERCENTAGE
from Util.common import atf,edg
# from .. import config 

#ITERATION = 200
# Iterration = 200 is a stable value from nethide (Figure 9 in nethide), whose acc and uti value are near the optimial one.
# Iteration = 100, cost 400s and 20GiB memory, (Run UsCarrier)
# Iteration = 200, cost 843s and 40GiB memory. (Run UsCarrier)
#LINKCAPACITY_PERCENTAGE = 0.9989
FR = 0
TOPOLOGY = 'Abilene.gml'
LOGFILE_NAME = TOPOLOGY + str(time.localtime().tm_mon) + "_"+str(time.localtime().tm_mday) 
# The percentage of link capacity can be changed from 0.1 to 1.0, the link capacity from nethide is defined as MAX_FLOW_DENSITY * <this parameter>. 
# 0.1 is the worst case, which means the attackers are the strongest.
# DEBUG = True


def fr_linkcapacity(avg_fd,fr): 
    link_c = (1 - fr) * avg_fd 
    return int(link_c)

#TOPOLOGY = 'SwitchL3.gml'
#TOPOLOGY = 'Chinanet.gml'
# 

def avg_path_length(topo,Flowset):
    path_len = nx.avg_shortest_path_length(topo)
    return path_len
    
def DistanceCalculate(cood1,cood2):
    #cood1 = (latitude1,longitude1)
    #cood2 = (latitude2,longtitude2)
    return geopy.distance.geodesic(cood1,cood2).km

def fd_calcu_from_tree_nh(topo,fwd_trees):
    for edge in topo.edges:
        topo.edges[edge]["FlowDensity_NH"] = 0
        topo.edges[edge]["NH_ATF_SET"] = []
    
    for node_dst in topo.nodes:
        fdt = fwd_trees[node_dst]
        for node_src in fdt: 
            path = fdt[node_src]
            edges = get_edges(path)
            for _e in edges:
                topo.edges[_e]["FlowDensity_NH"] +=1
                topo.edges[_e]["NH_ATF_SET"].append((node_src,node_dst))
    avg_fd = 0
    max_fd = 0
    min_fd = 0
    for edge in topo.edges:
        fd =  topo.edges[edge]['FlowDensity_NH']
        max_fd = fd if fd > max_fd else max_fd
        min_fd = fd if fd < min_fd else min_fd
        avg_fd +=fd
    avg_fd = avg_fd / len(topo.edges)
    print("[NewCalculate][]Virtual topo avgfd and maxfd is {} and {}".format(avg_fd,max_fd))
    return avg_fd,max_fd

def fd_calcu_result(topo,Flowset):
    for edge in topo.edges:
        topo.edges[edge]['FlowDensity'] = 0
        topo.edges[edge]['FD_ATF_SET'] = []

    for node_src in topo.nodes:
        fwd_tree = nx.shortest_path(topo,node_src)
        for idx in range(0,len(fwd_tree)):
            edges = get_edges(fwd_tree[idx])
            for _edge in edges:
                topo.edges[_edge]["FlowDensity"] += 1
                topo.edges[_edge]["FD_ATF_SET"].append(_edge)
    avg_fd = 0
    max_fd = 0
    min_fd = 0
    for edge in topo.edges:
        fd =  topo.edges[edge]['FlowDensity']
        max_fd = fd if fd > max_fd else max_fd
        min_fd = fd if fd < min_fd else min_fd
        avg_fd +=fd
    avg_fd = avg_fd / len(topo.edges)
    print("Virtual topo avgfd and maxfd is {} and {}".format(avg_fd,max_fd))
    return avg_fd,max_fd

def FlowDensityCalculate(topo,fr = 0):

    # Initialize  
    for edge in topo.edges:
        topo.edges[edge]['FlowDensity'] = 0
        topo.edges[edge]['FD_ATF_SET'] = []
    
    # Default attack policy in NetHide.
    for node_src in topo.nodes:
        fwd_tree = {}
        for node_dst in topo.nodes:
            route_tmp = list(nx.all_shortest_paths(topo,source=node_src,target=node_dst))
            route_tmp.sort()
            fwd_tree[node_dst] = route_tmp[0]

        
        # fwd_tree = nx.shortest_path(topo,node_src)
        for idx in range(0,len(fwd_tree)):
            edges = get_edges(fwd_tree[idx])
            for _edge in edges:
                topo.edges[_edge]["FlowDensity"] += 1
                topo.edges[_edge]["FD_ATF_SET"].append((node_src,idx))
            #end for, update edges fd
        #end for, every dst in the fwd_tree
    # end for , every src in the nodes

    avg_fd = 0
    max_fd = 0
    min_fd = 0
    for edge in topo.edges:
        fd =  topo.edges[edge]['FlowDensity']
        max_fd = fd if fd > max_fd else max_fd
        min_fd = fd if fd < min_fd else min_fd
        avg_fd +=fd
    avg_fd = avg_fd / len(topo.edges)
   


    if fr!=0:
        link_capacity = fr_linkcapacity(avg_fd,fr)
    else:
        link_capacity = int( max_fd * LINKCAPACITY_PERCENTAGE)
        fr = 1 - link_capacity / avg_fd

    #print("Flow Density Calculation, MAX_FD={},AVG_FD={},FR={}\n".format(max_fd,avg_fd,fr))
    
    #print(111)
    
    return max_fd,avg_fd
    
def NHAttackFlows(P):
    #TODO
    #This uses the NetHide default policy, which assumes all the nodes are ingress and egress
    atk_node_list = []
    for node in P.nodes:
        if P.nodes[node]["Internal"] == 1:
            atk_node_list.append(node)
    p_tmp = nx.Graph()
    p_tmp.add_nodes_from(atk_node_list)
    P_CompleteGraph = nx.complete_graph(p_tmp.nodes,create_using = nx.DiGraph)
    FlowSet = [edge for edge in P_CompleteGraph.edges] 
    #print(FlowSet)
    return FlowSet 

def NHAccuracyMetric(flow,P,V,x_index) -> float:
    #The x_index is the index of the pre-selected forwarding tree rooted at the target node.
    #This is a function that calculate the accuracymetric defined in NetHide [Security 2018]. 
    (source,destination) = flow
    
    def _LD(path1,path2):
        #TODO
        #Return the Levenshtein distance of two path.
        dist = LD.distance(path1,path2)
        return dist
    
    def getPath(G,x_index):
        node = G.nodes[destination]
        forwardingtree = node["NHtrees"][x_index]
        path = forwardingtree[source]
        return path
        link_path = []
        for idx in range(len(path)-1):
            link_path.append((path[idx],path[idx+1]))
        return link_path

    link_path1= getPath(P,0)
   
    link_path2 = getPath(V,x_index)
 
   
    acc = 1 - _LD(link_path1,link_path2)/(len(link_path1) + len(link_path2) )
    
    #print(acc)

    return acc

def NHUtilityMetric(flow,P,V,x_index) -> float:

    
    def getPath(G,x_index):
        node = G.nodes[destination]
        forwardingtree = node["NHtrees"][x_index]
        path = forwardingtree[source]
        link_path = []
        for idx in range(len(path)-1):
            link_path.append((path[idx],path[idx+1]))
        link_path = path
        return link_path
    
    def getPathFromsrctodst(G,src,dst):
        node = G.nodes[dst]
        forwardingtree = node["NHtrees"][0]
        path = forwardingtree[src]
        link_path = []
        for idx in range(len(path)-1):
            link_path.append((path[idx],path[idx+1]))
        link_path = path
        return link_path

    def getLenCommonLinks(link_path1,link_path2):
        C = 0
        set_links1 = set()
        set_links2 = set()
        
        def isequal_link(link_a,link_b):
            try:
                if link_a == link_b:
                    return True
                elif link_a[0] == link_b[1] and link_a[1] == link_b[0]:
                    return True
                else:
                    return False
            except:
                return False
            # end if 

        for link1 in link_path1:
            for link2 in link_path2:
                if isequal_link(link1,link2):
                    C +=1
                else:
                    continue
                #end if . 
            # end for, for each link in link2
        # end for, for each link in link1

        return C
       
        # for idx in range(0,len(path1)-1):
        #     set_links1.add((path1[idx],path1[idx+1]))
        #     set_links1.add((path1[idx+1],path1[idx]))
        # for idx in range(0,len(path2)-1):
        #     set_links2.add((path2[idx],path2[idx+1]))
        #     set_links2.add((path2[idx+1],path2[idx]))
        # for link in link_path1:
        #     set_links1.add(link)
        #     set_links1.add((link[1],link[0]))
        # for link in link_path2:
        #     set_links2.add(link)
        #     set_links2.add((link[1],link[0]))

        # C = set_links1.intersection(set_links2)
        #return len(C)


    (source,destination) = flow
    virtual_path = getPath(V,x_index)

    u_all = 0

    for idx in range(len(virtual_path)):
        #phy_path = getPathFromsrctodst(P,source,virtual_path[idx][1])
        phy_path = getPathFromsrctodst(P,source,virtual_path[idx])

        virtual_path_slice = virtual_path[0:idx+1]
        C = getLenCommonLinks(phy_path,virtual_path_slice)
        if C > len(phy_path) or C > len(virtual_path_slice):
            print("C is error,C={},phy_len={},and Virtual len={}".format(C,len(phy_path),len(virtual_path_slice)))
        u_n = 0.5 * ( C/len(phy_path) + C/len(virtual_path_slice))
        # u_n = ( C/len(phy_path))

        u_all += u_n
    uti = u_all / len(virtual_path)
    #print(uti)
    # path2 = getPath(V,x_index)
    # print(path2)
    # u_all = 0
    # for idx in range(0,len(path2)):
    #     if idx ==0:
    #         continue
    #     path1 = _getPathFromDst(P,source,path2[idx])
    #     #print(path2[idx],path1)
    #     C = getLenCommonLinks(path1,path2)
    #     u_n = 0.5 * (C/len(path1) + C/len(path2[0:idx+1]))
    #     u_all += u_n
    # uti = u_all / len(path2)

    return uti

def UniformlyAssignWeights(P):
    for edge in P.edges:
        #P.edges[edge]['weight']  = random.randint(1,10) 
        P.edges[edge]['weight'] = random.uniform(1,10)
        # This is defined in NetHide. May be the random is float. But int or float do not influence the results of nethide.
        # we cannot contact with the authors. We think int is good, it performs just like it in the paper. 
    return 

def addForwardingTreesFromAtoB(A,B):
    for idx in range(len(B.nodes)):
        tmp = A.nodes[idx]["NHtrees"][0]
        B.nodes[idx]["NHtrees"].append(tmp)
        fdtmp = NHFDmatrix(tmp)
        B.nodes[idx]["NHFD"].append(fdtmp)
    return

def ProduceForwardingTrees(V,iteration,P):
    #pre-calculate forwardingtrees for NetHide, with the method in Section 4.4
    #Initialize
    for edge in V.edges:
        V.edges[edge]['weight'] = 1        
    #UniformlyAssignWeights(P)
    for node in V.nodes:
        V.nodes[node]['NHtrees'] = []
        

        
        tmp = nx.shortest_path(V, target = node, weight = "weight")
        V.nodes[node]["NHtrees"].append(tmp)
        # NHFD is unuesd in my code, instead, I use Link_Capacity_Cons function below.
        # fdtmp = NHFDmatrix(tmp)
        # V.nodes[node]["NHFD"] = []
        # V.nodes[node]["NHFD"].append(fdtmp)
    P11 = P
    P11 = nx.convert_node_labels_to_integers(P11)
    for node in V.nodes:
        fwd_tree = {}
        for node_src in P11.nodes:
            route_tmp = list(nx.all_shortest_paths(P11,source=node_src,target=node))
            route_tmp.sort()
            fwd_tree[node_src] = route_tmp[0]
        
        tmp = fwd_tree
        
        # tmp = nx.shortest_path(P11,target = node)
        V.nodes[node]["NHtrees"].append(tmp)
        # fdtmp = NHFDmatrix(tmp)
        # V.nodes[node]["NHFD"].append(fdtmp)
        # app += fdtmp
    #print(app)
           
    #Calculate number='iteration' of the forwarding trees randomly. 
    for i in range(0,iteration):
        UniformlyAssignWeights(V)
        for node in V.nodes:
            #UniformlyAssignWeights(V)
            tmp = nx.shortest_path(V, target = node, weight = "weight")
            V.nodes[node]['NHtrees'].append(tmp)
            # fdtmp = NHFDmatrix(tmp)
            # V.nodes[node]["NHFD"].append(fdtmp)
    
    # for edge in P.edges:
    #     P.edges[edge]['weight'] = 1
    # for node in P.nodes:
    #         tmp = nx.shortest_path(P,source = node, weight = "weight")
    #         P.nodes[node]['NHtrees'].append(tmp)
    #         fdtmp = NHFDmatrix(tmp)
    #         P.nodes[node]["NHFD"].append(fdtmp)
    
    return

def _produceforwardingtree(P):
    for edge in P.edges:
        P.edges[edge]['weight'] = 1        
    for node in P.nodes:
        P.nodes[node]['NHtrees'] = []
        
        fwd_tree = {}
        for node_src in P.nodes:
            route_tmp = list(nx.all_shortest_paths(P,source=node_src,target=node))
            route_tmp.sort()
            fwd_tree[node_src] = route_tmp[0]
        tmp = fwd_tree 
        #tmp = nx.shortest_path(P, target = node, weight = "weight")
        P.nodes[node]["NHtrees"].append(tmp)
    return

class NHTopology(object):
    def __init__(self,nxgraph,T) -> None:
        super().__init__()
        self.nxgraph = nxgraph
        self.T = T

class NTVirtualTopology(object):
    def __init__(self,x_binary,Links):
        super().__init__()
        self.x_binary = x_binary
        self.Links = Links
    
    def isSecure(self):
        for link in self.Links:
            return 

def NHFDmatrix(tree) -> np.ndarray :
    fd = np.full((len(tree),len(tree)),0)
    for values in tree.values():
        for idx in range(len(values)-1):
            source = values[idx]
            dest = values[idx+1]
            fd[source,dest] +=1
    #print(fd)
    #print(tree)
    return fd

def Cij_Metric(FlowSet,P,V,iteration) -> np.ndarray:
    
    AccMetricMatrix = np.full((len(P.nodes),iteration),0.0)
    UtiMetricMatrix = np.full((len(P.nodes),iteration),0.0)

    for flow in FlowSet:
        i = flow[0] # Source

        for j in range(0,iteration):
            acctmp = NHAccuracyMetric(flow,P,V,j) 

            AccMetricMatrix[i,j] += acctmp
            utiltmp = NHUtilityMetric(flow,P,V,j)

            UtiMetricMatrix[i,j] += utiltmp
        
    C_metric = AccMetricMatrix * 0.5 + UtiMetricMatrix * 0.5 
    return C_metric

def Array_to_tupledict(C_metric):
    C_tupledict = gp.tupledict()
    for i in range(len(C_metric)):
        for j in range(len(C_metric[i])):
            C_tupledict[(i,j)] = C_metric[i,j]
    return C_tupledict
    
def FD_to_tupledict_list(nodes):
    ourdict = {}
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            tmparray = np.full((len(nodes),ITERATION),0)
            for i in range(len(nodes)):
                for j in range(ITERATION):
                    tmparray[i,j] = nodes[i]["NHFD"][j][k,l]
            ourdict[(k,l)] = tmparray 
    return ourdict

def get_edges(path_list)->list: # return list of edges
    edge_list = []
    idx = 0
    while idx +1 < len(path_list):
        edge_list.append((path_list[idx],path_list[idx+1]))
        idx = idx + 1
    return edge_list

def get_optimal_value(OptimalSolution,flowset,P,V):

    acc = 0
    uti = 0

    for flow in flowset:
        acc += NHAccuracyMetric(flow,P,V,OptimalSolution[flow[0]])
        uti += NHUtilityMetric(flow,P,V,OptimalSolution[flow[0]])
    acc = acc / len(flowset)
    uti = uti / len(flowset)
    return acc,uti

def Link_Capacity_Cons(topo,iterations):
    
    def get_edge_from_list(l):
        list_of_edge = []
        for idx in range(len(l)-1):
            list_of_edge.append((l[idx],l[idx+1]))
        return list_of_edge

    def get_edge_fd_from_tree(e,tree):
        # e is the edge's dict with fd.
        e_fd = 0
        edges_fd = []
        for source in tree:
            edges_fd += get_edge_from_list(tree[source])
            # edge_list = get_edge_from_list(tree[source])
            # for es in edge_list:
            #     edges_fd.append(es)
        #print(edges_fd)
        #print(e)
        # while e in edges_fd:
        #     e_fd +=1
        #     edges_fd.remove(e)
        # e_reverse = (e[1],e[0])
        # while e_reverse in edges_fd:
        #     e_fd +=1
        #     edges_fd.remove(e_reverse)
        e_fd += edges_fd.count(e)
        e_fd += edges_fd.count((e[1],e[0]))
        return e_fd
    
    def get_edge_fd_from_tree_all(tree,i,j):
        
        edge_list_all = []
        for source in tree:
            edge_list_all += get_edge_from_list(tree[source])
        
        edge_set = set(edge_list_all)
        #print(edge_set)
        #print("finished a tree")
        for item in edge_set:
            topo.edges[item]["Const_C"][(i,j)] += edge_list_all.count(item)
            #print(edge_list_all.count(item))
            pass
        pass


    # Initialize
    for edge in topo.edges:
        topo.edges[edge]["Const_C"] = {}
        for i in range(len(topo.nodes)):
            for j in range(iterations):
                topo.edges[edge]["Const_C"][(i,j)] = 0
            #end for j ,column
        # end for i, row
    # end for , all egdes.
    if DEBUG:
        print("FD initialized..")
    
    for node in topo.nodes:
        #print(f"start for {node}")
        for j in range(iterations):
            tree = topo.nodes[node]["NHtrees"][j]
            get_edge_fd_from_tree_all(tree,node,j)
            # for edge in topo.edges:
            #     inc_fd = get_edge_fd_from_tree(edge,tree)
            #     topo.edges[edge]["Const_C"][(node,j)] += inc_fd
                #print(edge,inc_fd)
            #end for, every edge inc from this tree
        #end for. every X[*,j]
    # end for , every X[i, *]
    
    # debug
    # for edge in topo.edges:
    #     topo.edges[edge]['Const_C'] = gp.tupledict(topo.edges[edge]["Const_C"])
    #     print(topo.edges[edge]["Const_C"])
    return

def gurobi_opt(P,link_capacity):    
    # return values: 1. forwarding trees; 2. log values; 3. virtual topology 
    # 2. logvalues : avg_fd_v,avg_fd_before,acc,uti,link_capacity,real_fr    
    def LogTheOptimalVal(filename,OptimalSolution):
        with open(filename,"w") as fn:
            fn.write(time.asctime( time.localtime(time.time()) ))
            fn.write('\n')
            for keys in OptimalSolution:
                tmpFT = P_CompleteGraph.nodes[keys]["NHtrees"][OptimalSolution[keys]]
                fn.write(str(tmpFT))
                fn.write('\n')
            fn.close()
        return
    
    #P = nx.read_gml(TOPOLOGY)
    P_CompleteGraph = nx.complete_graph(P.nodes)
    TP_ComleteGraph = []
    P_CompleteGraph = nx.convert_node_labels_to_integers(P_CompleteGraph)
    P = nx.convert_node_labels_to_integers(P)
    for node in P_CompleteGraph:

        
        
        tmp = nx.shortest_path(P_CompleteGraph,target = node)
        TP_ComleteGraph.append(tmp)

    tmp,avg_fd_before = FlowDensityCalculate(P,fr = FR)
    if DEBUG:
        print("NetHide Produces Forwarding trees....")
    ######################################
    # NetHide Producing Forwarding Trees #
    ######################################
    # this fuction produces forwarding trees for the initial topology
    # which is used as the actual routing path 
    _produceforwardingtree(P)
    # randomly producing ITERATION trees
    ProduceForwardingTrees(P_CompleteGraph,ITERATION,P)
    #addForwardingTreesFromAtoB(P,P_CompleteGraph)
    if DEBUG:
        print("NetHide Calculates FD for each link....")
    ######################################
    # NetHide Calculate FD for each link #
    ######################################
    Link_Capacity_Cons(P_CompleteGraph,ITERATION)
    if DEBUG:
        print("Link Capacity finished, generating attack flows")
    flowset = NHAttackFlows(P)
    if DEBUG:
        print("Attack flow set init finished, next is generating C metric")
    c_metric = Cij_Metric(flowset,P,P_CompleteGraph,ITERATION)
    #print(c_metric)
    #print(P_CompleteGraph.nodes[0]["NHFD"][2])

    # ############################ #
    # #NetHide Gurobipy Optimizer# #
    # ############################ #
    if DEBUG:
        print("NetHide Prepares model for optimize....")
    NHModel = gp.Model("NetHide")
    NHModel.setParam("OutputFlag", 0)
    NHModel.setParam("LogFile", P.graph.get("label","topo")+'-nethide-gurobilog.txt')
    X = NHModel.addVars(len(P.nodes),ITERATION,vtype = GRB.BINARY,name="X")
    C = Array_to_tupledict(c_metric) # C = acc(i,j) + util(i,j)
    #FD_c = FD_to_tupledict_list(P_CompleteGraph.nodes)
    #NHModel.setObjective(X.prod(C),GRB.MINIMIZE)
    NHModel.setObjective(X.prod(C),GRB.MAXIMIZE)
        
    ################################
    #    Net Hide Add Constrains   #
    ################################
    
    NHModel.addConstrs((X.sum(i,'*') == 1 for i in range(len(P.nodes))),name = "C1")
    i_idx_tmp = 2
    for edge in P_CompleteGraph.edges:
        fd_tuple = P_CompleteGraph.edges[edge]["Const_C"]
        NHModel.addConstr(X.prod(fd_tuple) <= link_capacity, name = "C"+str(i_idx_tmp))
        i_idx_tmp +=1

    # for i in range(len(P.nodes)):
    #     for j in range(len(P.nodes)):
    #         fd_metric = FD_c[(i,j)]
    #         fd_tuple = Array_to_tupledict(fd_metric)

    #         #print(fd_metric)
    #         #print(fd_tuple)
    #         NHModel.addConstr(X.prod(fd_tuple) <= link_capacity,name = "C"+str(i)+str(j))
    NHModel.Params.MIPGap = 0.02 
    NHModel.optimize()
    mm = gp.tupledict(X)
    print("Model's Objective Value is {}".format(NHModel.ObjVal/len(flowset)))

    OptimalSolution = {}
    for i in range(len(P.nodes)):
        for j in range(ITERATION):        
            sas= str(X[i,j]).split(' ')
            if sas[-1] == '1.0)>':
                #print("Optimial Forwarding Tree For Node {} is number {}".format(i,j))
                OptimalSolution[i] = j
    acc,uti = get_optimal_value(OptimalSolution,flowset,P,P_CompleteGraph)
    print("Optimal Sulution for This topology, acc={},uti={}".format(acc,uti))
    #LogTheOptimalVal("NHLoginfo",OptimalSolution)
    
        # for keys in OptimalSolution:
    #     OptimalSolution[keys] = 1
    # acc,uti = get_optimal_value(OptimalSolution,flowset,P,P_CompleteGraph)
    # print("Opt values that should for This topology, acc={},uti={}".format(acc,uti))
    # ################################## #
    #  NetHide Virtual Topology Analysis #
    # ################################## #
    #print(X.prod(fd_tuple))
    V = nx.Graph()
    V_1 = nx.DiGraph()
    # Why digraph? because the nethide gives the forwarding tree instead of adding realistic links.
    # The forwarding behavior in Nethide, by adding links, is similar to a digraph.
    
    V.add_nodes_from(P)
    V_1.add_nodes_from(P) 

    return_trees = {}
    for keys in OptimalSolution:
        tmpFT = P_CompleteGraph.nodes[keys]["NHtrees"][OptimalSolution[keys]]
        return_trees[keys] = tmpFT
        #print(tmpFT)
        for node in tmpFT:
            _edges = get_edges(tmpFT[node])
            _edges_di = get_edges(tmpFT[node])
            for _edge in _edges:
                if _edge in V.edges:
                    _edges.remove(_edge)
                # end if 
            # end for , adding all edges in this path
            V.add_edges_from(_edges)
            for _edge in _edges_di:
                if _edge in V_1.edges:
                    _edges_di.remove(_edge)
            V_1.add_edges_from(_edges)
        # end for, for all pathes from src to all nodes.
    #end for , for all sources
    #nx.draw(V_1,pos=nx.circular_layout(V_1))
    #plt.show()
    #nx.draw(P,pos=nx.circular_layout(P))

    avg_fd_v,max_fd = fd_calcu_result(V,flowset)
    # real_fr = 1 - avg_fd_v/ avg_fd_before
    # print("The Final FR of this capacity is:{}".format(real_fr))
    avg_fd,max_fd = fd_calcu_from_tree_nh(V,return_trees)
    real_fr = 1 - avg_fd/ avg_fd_before
    print("[New]The Final FR of this capacity is:{}, while the link_capacity is {}".format(real_fr,link_capacity))
    #nx.draw(V,pos=nx.circular_layout(V))
    #plt.show()
    # log_values(avg_fd_v,avg_fd_before,acc,uti,link_capacity,real_fr)
    log_values_line = [real_fr,acc,uti,avg_fd_v,avg_fd_before,link_capacity]
    return return_trees, log_values_line, V

def log_values(avg_fd_v,avg_fd_before,acc,uti,link_capacity,real_fr):
               
    with open(LOGFILE_NAME, 'a') as p :
        # ass = p.readline()
        # if not ass:
        #     p.write("realfr,acc,uti,avg_fd_v,avg_fd_before,link_capacity\n")
        # else:    
        line = [real_fr,acc,uti,avg_fd_v,avg_fd_before,link_capacity]
        line = str(line)[1:-1]
        p.write(line + '\n')
        p.close()
    return

if __name__ == "__main__":

    P = nx.read_gml(TOPOLOGY,label='id')
    #P = nx.convert_node_labels_to_integers(P)
    max_fd,avg_fd = FlowDensityCalculate(P)
    c = max_fd
    while c > int(0.1 * max_fd):
        gurobi_opt(P,c)
        c -= 1
    
    
    
    # if len(sys.argv)>=2:
    #     link_capacity = sys.argv[1]
    # else:
    #     print('Please input link capacity')
    #     exit(0)



    
   