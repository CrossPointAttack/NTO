import networkx as nx 
import Levenshtein as LD
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd 
import random
import itertools
import time
import multiprocessing

from Util.common import atf,edg,topo_to_link,atf_set_init,init_topo
from Util.config import SPIDER_NUM_FACTOR,STEP3_SHRINK_FACTOR,SIZE_OF_TCAM 
from Util.linkcp import link_cp_bics,link_cp_uscarrier,link_cp_viatel
from Nethide.NethideOptimizer import *

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)  

def NHAttackFlows(P):
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

def acc(atf,route_type="sp"):
    
    def _LD(path1,path2):
        dist = LD.distance(path1,path2)
        return dist
    
    link_path1 = atf.route
    if route_type == 'nh':
        link_path2 = atf.route_nh
    elif route_type == "eq":
        link_path2 = atf.route_eq
    elif route_type == "sp":
        link_path2 = atf.route_sp
        
    acc = 1 - _LD(link_path1,link_path2)/(len(link_path1) + len(link_path2) )
    return acc

def uti(atf,route_type="sp") -> float:
    
    def get_path(atf,route_type="sp"):
        if route_type == 'nh':
            path = atf.route_nh
        elif route_type == "eq":
            path = atf.route_eq
        elif route_type == "sp":
            path = atf.route_sp
        else:
            path = atf.route
        link_path = []
        for idx in range(len(path)-1):
            link_path.append((path[idx],path[idx+1]))
        link_path = path
        return link_path

    def getPathFromsrctodst(G,src,dst):
        path = nx.shortest_path(atf.Graph,source=src,target=dst)
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
    

    src,dst = atf.src,atf.dst
    virtual_path = get_path(atf)
    u_all = 0
    for idx in range(len(virtual_path)):
        #phy_path = getPathFromsrctodst(P,source,virtual_path[idx][1]) 
        phy_path = get_path(atf,route_type="phy")
        virtual_path_slice = virtual_path[0:idx+1]
        C = getLenCommonLinks(phy_path,virtual_path_slice)
        if C > len(phy_path) or C > len(virtual_path_slice):
            print("C is error,C={},phy_len={},and Virtual len={}".format(C,len(phy_path),len(virtual_path_slice)))
        u_n = 0.5 * ( C/len(phy_path) + C/len(virtual_path_slice))
        #u_n = ( C/len(phy_path))
        u_all += u_n
    uti = u_all / len(virtual_path)

    return uti

def _atf_set_init(topo):
    _atf_set = NHAttackFlows(topo)
    #print(_atf_set)
    atf_set = []
    for att_flow in _atf_set:
        #print(att_flow)
        tmp = atf(src=att_flow[0],dst=att_flow[1])
        atf_set.append(tmp)
    for atf_flow in atf_set:
        atf_flow.route = nx.shortest_path(topo,source=atf_flow.src,target=atf_flow.dst)
        # atf_flow.delay = get_path_delay(atf_flow.route,topo)
        atf_flow.weight = 0
    
    for atf_flow in atf_set:
        atf_flow.Graph = topo
        atf_flow._route_edge = get_edge_from_list(atf_flow.route)
    return atf_set

def get_edge_from_list(l):
    list_of_edge = []
    for idx in range(len(l)-1):
        e_tmp = edg(l[idx],l[idx+1])
        list_of_edge.append(e_tmp)
    return list_of_edge

def inc_fd_in_edg_list(edg,atf,edg_list):
    for e in edg_list:
        if e == edg:
            e.atf_set.append(atf)
            e.fd +=1
    return edg_list

def link_map_crossfire(atf_set):
    edg_list = []
    for atf in atf_set:
        edges = get_edge_from_list(atf.route)
        for e in edges:
            if e not in edg_list:
                edg_list.append(e)
                inc_fd_in_edg_list(e,atf,edg_list)
            else:
                inc_fd_in_edg_list(e,atf,edg_list)
    return edg_list

def node_map_attacker(atf_set,typeofroute):
    edg_list = []
    for atf in atf_set:
        if typeofroute == 'sp':
            edges = get_edge_from_list(atf.route_sp)
        elif typeofroute == 'nh':
            edges = get_edge_from_list(atf.route_nh)
        elif typeofroute == 'eq':
            edges = get_edge_from_list(atf.route_eq)
        else:
            edges = get_edge_from_list(atf.route)
        for e in edges:
            if e not in edg_list:
                edg_list.append(e)
                inc_fd_in_edg_list(e,atf,edg_list)
            else:
                inc_fd_in_edg_list(e,atf,edg_list)
    return edg_list

def link_map_attacker(atf_set,typeofroute):
    edg_list = []
    for atf in atf_set:
        if typeofroute == 'sp':
            edges = get_edge_from_list(atf.route_sp)
        elif typeofroute == 'nh':
            edges = get_edge_from_list(atf.route_nh)
        elif typeofroute == 'eq':
            edges = get_edge_from_list(atf.route_eq)
        else:
            edges = get_edge_from_list(atf.route)
        for e in edges:
            if e not in edg_list:
                edg_list.append(e)
                inc_fd_in_edg_list(e,atf,edg_list)
            else:
                inc_fd_in_edg_list(e,atf,edg_list)
    return edg_list

def get_topo(topo_name):
    t = nx.read_gml(topo_name,label="id")
    return t

def spider_init(topo_name):
    topo = get_topo(topo_name)
    atf_set = atf_set_init(topo)
    e_list = link_map_crossfire(atf_set)
    e_list.sort(reverse=True)
    return e_list,atf_set

def add_spider_link(pr,a,spiderNode):
    random_number = random.random()
    if random_number < pr:
        if len(a.route_sp) == 0:
            random_idx = random.randint(0,len(a.route)-1)
            a.route_sp = [r for r in a.route]
            a.route_sp.insert(random_idx,spiderNode)
        else:
            while True:
                random_idx = random.randint(0,len(a.route_sp)-1)
                if 's' not in str(a.route_sp[random_idx]):
                    break
            a.route_sp.insert(random_idx,spiderNode)
    else:
        return 
    return 

def force_add_spider_link(a,spiderNode):
    
    if len(a.route_sp) == 0:
        random_idx = random.randint(0,len(a.route)-1)
        a.route_sp = [r for r in a.route]
        a.route_sp.insert(random_idx,spiderNode)
    else:
        while True:
            random_idx = random.randint(0,len(a.route_sp)-1)
            if 's' not in str(a.route_sp[random_idx]):
                break
        a.route_sp.insert(random_idx,spiderNode)

def write_log(e):
    with open("optlog",'a') as a:
        a.write(e+'\n')
        a.close()
        
def nethide_a_topo(topo,c = 0):
    g = nx.read_gml(topo,label='id')
    # init_delays(g)
    atf_set = atf_set_init(g)
    max_fd,avg_fd = FlowDensityCalculate(g)
    if c == 0:
        c = max_fd * LINKCAPACITY_PERCENTAGE
    print(f"start for topo {topo[0:-4]}, nodes {len(g.nodes)}, edges {len(g.edges)}")
    try:
        return_trees,log_values_line,V = gurobi_opt(g,c)
        return return_trees,log_values_line,V,atf_set
    except Exception as e:
        write_log(str(e))

def gen_random_number(a,b,n):
    if a >= b:
        logging.warning(f"[Random] error input b:{b}, a:{a}")
        return
    if n>= (b - a):
        logging.warning(f"[Random] error sample b:{b},a:{a},n:{n}")
    return random.sample(range(a, b), n)

def spider_gen(atf_set,e_list,num):
    max_fd = len(atf_set)
    probability = []
    fd_list = []
    for i in range(num):
        pr = e_list[i].fd / max_fd
        probability.append(pr)
    
    for idx,p in enumerate(probability):
        n = int(p * max_fd)
        r_list = gen_random_number(n,0,max_fd)
        print(len(r_list))
        for r in r_list:
            force_add_spider_link(atf_set[r],'s'+str(idx))
        #         for a in atf_set:
#             for idx,p in enumerate(probability):
#                 add_spider_link(p,a,'s'+str(idx))
#         return atf_set
    return atf_set

def process_element(a):
    if not a.route_sp:
        a.route_sp = [x for x in a.route]
    return a

def add_spider_node(spider_node,e,atf):
    logging.debug(f"[SpiderNode] Add virtual node {(spider_node)} to atf {(atf.src,atf.dst)}")
    src,dst = e.src,e.dst
    
    if src not in atf.route or dst not in atf.route:
        logging.warning(f"[SpiderAdd] Link {(src,dst)} not in the atf's route.")
        return -1 
    
    if atf.route_sp != []:
        idx_src = atf.route_sp.index(src)
        idx_dst = atf.route_sp.index(dst)
    else:
        idx_src = atf.route.index(src)
        idx_dst = atf.route.index(dst)
    
    insert_idx = min(idx_src,idx_dst)
    
#     if idx_dst - idx_src != 1:
#         logging.fatal(f"[SpiderAdd] Link {(src,dst)} are not consequtive in the atf route")
#         return -1
    if atf.route_sp == []:
        atf.route_sp = [i for i in atf.route]
    atf.route_sp.insert(insert_idx+1,spider_node)
    
    return 

def add_spider_link_new(spider_node,e,a):
    
    logging.info(f"[SpiderLink] Add virtual link {(e.src,spider_node)} to atf {(a.src,a.dst)}")
    if len(a.route_sp) == 0:
        random_idx = random.randint(1,len(a.route)-1)
        a.route_sp = [r for r in a.route]
        a.route_sp.insert(random_idx,e.src)
        a.route_sp.insert(random_idx+1,spider_node)
    else:
        while True:
            random_idx = random.randint(1,len(a.route_sp)-1)
            if 's' not in str(a.route_sp[random_idx]):
                break
        a.route_sp.insert(random_idx,e.src)
        a.route_sp.insert(random_idx+1,spider_node)
    return 

def divide_atf_set(atf_set,e_list,num,link_capacity):
    
    ob_target = e_list[0:num]
    safe_link = e_list[num:]
    
    # Divide attack_flows into three set.
    non_threat_atf_idx = [] 
    for idx,a in enumerate(atf_set):
        edg_list = get_edge_from_list(a.route)
        for e in edg_list:
            if e not in ob_target:
                non_threat_atf_idx.append(idx)
    
    return non_threat_atf_idx

def step_1_calculate_sp_numbers(e,link_cp):
    num = 0
    tmp = e.fd
    while tmp >= link_cp:
       tmp = int(tmp / 2)
       num += 1
    return num   

def split_list(input_list, v_nodes):
    # 计算每份的长度
    avg_length = len(input_list) // v_nodes
    remainder = len(input_list) % v_nodes  # 处理余数

    result = []
    start = 0

    for i in range(v_nodes):
        # 计算当前份的结束索引
        end = start + avg_length + (1 if i < remainder else 0)
        result.append(input_list[start:end])
        start = end

    return result

def find_new_indices(ListA, ListB, index_sequence):
    # 创建一个字典，将ListB中的元素映射到它们在ListA中的索引
    index_mapping = {element: i for i, element in enumerate(ListA)}

    # 使用字典映射来查找ListB中元素的新索引
    new_indices = [index_mapping[ListB[i]] for i in index_sequence]

    return new_indices

def spider_net_gen_link(atf_set,e_list,num,link_capacity):
    
    ob_target = e_list[0:num]

    safe_link = e_list[num:]

    reduce_target = [e for e in safe_link if e.fd > 0.5*link_capacity]
    #print(reduce_target)
    for idxe,e in enumerate(ob_target):
        logging.info(f"[SpiderGen] Gen virtual link for {(e.src,e.dst)}")

        if e.fd < link_capacity:
            reduce_target.append(e)
            logging.info(f"[SpiderGen] Link {(e.src,e.dst)} has fd {e.fd} capacity {link_capacity}, continue.")
            continue
        
        # STEP 1: Hide 50% target link to avoid flooding. 
        logging.info(f"[SpiderGen] Gen virtual link for {(e.src,e.dst)} step 1")
        
        #number_of_virtual_nodes = step_1_calculate_sp_numbers(e,link_capacity)
        # print(e.fd,link_capacity,number_of_virtual_nodes)
    
        #one_piece = int(e.fd / (2**number_of_virtual_nodes))
        # one_piece = int(0.5*link_capacity) + random.randint(int(-0.02*link_capacity),int(0.02*link_capacity))
        one_piece = int(0.5*link_capacity) 
        number_of_virtual_nodes = - (- e.fd // one_piece)
        random_index = gen_random_number(0, len(e.atf_set), int(e.fd- one_piece))
        
        new_index = find_new_indices(atf_set,e.atf_set,random_index)
        # v_nodes = 2**number_of_virtual_nodes - 1 
        v_nodes = number_of_virtual_nodes
        v_nodes_atf_index = split_list(new_index,v_nodes)
 
        #print(f"STEP 1: ( {e.src}, {e.dst}), {e.fd}, {one_piece}, num {number_of_virtual_nodes} atf {len(random_index)}")
        for idxtmp,v in enumerate(v_nodes_atf_index):
            for r in v:
                emm = add_spider_node('s'+str(idxe)+str(idxtmp),e,atf_set[r])
        
        v_link_fd = one_piece
        
        logging.info(f"[SpiderGen] Gen virtual link for {(e.src,e.dst)} step 2")
        
        # STEP 2: Add fd to virtual link to manipulate deceptive targets.
        candidate_atf = [] 

        noncandidate_atf_idx = divide_atf_set(atf_set,e_list,num,link_capacity)
        #print("+++++step2 divide!!!",len(noncandidate_atf_idx))
        candidate_atf = []
        noncandidate_atf = []
        for a_idx in noncandidate_atf_idx:
            if e.src in atf_set[a_idx].route:
                if e.src == atf_set[a_idx].route[0] or e.src == atf_set[a_idx].route[-1]:
                    continue
                if 's'+str(idxe) in atf_set[a_idx].route_sp:
                    continue
                candidate_atf.append(atf_set[a_idx])
            else:
                noncandidate_atf.append(atf_set[a_idx])
       
#         noncandidate_atf = []
#         for se in safe_link:
#             if se.src == e.src:       
#                 for a in se.atf_set:
#                     candidate_atf.append(a)
#             else:
#                 for a in se.atf_set:
#                     noncandidate_atf.append(a)
        
        if candidate_atf != []:
            candidate_atf = list(set(candidate_atf))
            
            if v_link_fd + len(candidate_atf) >= int(e.fd * 0.8):
                
                random_length = int(0.25 * e.fd) + 1 
                
            else: 
                random_length = len(candidate_atf)
            logging.debug(f"[DEBUG]lengths of candidate_atf {len(candidate_atf)}" + 
                          f" e.fd {e.fd} v_link_fd {v_link_fd} random_length {random_length}")
            if random_length > len(candidate_atf):
                random_index = range(0,len(candidate_atf))
            else:
                random_index = gen_random_number(0,len(candidate_atf),random_length)
            
            #print(f"STEP 2: ( {e.src}, {e.dst}), candidate atf {len(candidate_atf)} radom number {len(random_index)}")
            new_index = find_new_indices(atf_set,candidate_atf,random_index)
            logging.info(f"[SpiderGen-STEP2] add {len(new_index)} number of virtual nodes to s{str(idxe)}0")
            for r in new_index:
                add_spider_node('s'+str(idxe)+'0',e,atf_set[r])
        
            v_link_fd += len(random_index)
        
        logging.info(f"[SpiderGen] Gen virtual link for {(e.src,e.dst)} step 3")
        
        # STEP 3: Random add virtual link to others as fingerprints.
        noncandidate_atf_filter = []
        for a in noncandidate_atf:
            if 's'+str(idxe) not in a.route_sp and e.src not in a.route_sp:
                noncandidate_atf_filter.append(a)
        noncandidate_atf_filter = list(set(noncandidate_atf_filter))
        random_length = int(e.fd * STEP3_SHRINK_FACTOR) - v_link_fd 
        #print(f"STEP 3: ( {e.src}, {e.dst}), non candidate atf {len(noncandidate_atf_filter)} radom number {random_length}")
        if random_length > 0:
            if len(noncandidate_atf_filter) < random_length:
                random_index = range(0,len(noncandidate_atf_filter))
            else:
                random_index = gen_random_number(0,len(noncandidate_atf_filter),random_length)
            new_index = find_new_indices(atf_set,noncandidate_atf_filter,random_index)
            logging.info(f"[SpiderGen-STEP3] add {len(new_index)} number of virtual nodes to s{str(idxe)}0")
            for r in new_index:
                add_spider_link_new('s'+str(idxe)+'0',e,atf_set[r])
    
    for idxe,e in enumerate(reduce_target):
        logging.info(f"[SpiderGen] Gen virtual link for ruduce target {(e.src,e.dst)}")
        #print(f"STEP 4: Gen virtual link for ruduce target {(e.src,e.dst)}")
        
        #one_piece = int(e.fd / (2**number_of_virtual_nodes))

        #one_piece = int(0.5*link_capacity) + random.randint(int(-0.02*link_capacity),int(0.02*link_capacity))
        one_piece = int(0.5*link_capacity)
        if e.fd <=one_piece: 
            continue
        number_of_virtual_nodes = - (- e.fd // one_piece)
        #print(f"random_index debug: len(atf_set) {len(e.atf_set)}, fd {e.fd}, one_piece {one_piece}" )
        random_index = gen_random_number(0, len(e.atf_set), int(e.fd- one_piece))
        new_index = find_new_indices(atf_set,e.atf_set,random_index)
        # v_nodes = 2**number_of_virtual_nodes - 1 
        v_nodes = number_of_virtual_nodes
        v_nodes_atf_index = split_list(new_index,v_nodes)
 
        #print(f"---, ( {e.src}, {e.dst}), {e.fd}, {one_piece}, num {number_of_virtual_nodes} atf {len(random_index)}")
        for idxtmp,v in enumerate(v_nodes_atf_index):
            for r in v:
                emm = add_spider_node('s'+str(idxe)+str(idxtmp)+'v',e,atf_set[r])
        
    # STEP 4: keep Unchanged route (for debug)
    #print(f"STEP 5: keep unchanged route")
    


    num_processes = multiprocessing.cpu_count()  # 使用CPU核心数量
    #print(f"CPUS {num_processes}")
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            atf_set = pool.map(process_element, atf_set)
    except (KeyboardInterrupt,AssertionError) as e:
        print("Interrupted")
        for a in atf_set:
            if not a.route_sp:
                a.route_sp = [x for x in a.route]
    

    #print(f"STEP 5 finished")
    
    return atf_set

def spider_net_deployment():
    return

def check_atf_route(a):  
    (src,dst) = a.src,a.dst  
    # Rule 1: Must keep src and dst.
    if a.route_sp[0] != src or a.route_sp[-1] != dst:
        logging.error(f"[RuleCheck] wrong atf {(src,dst)} sp route: {a.route_sp}")
    
    # Rule 2: Cannot have any cycle.
    if len(set(a.route_sp)) != len(a.route_sp):
        logging.error(f"[RuleCheck] wrong elements {(src,dst)} sp route: {a.route_sp}")
    return         

def get_spidernum(link_cp_list,link_cp):
    count = 0
    for item in link_cp_list:
        if item >= link_cp:
            count += 1
    return int(count * SPIDER_NUM_FACTOR)+1   

def SpiderNet(topo_name,link_capacity):
    e_list,atf_set = spider_init(topo_name)
    link_cp_list = topo_to_link(topo_name)
    spidernum = get_spidernum(link_cp_list,link_capacity)
    atf_sp_set = spider_net_gen_link(atf_set,e_list,spidernum,link_capacity)
    return atf_sp_set

def fingerprint_deployment(topo_name,flow_set, M=SIZE_OF_TCAM,edge_id= None):
    # 读取拓扑
    G  = init_topo()
    # 创建Gurobi模型
    m = gp.Model() 
    m.setParam("OutputFlag",1)
    m.setParam("LogToConsole", False)
    m.setParam("LogFile", topo_name[:-4]+'-spidernet-gurobilog.txt')

    # 创建二进制变量yi，一个变量对应一个节点
    y = {}
    for node in G.nodes():
        y[node] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}")

    # 创建二进制变量x_ij，一个变量对应一条流i和一个节点j
    x = {}
    for flow in flow_set:
        for node in G.nodes():
            x[(flow, node)] = m.addVar(vtype=GRB.BINARY, name=f"x_{flow.id}_{node}")

    # 更新模型变量
    m.update() 
    m.Params.MIPGap = 0.02
    # 设置目标函数，最小化所有yi的和
    m.setObjective(gp.quicksum(y[node] for node in G.nodes()), GRB.MINIMIZE)

    # 添加流监听需求的约束
    for flow in flow_set:
        m.addConstr(gp.quicksum(x[(flow, node)] for node in G.nodes()) == 1, f"flow_constraint_{flow.id}")

    # 添加流路径约束，使得流只能被其路径上的节点所监听
    for flow in flow_set:
        for node in G.nodes():
            if node in flow.route:
                m.addConstr(x[(flow, node)] <= 1, f"path_constraint_{flow.id}_{node}")
            else:
                m.addConstr(x[(flow, node)] == 0, f"path_constraint_{flow.id}_{node}")

    # 添加节点监听流数量的约束
    for node in G.nodes():
        m.addConstr(gp.quicksum(x[(flow, node)] for flow in flow_set) <= M * y[node], f"node_constraint_{node}")

    # 优化模型
    m.optimize()

    # 检查优化结果
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found")
        # 输出每个节点的选择情况
        # for node in G.nodes():
        #     print(f"Node {node}: y_{node} = {y[node].x}")
        # 输出流监听情况
        for flow in flow_set:
            for node in G.nodes():
                if x[(flow, node)].x == 1:
                    if edge_id == None:
                        print(f"Flow {(flow.src,flow.dst)} is listened by Node {node}")
                    else:
                        flow.monitored_node[edge_id] = node
        return 1
    else:
        print("No optimal solution found")
        return -1

