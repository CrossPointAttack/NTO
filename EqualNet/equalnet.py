from EqualNet.EqualNetSolver import *
from Util import common
from Util import config

def equalnet_a_topo(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo(topo_name=topo_name)
    atf_eq_set = EqualNet_a_topo(g,config.ip_addr_dataset)
    edg_list_eq = common.link_map(atf_eq_set,'eq')
    return atf_eq_set,edg_list_eq

def equalnet_a_topo_fast(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo(topo_name=topo_name)
    atf_eq_set = EqualNet_a_topo(g,config.ip_addr_dataset)
    return atf_eq_set

def physical_link_map(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo(topo_name=topo_name)
    atf_eq_set = EqualNet_a_topo(g,config.ip_addr_dataset)
    edg_list_p = common.link_map(atf_eq_set,'eqph')
    edg_list_p.sort(reverse=True)
    edg_list_v = common.link_map(atf_eq_set,'eq')
    edg_list_v.sort(reverse=True)
    return atf_eq_set,edg_list_p,edg_list_v