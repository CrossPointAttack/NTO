from Nethide.NethideOptimizer import *

from Util import common
from Util import config


def nethide_a_topo(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo(topo_name=topo_name)
    atf_nh_set = common.atf_set_init(g)
    _nethide_dict,nethide_log_values, nethide_v = gurobi_opt(g,link_capacity)
    forwrding_trees = _nethide_dict
    for a in atf_nh_set:
        a.route_nh = forwrding_trees[a.dst][a.src]
    edg_list_nh = common.link_map(atf_nh_set,'nh')
    return atf_nh_set,edg_list_nh

def nethide_a_topo_fast(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo(topo_name=topo_name)
    atf_nh_set = common.atf_set_init(g)
    _nethide_dict,nethide_log_values, nethide_v = gurobi_opt(g,link_capacity)
    forwrding_trees = _nethide_dict
    for a in atf_nh_set:
        a.route_nh = forwrding_trees[a.dst][a.src]
    #edg_list_nh = common.link_map_fast(atf_nh_set,'nh')
    return atf_nh_set