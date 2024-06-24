from SpiderNet.SpiderNetSolver import *
from Util import common
from Util import config

def spidernet_a_topo(link_capacity,topo_name=config.TOPO_NAME,omit_flow=True):
    topo = config.TOPO_PATH + topo_name
    atf_sp_set = SpiderNet(topo,link_capacity)
    start = time.time()
    #edg_list_sp = common.link_map_fast(atf_sp_set,'sp')
    if omit_flow:

        edg_list_sp = common.link_map_fast(atf_sp_set,'sp')
    else:
        edg_list_sp = common.link_map(atf_sp_set,'sp')
    end = time.time()
    print(f"[SpiderNet] costs {end - start} seconds.")
    return atf_sp_set,edg_list_sp

def spidernet_a_topo_fast(link_capacity,topo_name=config.TOPO_NAME):
    start = time.time()
    topo = config.TOPO_PATH + topo_name
    atf_sp_set = SpiderNet(topo,link_capacity)
    end = time.time()
    print(f"[SpiderNet] costs {end - start} seconds.")
    return atf_sp_set

def physical_link_map(link_capacity,topo_name=config.TOPO_NAME):
    g = common.init_topo()
    atf_set = atf_set_init(g)
    edg_list = common.link_map(atf_set,'ph')
    edg_list.sort(reverse=True)

    return atf_set,edg_list

def spidernet_full(link_capacity,topo_name=config.TOPO_NAME):
    start = time.time()
    atf_sp_set,edg_list_sp = spidernet_a_topo(link_capacity,topo_name,omit_flow=False)

    e_monitor = [e for e in edg_list_sp if e.fd > link_capacity]
    print(f"[SpiderNet] deployment creates {len(e_monitor)} honeypot links.")
    
    for idx,emm in enumerate(e_monitor):
        emm.id = idx
        result = fingerprint_deployment(topo_name,emm.atf_set,edge_id=emm.id)
        if result != 1:
            print(f"error in monitor {(emm.src,emm.dst)}")
    end = time.time()
    
    print(f"[SpiderNet] costs {end - start} seconds")

    return atf_sp_set,e_monitor