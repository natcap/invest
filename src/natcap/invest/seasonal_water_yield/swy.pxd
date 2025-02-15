from libcpp.vector cimport vector

cdef extern from "swy.h":
    void run_calculate_local_recharge[T](
        vector[char*], # precip_path_list
        vector[char*], # et0_path_list
        vector[char*], # qf_m_path_list
        char*, # flow_dir_mfd_path
        vector[char*], # kc_path_list
        vector[float], # alpha_values
        float, # beta_i
        float, # gamma
        char*, # stream_path
        char*, # target_li_path
        char*, # target_li_avail_path
        char*, # target_l_sum_avail_path
        char*, # target_aet_path
        char* # target_pi_path
    ) except +

    void run_route_baseflow_sum[T](
        char*,
        char*,
        char*,
        char*,
        char*,
        char*,
        char*) except +
