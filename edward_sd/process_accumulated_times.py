"""Residual total_gn_time: 1.3835090000000048
Residual total_silu_time: 1.4853110000000123
Residual total_conv3_time: 57.99201399999995
Residual total_conv1_time: 3.948386000000002
Residual total_add_time: 0.4413209999999997
Residual total_FC_time: 0.2984339999999995
Transfo total_gn_time: 0.17931100000000008
Transfo total_conv1_time: 5.184271999999997
Transfo total_ln_time: 0.8870179999999986
Transfo total_static_mm_time: 27.687703000000084
Transfo total_dynamic_mm_time: 33.846508000000014
Transfo total_softmax_time: 58.0471600000001
Transfo total_add_time: 0.8842659999999998
Transfo total_GEGLU_time: 4.503837999999998
Total upsampling_blocks_conv3_time: 7.355182000000003
Total attention_block_times: 142.41630399999994
Total residual_block_times: 66.20441300000007
CLIP_time: 0.134198
UNET_time: 220.822718
Decoder_time: 7.078588
Total_time: 228.03999199999998"""

import re

# convert the above string to a dictionary
time_str = """Residual total_gn_time: 1.3835090000000048
Residual total_silu_time: 1.4853110000000123
Residual total_conv3_time: 57.99201399999995
Residual total_conv1_time: 3.948386000000002
Residual total_add_time: 0.4413209999999997
Residual total_FC_time: 0.2984339999999995
Transfo total_gn_time: 0.17931100000000008
Transfo total_conv1_time: 5.184271999999997
Transfo total_ln_time: 0.8870179999999986
Transfo total_static_mm_time: 27.687703000000084
Transfo total_dynamic_mm_time: 33.846508000000014
Transfo total_softmax_time: 58.0471600000001
Transfo total_add_time: 0.8842659999999998
Transfo total_GEGLU_time: 4.503837999999998
Total upsampling_blocks_conv3_time: 7.355182000000003
Total attention_block_times: 142.41630399999994
Total residual_block_times: 66.20441300000007
CLIP_time: 0.134198
UNET_time: 220.822718
Decoder_time: 7.078588
Total_time: 228.03999199999998"""

time_dict = dict(re.findall(r'(\S+): ([\d.]+)', time_str))
time_dict = {k: float(v) for k, v in time_dict.items()}
# print(time_dict)
for k, v in time_dict.items():
    print(f"{k}: {v}")
