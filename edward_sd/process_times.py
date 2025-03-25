def sum_inside_residual_block(file_path):
    total_gn_time = 0.0
    total_silu_time = 0.0
    total_conv3_time = 0.0
    total_conv1_time = 0.0
    total_add_time = 0.0
    total_FC_time = 0.0
    with open(file_path, 'r') as file:
        for line in file: # read line by line of the file
            if any(x in line for x in ['residual_block_gn1_t', 'residual_block_gn2_t']):
                time_str = line.split(':')[-1].strip()
                # print(time_str)
                total_gn_time += float(time_str)
            elif any(x in line for x in ['residual_block_silu1_t', 'residual_block_silu2_t', 'residual_block_silu3_t']):
                time_str = line.split(':')[-1].strip()
                total_silu_time += float(time_str)
            elif any(x in line for x in ['residual_block_firstconv3_t', 'residual_block_secondconv3_t']):
                time_str = line.split(':')[-1].strip()
                total_conv3_time += float(time_str)
            elif 'residual_block_conv1_t' in line:
                time_str = line.split(':')[-1].strip()
                total_conv1_time += float(time_str)
            elif any(x in line for x in ['residual_block_add1_t', 'residual_block_add2_t']):
                time_str = line.split(':')[-1].strip()
                total_add_time += float(time_str)
            elif 'residual_block_FC_t' in line:
                time_str = line.split(':')[-1].strip()
                total_FC_time += float(time_str)
    return total_gn_time, total_silu_time, total_conv3_time, total_conv1_time, total_add_time, total_FC_time


def sum_inside_transfo_block(file_path):
    total_gn_time = 0.0
    total_conv1_time = 0.0
    total_ln_time = 0.0
    total_static_mm_time = 0.0
    total_dynamic_mm_time = 0.0
    total_softmax_time = 0.0
    total_add_time = 0.0
    total_GEGLU_time = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            if 'transfo_block_gn1_t' in line:
                time_str = line.split(':')[-1].strip()
                total_gn_time += float(time_str)
            elif any(x in line for x in ['transfo_block_firstconv1_t', 'transfo_block_secondconv1_t']):
                time_str = line.split(':')[-1].strip()
                total_conv1_time += float(time_str)
            elif any(x in line for x in ['transfo_block_ln1_t', 'transfo_block_ln2_t', 'transfo_block_ln3_t']):
                time_str = line.split(':')[-1].strip()
                total_ln_time += float(time_str)
            elif any(x in line for x in ['selfattn_block_static_mms1_t', 'selfattn_block_static_mm2_t', 
                                         'crossattn_block_static_mms1_t', 'crossattn_block_static_mm2_t', 
                                         'transfo_block_MM1_geglu_t', 'transfo_block_MM2_geglu_t']):
                time_str = line.split(':')[-1].strip()
                total_static_mm_time += float(time_str)
            elif any(x in line for x in ['selfattn_block_dynamic_mm1_t', 'selfattn_block_dynamic_mm2_t', 
                                         'crossattn_block_dynamic_mm1_t', 'crossattn_block_dynamic_mm2_t']):
                time_str = line.split(':')[-1].strip()
                total_dynamic_mm_time += float(time_str)
            elif any(x in line for x in ['selfattn_block_softmax_t', 'crossattn_block_softmax_t']):
                time_str = line.split(':')[-1].strip()
                total_softmax_time += float(time_str)
            elif any(x in line for x in ['transfo_block_add1_t', 'transfo_block_add2_t', 'transfo_block_add3_t', 'transfo_block_add4_t']):
                time_str = line.split(':')[-1].strip()
                total_add_time += float(time_str)
            elif 'transfo_block_gelu_t' in line:
                time_str = line.split(':')[-1].strip()
                total_GEGLU_time += float(time_str)
    return total_gn_time, total_conv1_time, total_ln_time, total_static_mm_time, total_dynamic_mm_time, total_softmax_time, total_add_time, total_GEGLU_time

def sum_upsample_block_times(file_path):
    total_conv3_time = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            if 'upsample_block_conv3_t' in line:
                time_str = line.split(':')[-1].strip()
                total_conv3_time += float(time_str)
    return total_conv3_time

def sum_attention_block_times(file_path):
    total_time = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            if 'attention_block_t' in line:
                time_str = line.split(':')[-1].strip()
                total_time += float(time_str)
    return total_time

def sum_residual_block_times(file_path):
    total_time = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            if 'residual_block_t' in line:
                time_str = line.split(':')[-1].strip()
                total_time += float(time_str)
    return total_time

def print_CLIPtime_UNETtime_DECODERtime_TOTALtime(file_path):
    CLIP_time = 0.0
    UNET_time = 0.0
    Decoder_time = 0.0
    total_time = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            if 'clip_t' in line:
                time_str_minutes = line.split(':')[-2].strip()
                time_str_seconds = line.split(':')[-1].strip()
                CLIP_time += 60*float(time_str_minutes) + float(time_str_seconds)

            if 'UNET_total_t' in line:
                time_str_minutes = line.split(':')[-2].strip()
                time_str_seconds = line.split(':')[-1].strip()
                UNET_time += 60*float(time_str_minutes) + float(time_str_seconds)
            
            elif 'decoder_t' in line:
                time_str_minutes = line.split(':')[-2].strip()
                time_str_seconds = line.split(':')[-1].strip()
                Decoder_time += 60*float(time_str_minutes) + float(time_str_seconds)
            
            elif 'TOTAL_T' in line:
                time_str_minutes = line.split(':')[-2].strip()
                time_str_seconds = line.split(':')[-1].strip()
                total_time += 60*float(time_str_minutes) + float(time_str_seconds)
    
    return CLIP_time, UNET_time, Decoder_time, total_time

if __name__ == "__main__":
    file_path = 'time_measurements2.txt'

    total_gn_time, total_silu_time, total_conv3_time, total_conv1_time, total_add_time, total_FC_time = sum_inside_residual_block(file_path)
    print(f"Residual total_gn_time: {total_gn_time}")
    print(f"Residual total_silu_time: {total_silu_time}")
    print(f"Residual total_conv3_time: {total_conv3_time}")
    print(f"Residual total_conv1_time: {total_conv1_time}")
    print(f"Residual total_add_time: {total_add_time}")
    print(f"Residual total_FC_time: {total_FC_time}")
    print("############################################")

    total_gn_time, total_conv1_time, total_ln_time, total_static_mm_time, total_dynamic_mm_time, total_softmax_time, total_add_time, total_GEGLU_time = sum_inside_transfo_block(file_path)
    print(f"Transfo total_gn_time: {total_gn_time}")
    print(f"Transfo total_conv1_time: {total_conv1_time}")
    print(f"Transfo total_ln_time: {total_ln_time}")
    print(f"Transfo total_static_mm_time: {total_static_mm_time}")
    print(f"Transfo total_dynamic_mm_time: {total_dynamic_mm_time}")
    print(f"Transfo total_softmax_time: {total_softmax_time}")
    print(f"Transfo total_add_time: {total_add_time}")
    print(f"Transfo total_GEGLU_time: {total_GEGLU_time}")
    print("############################################")

    conv3_upsampling_blocks = sum_upsample_block_times(file_path)
    print(f"Total upsampling_blocks_conv3_time: {conv3_upsampling_blocks}")

    total_attn_time = sum_attention_block_times(file_path)
    print(f"Total attention_block_times: {total_attn_time}")
    total_residual_time = sum_residual_block_times(file_path)
    print(f"Total residual_block_times: {total_residual_time}")
    print("############################################")


    CLIP_time, UNET_time, Decoder_time, total_time = print_CLIPtime_UNETtime_DECODERtime_TOTALtime(file_path)
    print("CLIP_time:", CLIP_time)
    print("UNET_time:", UNET_time)
    print("Decoder_time:", Decoder_time)
    print("Total_time:", total_time)