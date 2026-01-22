
import datetime
import os


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def random_assign_to_avoid_collision(all_indices_dict, num_emb_list):
    unique_set = set()
    code_length = len(num_emb_list)

    new_indices_dict = dict()
    invalid_idx_lists = []
    for idx, codes in all_indices_dict.items():
        stage = code_length - 1
        code_str = "_".join(codes)
        if code_str in unique_set:
            invalid_idx_lists.append(idx)
        unique_set.add(code_str)
        new_indices_dict[idx] = codes

    for idx in invalid_idx_lists:
        codes = all_indices_dict[idx]
        count = 0
        preserve = codes[stage]
        code_str = "_".join(codes)
        while code_str in unique_set:
            if count >= num_emb_list[stage]:
                codes[stage] = preserve
                count = 0
                stage -= 1
                preserve = codes[stage]
            codes[stage] = codes[stage].split("_")[0] + "_" + str(count) + ">"
            code_str = "_".join(codes)
            count += 1
        print(f"Idx: {idx}, Codes: {codes}")
        unique_set.add(code_str)
        new_indices_dict[idx] = codes
    return new_indices_dict

