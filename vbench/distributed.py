import os
import torch

import torch.distributed


# ------------------------------------------------------- #
#                        distributed                      #
# ------------------------------------------------------- #
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def dist_init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    data_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

# ------------------------------------------------------- #


def merge_list_of_list(results):
    results = [item for sublist in results for item in sublist]
    return results


def gather_list_of_dict(results):
    results = all_gather(results)
    results = merge_list_of_list(results)
    return results


def distribute_list_to_rank(data_list):
    data_list = data_list[get_rank()::get_world_size()]
    return data_list
