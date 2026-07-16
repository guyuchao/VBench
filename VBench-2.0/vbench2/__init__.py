import os
import time
from numbers import Real

import importlib
from pathlib import Path

from .distributed import barrier, distribute_list_to_rank, get_rank, print0
from .utils import get_prompt_from_filename, init_submodules, save_json, load_json


def _load_dimension_result(path, dimension):
    try:
        result = load_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or dimension not in result:
        return None
    payload = result[dimension]
    if not isinstance(payload, list) or len(payload) != 2:
        return None
    if not isinstance(payload[0], Real):
        return None
    return payload


def _save_json_atomic(data, path):
    temporary_path = f'{path}.rank{get_rank()}.tmp'
    save_json(data, temporary_path)
    os.replace(temporary_path, path)


def _dimension_result_path(output_path, name, dimension):
    return os.path.join(output_path, f'{name}_{dimension}_eval_results.json')


def _dimension_error_path(output_path, name, dimension):
    return os.path.join(output_path, f'{name}_{dimension}_eval_error.json')


class VBench2(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return ["Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Composition", "Dynamic_Spatial_Relationship", 
                "Dynamic_Attribute", "Motion_Order_Understanding", "Human_Interaction", "Complex_Landscape", 'Complex_Plot', "Camera_Motion", 
                "Motion_Rationality", "Instance_Preservation", "Mechanics", "Thermotics", "Material", "Multi-View_Consistency"]        

    def check_dimension_requires_extra_info(self, dimension_list):
        dim_custom_not_supported = set(dimension_list) & set([
            'Composition', 'Dynamic_Attribute', 'Dynamic_Spatial_Relationship', 'Instance_Preservation', 'Complex_Plot', 'Complex_Landscape', 
            'Motion_Rationality', 'Motion_Order_Understanding', 'Mechanics', 'Thermotics', 'Material', "Camera_Motion", "Human_Interaction"
        ])

        assert len(dim_custom_not_supported) == 0, f"dimensions : {dim_custom_not_supported} not supported for custom input"

    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, mode='vbench_standard', **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions
        if mode=='custom_input':
            self.check_dimension_requires_extra_info(dimension_list)
            video_names = os.listdir(videos_path)
            assert len(video_names)>0, f"ERROR : The video files is empty"
            cur_full_info_list = []
            prompt_check_list = []
            for filename in video_names:
                postfix = Path(os.path.join(videos_path, filename)).suffix
                if postfix.lower() not in ['.mp4']:
                    continue
                if dimension_list[0]=='Diversity':
                    prompt_en = get_prompt_from_filename(filename)
                    if prompt_en in prompt_check_list:
                        continue
                    prompt_check_list.append(prompt_en)
                    item = {
                        "prompt_en": prompt_en, 
                        "dimension": dimension_list, 
                        "video_list": []
                    }
                    for ite in range(20):
                        item['video_list'].append(os.path.join(videos_path, f'{prompt_en}{special_str}-{str(ite)}{postfix}'))
                    cur_full_info_list.append(item)
                else:
                    cur_full_info_list.append({
                        "prompt_en": get_prompt_from_filename(filename), 
                        "dimension": dimension_list, 
                        "video_list": [os.path.join(videos_path, filename)]
                    })

        else:
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            
            postfix = Path(video_names[0]).suffix
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]): 
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    if 'Diversity' in prompt_dict["dimension"]:
                        num=20
                    else:
                        num=3
                    for i in range(num): # video index for the same prompt
                        intended_video_name = f'{prompt[:180]}{special_str}-{str(i)}{postfix}'
                        if intended_video_name in video_names: # if the video exists
                            intended_video_path = os.path.join(videos_path, intended_video_name)
                            prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print(f'Successfully found video: {intended_video_name}')
                        else:
                            print(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                            raise
                    cur_full_info_list.append(prompt_dict)

        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        if get_rank() == 0:
            save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')
        barrier()
        return cur_full_info_path


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', **kwargs):
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        if len(dimension_list) != len(set(dimension_list)):
            raise ValueError('dimension_list must not contain duplicate dimensions')
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        local_dimension_list = distribute_list_to_rank(dimension_list)
        pending_dimension_list = [
            dimension for dimension in local_dimension_list
            if _load_dimension_result(
                _dimension_result_path(self.output_path, name, dimension),
                dimension,
            ) is None
        ]
        for dimension in pending_dimension_list:
            try:
                os.remove(_dimension_error_path(self.output_path, name, dimension))
            except FileNotFoundError:
                pass
        barrier()
        
        for dimension in pending_dimension_list:
            try:
                submodules_list = init_submodules(
                    [dimension], local=local, read_frame=read_frame
                )[dimension]
                if dimension=="Multi-View_Consistency":
                    dimension_change = "Multi_View_Consistency"
                else:
                    dimension_change = dimension
                dimension_module = importlib.import_module(f'vbench2.{dimension_change.lower()}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension_change.lower()}')
                print(f'Rank {get_rank()} evaluating {dimension} with {cur_full_info_path}')
                results = evaluate_func(
                    cur_full_info_path,
                    self.device,
                    submodules_list,
                    local=local,
                    **kwargs,
                )
                dimension_output = _dimension_result_path(
                    self.output_path, name, dimension
                )
                _save_json_atomic({dimension: results}, dimension_output)
            except Exception as error:
                _save_json_atomic(
                    {
                        'dimension': dimension,
                        'rank': get_rank(),
                        'error': f'{type(error).__name__}: {error}',
                    },
                    _dimension_error_path(self.output_path, name, dimension),
                )
                raise

        poll_timeout = float(
            os.environ.get('VBENCH2_EVAL_TIMEOUT_SECONDS', 24 * 60 * 60)
        )
        poll_started_at = time.monotonic()
        while True:
            gathered_results = {}
            missing_dimensions = []
            for dimension in dimension_list:
                result = _load_dimension_result(
                    _dimension_result_path(self.output_path, name, dimension),
                    dimension,
                )
                if result is None:
                    error_path = _dimension_error_path(
                        self.output_path, name, dimension
                    )
                    try:
                        error = load_json(error_path)
                    except (OSError, ValueError):
                        error = None
                    if error is not None:
                        raise RuntimeError(
                            f'VBench2 scoring failed for {dimension}: {error}'
                        )
                    missing_dimensions.append(dimension)
                    continue
                gathered_results[dimension] = result
            if len(gathered_results) == len(dimension_list):
                break
            if time.monotonic() - poll_started_at >= poll_timeout:
                raise TimeoutError(
                    'Timed out waiting for VBench2 dimensions: '
                    f'{missing_dimensions}. Increase '
                    'VBENCH2_EVAL_TIMEOUT_SECONDS for longer evaluations.'
                )
            time.sleep(10)

        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            _save_json_atomic(gathered_results, output_name)
        print0(f'Evaluation results saved to {output_name}')
        barrier()

        return {
            dimension: gathered_results[dimension][0]
            for dimension in dimension_list
        }
