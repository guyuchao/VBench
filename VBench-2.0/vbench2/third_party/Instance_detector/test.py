import os
from typing import Literal
import cv2
import json
from .split import split_video
from tqdm import tqdm

import qwen_vl_utils.vision_process as vision_process


_factor = 28
_model_seq_len = getattr(vision_process, 'MODEL_SEQ_LEN', 128000)
_qwen_vl_defaults = {
    'IMAGE_FACTOR': _factor,
    'MIN_PIXELS': vision_process.IMAGE_MIN_TOKEN_NUM * _factor**2,
    'MAX_PIXELS': vision_process.IMAGE_MAX_TOKEN_NUM * _factor**2,
    'VIDEO_MIN_PIXELS': vision_process.VIDEO_MIN_TOKEN_NUM * _factor**2,
    'VIDEO_MAX_PIXELS': vision_process.VIDEO_MAX_TOKEN_NUM * _factor**2,
    'VIDEO_TOTAL_PIXELS': int(_model_seq_len * _factor**2 * 0.9),
}
for _name, _value in _qwen_vl_defaults.items():
    if not hasattr(vision_process, _name):
        setattr(vision_process, _name, _value)

def infer_lora(engine, request_config, infer_request: 'InferRequest'):
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    return response

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps, frame_count // fps

from swift.llm import (PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest,
                        safe_snapshot_download, get_model_tokenizer)
from swift.tuners import Swift
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '602112'
os.environ['VIDEO_MIN_PIXELS'] = '100352'
os.environ['FPS_MAX_FRAMES'] = '50'
os.environ['FPS_MIN_FRAMES'] = '32'

def compute_anomaly(prompt_dict_ls, device, submodules_dict):
    processed_json=[]
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_path = submodules_dict['model']
    args = BaseArguments.from_pretrained(adapter_path)
    if submodules_dict.get('base_model'):
        args.model = submodules_dict['base_model']
    args.attn_impl = None
    model, tokenizer = get_model_tokenizer(args.model)
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)
    final_score=0
    final_num=0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            frame_count, fps, video_length = get_video_info(video_path)
            os.environ['FPS'] = f'{fps}'
            valid=True
            new_item={
                'video_path':video_path,
            }
            iteration_dir = os.path.dirname(os.path.dirname(video_path))
            video_clip_paths = os.path.join(
                iteration_dir, 'vbench2_info', 'instance_clips'
            )
            os.makedirs(video_clip_paths, exist_ok=True)
            split_video(video_path, video_clip_paths)
            for idx in range(int(video_length)-1):
                clip_path = os.path.join(video_clip_paths, f"{video_path.split('/')[-1][:-4]}_clip_{idx:04d}.mp4")
                message = [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant."
                },
                {
                    'role': 'user',
                    'content': 
                        [
                            {
                                'type': 'video',
                                'video': clip_path
                            }, 
                            {
                                'type': 'text',
                                'text': 'Does the video contain one or more of the following anomalies: sudden appearance, disappearance, fusion, fission?\nOptions:\nA. Yes\nB. No'
                            }
                        ]
                }]
                infer_request = InferRequest(messages=message)
                try:
                    output_text = infer_lora(engine, request_config, infer_request)
                except Exception as error:
                    raise RuntimeError(
                        f'Error processing video: {clip_path}'
                    ) from error
                if '(A)' in output_text or 'yes' in output_text.lower() or 'A'==output_text:
                    valid=False
                    break
            if valid:
                final_score+=1
                new_item['video_results']=1.0
            else:
                new_item['video_results']=0.0
            final_num+=1
            processed_json.append(new_item)
    return final_score/final_num, processed_json
