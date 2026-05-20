import os
import cv2
import shutil
from pathlib import Path


def split_video_into_shots(video_path, output_dir, frame_counts):
    """
    将视频按指定帧数分割成多个片段
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        frame_counts: 每个片段的帧数列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"无法打开视频文件 {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频信息: 总帧数={total_frames}, FPS={fps}")
    
    # 复制完整视频并重命名为 full.mp4
    full_video_path = os.path.join(output_dir, "full.mp4")
    shutil.copy(video_path, full_video_path)
    print(f"完整视频已保存为: {full_video_path}")
    
    # 按指定帧数分割视频
    current_frame = 0
    shot_idx = 1
    
    for frame_count in frame_counts:
        # 计算该片段在原视频中的起始和结束帧
        start_frame = current_frame
        end_frame = min(current_frame + frame_count, total_frames)
        
        # 如果已经没有足够的帧了，则跳过
        if start_frame >= total_frames:
            break
        
        # 设置视频写入器参数
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        output_path = os.path.join(output_dir, f"shot_{shot_idx}.mp4")
        
        # 确保有足够的帧来创建片段
        actual_frame_count = end_frame - start_frame
        if actual_frame_count <= 0:
            print(f"片段 {shot_idx} 没有足够帧，跳过")
            continue
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"正在创建片段 {shot_idx}: 帧范围 {start_frame} - {end_frame}, 实际帧数 {actual_frame_count}")
        
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 复制指定范围内的帧
        copied_frames = 0
        while current_frame < end_frame and copied_frames < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            current_frame += 1
            copied_frames += 1
        
        out.release()
        shot_idx += 1
        
        # 如果已经处理完所有帧，则退出
        if current_frame >= total_frames:
            print("已处理完所有帧，停止分割")
            break
    
    cap.release()
    
    print(f"视频分割完成，结果保存在: {output_dir}")
    return True


def main():
    # 源视频目录
    source_dir_str = r"C:\Users\Administrator\Downloads\onpolicy_selective\experiments\onpolicy_selective\visualization\sample\iter_600_ema\task_t2v\sample_step_4\seed_0"
    source_dir = Path(source_dir_str)
    
    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        return
    
    # 目标输出根目录
    target_root = source_dir / "processed_videos"
    
    # 每个片段的帧数
    frame_counts = [85, 84, 84, 84, 84, 84]
    
    # 获取目录中的所有视频文件并排序
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(source_dir.glob(f'*{ext}')))
    
    # 根据文件名排序，确保处理顺序正确
    video_files.sort()
    
    # 处理每个视频
    for idx, video_file in enumerate(video_files):
        # 确定输出文件夹名称（如果第一个视频以'00'开头，对应video1）
        if idx == 0 and video_file.name.startswith('00'):
            folder_name = 'video1'
        else:
            folder_name = f'video{idx + 1}'
        
        output_dir = target_root / folder_name
        
        print(f"\n正在处理视频: {video_file.name}")
        print(f"输出文件夹: {output_dir}")
        
        success = split_video_into_shots(video_file, output_dir, frame_counts)
        
        if success:
            print(f"已完成视频 {video_file.name} 的分割")
        else:
            print(f"处理视频 {video_file.name} 时出错")


if __name__ == "__main__":
    main()