# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False):
    import ffmpeg
    
    if is_depths:
        # Scale depth frames to 10-bit range
        frames = frames.astype(np.float32)
        d_min, d_max = frames.min(), frames.max()
        print(f"Depth range - Min: {d_min:.4f}, Max: {d_max:.4f}")
        frames = ((frames - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
        
        # Create temporary raw file
        raw_file = output_video_path + '.raw'
        frames.tofile(raw_file)
        
        # Setup ffmpeg stream
        stream = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray16le',
                  s=f'{frames.shape[2]}x{frames.shape[1]}', r=fps)
            .output(output_video_path, 
                   pix_fmt='yuv420p10le',
                   crf=18,
                   vcodec='libx265',
                   **{'x265-params': 'lossless=1'})
            .overwrite_output()
        )
        
        # Run ffmpeg with raw input
        with open(raw_file, 'rb') as raw:
            stream.run(input=raw.read())
            
        # Clean up temporary file
        import os
        os.remove(raw_file)
        
    else:
        # For regular 8-bit frames, use original imageio implementation
        writer = imageio.get_writer(
            output_video_path,
            fps=fps,
            macro_block_size=1,
            codec='libx265',
            ffmpeg_params=['-crf', '18'],
            pixelformat='yuv420p'
        )
        
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
        writer.close()
