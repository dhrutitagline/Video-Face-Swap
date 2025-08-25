import numpy as np
import cv2
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import gradio as gr

theme = gr.themes.Default(
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)

def add_bbox_padding(bbox, margin=5):
    return [
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin]


def select_handler(img, evt: gr.SelectData):
    if img is None:
        return None, -1

    faces = app.get(img)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    cropped_image = None  # must be None if no face is found
    face_index = -1
    sel_face_index = 0

    for idx, face in enumerate(faces):
        box = face.bbox.astype(np.int32)
        if point_in_box((box[0], box[1]), (box[2], box[3]), (evt.index[0], evt.index[1])):
            margin = int((box[2]-box[0]) * 0.35)
            box = add_bbox_padding(box, margin)
            box = np.clip(box, 0, None)
            cropped_image = img[box[1]:box[3], box[0]:box[2]]
            sel_face_index = idx
            break  # stop after first matching face

    return cropped_image, sel_face_index


def point_in_box(bl, tr, p):
   return bl[0] < p[0] < tr[0] and bl[1] < p[1] < tr[1]

def get_faces(img):
    if img is None:
        return None, 0
    faces = app.get(img)
    return img, len(faces)

def swap_face_fct(img_source, face_index, img_swap_face):
    if img_source is None or img_swap_face is None:
        return None
    faces = app.get(img_source)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    if len(faces) == 0 or face_index >= len(faces):
        return img_source
    src_face = app.get(img_swap_face)
    if len(src_face) == 0:
        return img_source
    src_face = sorted(src_face, key=lambda x: x.bbox[0])
    return swapper.get(img_source, faces[face_index], src_face[0], paste_back=True)

import subprocess

def merge_audio_ffmpeg(original_video, processed_video, final_output):
    """
    Merge original audio into the processed (silent) video using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-y",                      # overwrite output if exists
        "-i", processed_video,     # processed video without audio
        "-i", original_video,      # original video with audio
        "-c:v", "copy",            # copy video stream (no re-encode)
        "-c:a", "aac",             # encode audio in AAC (widely supported)
        "-map", "0:v:0",           # take video from processed file
        "-map", "1:a:0",           # take audio from original file
        final_output
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return final_output
    except Exception as e:
        print("FFmpeg merge failed:", e)
        return processed_video   # fallback: return video without audio


def swap_video_fct(video_path, output_path, source_face, destination_face, tolerance, preview=-1, progress=None):
    if source_face is None or destination_face is None:
        print("Missing source or destination face image")
        return None

    if progress is None:
        class DummyProgress:
            def __call__(self, *args, **kwargs): pass
        progress = DummyProgress()

    # Precompute destination & source
    dest_face = app.get(destination_face)
    if len(dest_face) == 0:
        print("No destination face found")
        return None
    dest_face = sorted(dest_face, key=lambda x: x.bbox[0])
    dest_face_feats = np.array([dest_face[0].normed_embedding], dtype=np.float32)

    src_face = app.get(source_face)
    if len(src_face) == 0:
        print("No source face found")
        return None
    src_face = sorted(src_face, key=lambda x: x.bbox[0])

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Save silent video in current working directory
    temp_output = os.path.join(os.getcwd(), "temp_silent.mp4")
    video_out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Process frames
    for i in range(frame_count if preview == -1 else 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i if preview == -1 else preview)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        faces = sorted(faces, key=lambda x: x.bbox[0])
        if len(faces) > 0:
            feats = np.array([f.normed_embedding for f in faces], dtype=np.float32)
            sims = np.dot(dest_face_feats, feats.T)
            max_index = np.argmax(sims)
            if sims[0][max_index] * 100 >= (100 - tolerance):
                frame = swapper.get(frame, faces[max_index], src_face[0], paste_back=True)

        if preview == -1:
            video_out.write(frame)
        else:
            cap.release()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()
    if preview == -1:
        video_out.release()

        # Save final output in current working directory
        final_output = output_path if output_path else os.path.join(os.getcwd(), "final_swapped.mp4")
        if os.path.isdir(final_output):
            final_output = os.path.join(final_output, "final_swapped.mp4")

        merge_audio_ffmpeg(video_path, temp_output, final_output)

        # cleanup silent file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"Final video with audio: {final_output}")
        return final_output, final_output

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video"
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    length = frame_count/fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    return f"Resolution: {width}x{height}\nLength: {length:.2f}s\nFps: {fps}\nFrames: {frame_count}"

def update_slider(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    return gr.update(minimum=0, maximum=frame_count-1, value=frame_count//2)

def show_preview(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_interface():
    title = 'Face Swap UI'
    with gr.Blocks(analytics_enabled=False, title=title) as face_swap_ui:
        with gr.Tab("Swap Face Video"):
            with gr.Row():
                with gr.Column():
                    source_video = gr.Video()
                    video_info = gr.Textbox(label="Video Information")
                    gr.Markdown("Select a frame for preview with the slider. Then select the face which should be swapped by clicking on it with the cursor")
                    video_position = gr.Slider(label="Frame preview",interactive=True)
                    frame_preview = gr.Image(label="Frame preview")
                    face_index = gr.Textbox(label="Face-Index",interactive=False)
                    with gr.Row():
                        dest_face_vid = gr.Image(label="Face tow swap",interactive=True)
                        source_face_vid = gr.Image(label="New Face")
                    gr.Markdown("The higher the tolerance the more likely a wrong face will be swapped. 30-40 is a good starting point.")
                    face_tolerance = gr.Slider(label="Tolerance",value=40,interactive=True)
                    preview_video = gr.Button("Preview")
                    video_file_path = gr.Text(label="Output Video path incl. file.mp4 (when left empty it will be put in the gradio temp dir)")
                    process_video = gr.Button("Process")
                with gr.Column():
                    with gr.Column(scale=1):
                        image_output = gr.Image()
                        output_video = gr.Video(interactive=False)
                        download_link = gr.File(label="Download Final Video", type="filepath")
                    with gr.Column(scale=1):
                        pass
            # Component Events
            source_video.upload(fn=analyze_video,inputs=source_video,outputs=video_info)
            video_info.change(fn=update_slider,inputs=source_video,outputs=video_position)
            #preview_button.click(fn=show_preview,inputs=[source_video, video_position],outputs=frame_preview)
            frame_preview.select(select_handler, frame_preview, [dest_face_vid, face_index ])
            video_position.change(show_preview,inputs=[source_video, video_position],outputs=frame_preview)
            process_video.click(fn=swap_video_fct,inputs=[source_video,video_file_path,source_face_vid,dest_face_vid, face_tolerance], outputs=[output_video, download_link])
            preview_video.click(fn=swap_video_fct,inputs=[source_video,video_file_path,source_face_vid,dest_face_vid, face_tolerance, video_position], outputs=image_output)

    face_swap_ui.queue().launch(debug=True,share=True)
    #face_swap_ui.launch()



if __name__ == "__main__":
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    create_interface()