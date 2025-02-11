# Prediction interface for Cog ⚙️
# https://cog.run/python

import time
from cog import BasePredictor, Input, Path
import subprocess
import tempfile
import os
import shutil
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
import glob

MODEL_CACHE = "model_cache"
BASE_URL = f"https://weights.replicate.delivery/default/Hibiki/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    HF_REPO = "kyutai/hibiki-1b-pytorch-bf16"
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global load_t5, load_clap, RF, build_model

        model_files = [
            "models--kyutai--hibiki-1b-pytorch-bf16.tar",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

    def predict(
        self,
        audio_input: Path = Input(description="Input audio file to translate", default=None),
        video_input: Path = Input(description="Optional input video file", default=None),
        volume_reduction_db: int = Input(
            description="Volume reduction for original audio (dB)",
            default=30, ge=0, le=60
        ),
        cut_start_seconds: float = Input(
            description="Seconds to trim from start of translated audio",
            default=2.0, ge=0.0, le=4.0
        ),
        max_duration: int = Input(
            description="Maximum duration in seconds (0 for no limit)",
            default=0, ge=0
        )
    ) -> Path:
        """Run audio translation using Hibiki model"""
        temp_dir = tempfile.mkdtemp()
        
        # Handle video input
        original_video_path = None
        if video_input:
            original_video_path = str(video_input)
            input_path = self.extract_audio_from_video(original_video_path, temp_dir)
        else:
            input_path = str(audio_input)

        # Process audio input
        if max_duration > 0:
            audio = AudioSegment.from_file(input_path)
            audio = audio[:max_duration*1000]
            input_path = os.path.join(temp_dir, "processed_input.mp3")
            audio.export(input_path, format="mp3")

        # Run inference
        output_base = os.path.join(temp_dir, "out_en")
        command = [
            "python", "-m", "moshi.run_inference",
            input_path,
            output_base,
            "--hf-repo", self.HF_REPO,
            "--half",
            "--batch-size", "2"
        ]
        subprocess.run(command, check=True)

        # Get and process first chunk
        chunk_files = sorted(glob.glob(f"{output_base}-*"))
        if not chunk_files:
            raise RuntimeError("No audio files generated")
        
        translated_path = chunk_files[0] + ".wav"
        os.rename(chunk_files[0], translated_path)

        # Create combined audio using app.py's logic
        combined_path = self._create_combined_audio(
            input_path,  # Original audio path
            translated_path,
            temp_dir,
            volume_reduction_db,
            cut_start_seconds
        )

        # Handle video output if provided
        if original_video_path:
            final_path = self.replace_video_audio(original_video_path, combined_path, temp_dir)
        else:
            final_path = combined_path
        
        return Path(final_path)

    def _create_combined_audio(self, original_path, translated_path, temp_dir, db_reduction, cut_start):
        """Enhanced to match app.py's overlay_audio exactly"""
        original = AudioSegment.from_file(original_path).set_frame_rate(16000).set_channels(1)
        translated = AudioSegment.from_wav(translated_path).set_frame_rate(16000).set_channels(1)
        
        # Apply processing
        original = original - db_reduction
        if cut_start > 0:
            translated = translated[int(cut_start*1000):]

        # Add silence padding logic from app.py
        final_length = max(len(original), len(translated))
        
        if len(original) < final_length:
            original += AudioSegment.silent(duration=final_length - len(original))
        if len(translated) < final_length:
            translated += AudioSegment.silent(duration=final_length - len(translated))

        # Mix audio
        combined = original.overlay(translated)
        combined_path = os.path.join(temp_dir, "final_mix.wav")
        combined.export(combined_path, format="wav")
        
        return combined_path

    def extract_audio_from_video(self, video_path: str, temp_dir: str) -> str:
        """Extract audio from video matching app.py's functionality"""
        video = VideoFileClip(video_path)
        temp_audio = os.path.join(temp_dir, "extracted_audio.mp3")
        video.audio.write_audiofile(temp_audio, codec="mp3")
        return temp_audio

    def replace_video_audio(self, video_path: str, new_audio_path: str, temp_dir: str) -> str:
        """Enhanced with app.py's frame extension logic"""
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)
        
        # Extend video if needed (exact logic from app.py)
        if new_audio.duration > video.duration:
            last_frame = video.get_frame(video.duration - 0.1)  # 0.1s offset match
            freeze_frame = ImageClip(last_frame).set_duration(
                new_audio.duration - video.duration
            ).set_fps(video.fps)
            
            video = concatenate_videoclips([video, freeze_frame])

        # Set new audio and export
        video = video.set_audio(new_audio)
        output_path = os.path.join(temp_dir, "output_video.mp4")
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=video.fps,
            preset="medium"
        )
        return output_path

    def process_audio_chunks(self, output_base: str, temp_dir: str) -> str:
        """Handle multiple output chunks"""
        # Modified pattern to match extension-less files
        chunk_pattern = f"{output_base}-*"  # Removed .wav from pattern
        
        temp_files = []
        for src in sorted(glob.glob(chunk_pattern)):
            dest = os.path.join(temp_dir, os.path.basename(src) + ".wav")  # Add extension
            shutil.move(src, dest)
            temp_files.append(dest)
        
        if not temp_files:
            raise FileNotFoundError(f"No audio chunks found matching {chunk_pattern}")
        
        return temp_files[0]
