#!/usr/bin/env python3

import argparse
import json
import os
import socket
import subprocess
import sys
import webbrowser
from decimal import ROUND_HALF_UP, Decimal
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

# Import the refactored function from the new file
from html_generator import generate_html_content


def _run_ffprobe_duration(path: Path) -> Decimal | None:
    """Return duration (in seconds) as Decimal using ffprobe, or None if unreadable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if res.returncode != 0:
        return None
    txt = res.stdout.strip()
    if not txt:
        return None
    try:
        return Decimal(txt)
    except Exception:
        return None


def _breakdown_time(seconds: Decimal) -> dict[str, str]:
    """Break seconds into hr, min, sec, ms (all as zero-padded strings)."""
    ms_total = (seconds * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ms_total = int(ms_total)
    hr, rem_ms = divmod(ms_total, 3600 * 1000)
    minute, rem_ms = divmod(rem_ms, 60 * 1000)
    sec, ms = divmod(rem_ms, 1000)
    return {"hr": f"{hr:02d}", "min": f"{minute:02d}", "sec": f"{sec:02d}", "ms": f"{ms:03d}"}


def _build_output_stem(video_dir: Path, video_path: Path, naming_mode: str) -> str:
    """Build destination folder stem for one input video."""
    rel = video_path.relative_to(video_dir)
    parts = rel.parts
    if naming_mode == "level1_level2" and len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return video_path.stem


class HTMLVideoAnnotationVisualizer:
    """A class to create and serve an HTML page for video annotation visualization."""

    def __init__(
        self,
        video_dir: str,
        ann_dir: str,
        output_dir: str | None,
        durations: str | None,
        annotation_file: str,
        mapping_file: str | None = None,
        auto_mapping: bool = False,
        recursive: bool = False,
        output_name_mode: str = "auto",
        port: int = 8000,
    ) -> None:
        """Initialize the visualizer with directories and settings."""
        self.video_dir = os.path.abspath(video_dir)
        self.ann_dir = os.path.abspath(ann_dir)
        self.output_dir = os.path.abspath(output_dir) if output_dir else self.ann_dir
        self.durations_file = durations
        self.durations_data: dict[str, Any] = {}
        self.annotation_file = annotation_file
        self.mapping_file = mapping_file
        self.auto_mapping = auto_mapping
        self.recursive = recursive
        self.output_name_mode = output_name_mode
        self.video_mapping: dict[str, str] = {}
        self.annotations_data: dict[str, Any] = {}
        self.video_paths_map: dict[str, str] = {}
        self.video_output_ids: dict[str, str] = {}
        self.port = port
        self.max_port_attempts = 5
        self.videos = self.load_videos()
        self.load_mapping()
        self.load_all_annotations()
        self.load_durations()

    def _candidate_output_ids(self, video_key: str) -> list[str]:
        """Return candidate annotation folder ids in priority order."""
        source_path = Path(self.video_paths_map[video_key])
        root = Path(self.video_dir)
        level_id = _build_output_stem(root, source_path, "level1_level2")
        stem_id = source_path.stem
        if self.output_name_mode == "stem":
            return [stem_id]
        if self.output_name_mode == "level1_level2":
            return [level_id]
        # auto: try level1_level2 first, then stem
        return [level_id, stem_id] if level_id != stem_id else [stem_id]

    def load_durations(self) -> None:
        """Load video durations from the provided JSON file, or compute from video_dir if not provided."""
        if self.durations_file and os.path.exists(self.durations_file):
            try:
                with open(self.durations_file, encoding="utf-8") as f:
                    self.durations_data = json.load(f)
                print(f"Successfully loaded video durations from {self.durations_file}")
                return
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.durations_file}. Computing durations from videos.")
            except Exception as e:
                print(f"Warning: Error reading {self.durations_file}: {e}. Computing durations from videos.")
        # No file or failed to load: compute durations from video_dir
        print("Computing video durations from video directory...")
        self.durations_data = {}
        for video_key, source_path in self.video_paths_map.items():
            dur = _run_ffprobe_duration(Path(source_path))
            if dur is None:
                continue
            youtube_id = self.video_output_ids.get(video_key, Path(video_key).stem)
            self.durations_data[youtube_id] = _breakdown_time(dur)
        if self.durations_data:
            print(f"Computed durations for {len(self.durations_data)} video(s).")
        else:
            print("No video durations could be computed; video metadata may be used when available.")

    def auto_generate_mapping(self) -> dict[str, str]:
        """Automatically generate video to annotation mapping based on filename matching."""
        print("Auto-generating video mapping...")

        mapping = {}
        matched_count = 0

        ann_root = Path(self.ann_dir)
        for video_key in self.videos:
            candidates = self._candidate_output_ids(video_key)

            matched = next((candidate for candidate in candidates if (ann_root / candidate).exists()), None)
            if matched:
                mapping[video_key] = matched
                matched_count += 1
                print(f"✓ Matched: {video_key} -> {matched}")
            else:
                print(f"✗ No annotation found for: {video_key}")

        print(f"\nAuto-mapping complete: {matched_count}/{len(self.videos)} videos matched")
        return mapping

    def load_mapping(self) -> None:
        """Load video mapping from file or auto-generate if enabled."""
        if self.auto_mapping or not self.mapping_file or not os.path.exists(self.mapping_file):
            self.video_mapping = self.auto_generate_mapping()
            if self.mapping_file:
                with open(self.mapping_file, "w") as f:
                    json.dump(self.video_mapping, f, indent=2)
                print(f"Auto-generated mapping saved to: {self.mapping_file}")
        else:
            try:
                with open(self.mapping_file) as f:
                    self.video_mapping = json.load(f)
                print(f"Loaded mapping from: {self.mapping_file}")
            except FileNotFoundError:
                print(f"Mapping file not found: {self.mapping_file}")
                print("Falling back to auto-mapping...")
                self.video_mapping = self.auto_generate_mapping()
            except json.JSONDecodeError as e:
                print(f"Error parsing mapping file: {e}")
                print("Falling back to auto-mapping...")
                self.video_mapping = self.auto_generate_mapping()

    def load_videos(self) -> list[str]:
        """Discover mp4 files and build key/path/id maps."""
        root = Path(self.video_dir)
        if not root.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        candidates = sorted(root.rglob("*.mp4") if self.recursive else root.glob("*.mp4"))
        videos: list[str] = []
        for video_path in candidates:
            key = str(video_path.relative_to(root)).replace("\\", "/") if self.recursive else video_path.name
            # Preferred id shown in metadata; mapping logic still tries fallbacks.
            preferred_mode = self.output_name_mode if self.output_name_mode != "auto" else "level1_level2"
            output_id = _build_output_stem(root, video_path, preferred_mode)
            videos.append(key)
            self.video_paths_map[key] = str(video_path.resolve())
            self.video_output_ids[key] = output_id
        return videos

    def time_to_seconds(self, time_obj: dict[str, str] | str) -> float:
        """Convert {hr, min, sec, ms} format or 'H:MM:SS.mmm' string to seconds."""
        if isinstance(time_obj, str):
            return self._parse_timestamp_str(time_obj)
        hours = int(time_obj.get("hr", 0))
        minutes = int(time_obj.get("min", 0))
        seconds = int(time_obj.get("sec", 0))
        milliseconds = int(time_obj.get("ms", 0))
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def _parse_timestamp_str(self, s: str) -> float:
        """Parse 'H:MM:SS.mmm' or 'MM:SS.mmm' timestamp string to seconds."""
        parts = s.strip().split(".")
        frac = int(parts[1]) / 1000.0 if len(parts) == 2 else 0.0
        parts = parts[0].split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            return 0.0
        return h * 3600 + m * 60 + sec + frac

    def seconds_to_time_str(self, seconds: float) -> str:
        """Convert seconds to MM:SS.mmm format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def load_annotations(self, video_file: str) -> dict[str, Any]:
        """Load annotations for a specific video."""
        youtube_id = (
            self.video_mapping.get(video_file)
            or self.video_mapping.get(Path(video_file).name)
            or self.video_output_ids.get(video_file, os.path.splitext(Path(video_file).name)[0])
        )
        ann_path = os.path.join(self.ann_dir, youtube_id, self.annotation_file)

        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)

            summary = data.get("summary", "No summary")
            summary_video_only = data.get("summary_video_only", "")
            summary_audio_only = data.get("summary_audio_only", "")
            raw_list = data.get("video_descriptions", data.get("annotations", []))
            processed_annotations = []

            for ann_idx in range(len(raw_list)):
                ann = raw_list[ann_idx]
                if not isinstance(ann, dict):
                    continue
                # Prefer "text" (video_descriptions) then "desc" (annotations)
                segment_text = ann.get("text") or ann.get("desc") or ""
                if "t0" not in ann or "t1" not in ann:
                    continue
                t0_sec = self.time_to_seconds(ann["t0"])
                t1_sec = self.time_to_seconds(ann["t1"])
                processed_annotations.append(
                    {
                        "id": f"ann-{ann_idx}",
                        "object_id": ann.get("object_id", f"event_{ann_idx + 1:03d}"),
                        "start": t0_sec,
                        "end": t1_sec,
                        "text": str(segment_text),
                        "audio": ann.get("audio", ""),
                        "start_str": self.seconds_to_time_str(t0_sec),
                        "end_str": self.seconds_to_time_str(t1_sec),
                        "edited_desc": ann.get("edited_desc") == "Yes",
                        "edited_audio": ann.get("edited_audio") == "Yes",
                        "edited_start_time": ann.get("edited_start_time") == "Yes",
                        "edited_end_time": ann.get("edited_end_time") == "Yes",
                    }
                )

            def _process_query_list(raw_queries: Any, prefix: str) -> list[dict[str, Any]]:
                processed: list[dict[str, Any]] = []
                if not isinstance(raw_queries, list):
                    return processed
                for q_idx, query in enumerate(raw_queries):
                    if not isinstance(query, dict):
                        continue
                    if "t0" not in query or "t1" not in query:
                        continue
                    t0_sec = self.time_to_seconds(query["t0"])
                    t1_sec = self.time_to_seconds(query["t1"])
                    processed.append(
                        {
                            "id": f"{prefix}-{q_idx}",
                            "start": t0_sec,
                            "end": t1_sec,
                            "start_str": self.seconds_to_time_str(t0_sec),
                            "end_str": self.seconds_to_time_str(t1_sec),
                            "text": str(query.get("text", "")),
                            "acknowledged": bool(query.get("acknowledged", False)),
                            "edited_text": query.get("edited_text") == "Yes",
                            "edited_time": query.get("edited_time") == "Yes",
                        }
                    )
                return processed

            visual_queries = _process_query_list(data.get("visual_queries", []), "vq")
            audio_queries = _process_query_list(data.get("audio_queries", []), "aq")

            return {
                "summary": summary,
                "summary_video_only": summary_video_only,
                "summary_audio_only": summary_audio_only,
                "annotations": processed_annotations,
                "visual_queries": visual_queries,
                "audio_queries": audio_queries,
                "youtube_id": youtube_id,
                "edited_summary": data.get("edited_summary") == "Yes",
                "edited_summary_video_only": data.get("edited_summary_video_only") == "Yes",
                "edited_summary_audio_only": data.get("edited_summary_audio_only") == "Yes",
                "non_english": bool(data.get("non_english", False)),
            }
        else:
            return {
                "summary": "No summary available for this video.",
                "summary_video_only": "",
                "summary_audio_only": "",
                "annotations": [],
                "visual_queries": [],
                "audio_queries": [],
                "youtube_id": youtube_id,
                "edited_summary": False,
                "edited_summary_video_only": False,
                "edited_summary_audio_only": False,
                "non_english": False,
            }

    def load_all_annotations(self) -> None:
        """Load annotations for all videos."""
        for video_file in self.videos:
            self.annotations_data[video_file] = self.load_annotations(video_file)

    def get_ip_address(self) -> str:
        """Get the local IP address of the machine."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def start_server(self) -> None:
        """Start the HTTP server and open browser."""
        video_paths = {video_file: f"videos/{video_file}" for video_file in self.videos}

        html_content = generate_html_content(self.videos, self.annotations_data, video_paths, self.durations_data)

        with open("video_annotation_visualizer.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            def end_headers(self) -> None:
                self.send_header("Accept-Ranges", "bytes")
                super().end_headers()

            def translate_path(self, path: str) -> str:
                path = path.split("?", 1)[0].split("#", 1)[0]
                if path.startswith("/videos/"):
                    video_key = unquote(path[len("/videos/") :])
                    mapped = self.server.video_paths_map.get(video_key)
                    if mapped:
                        return mapped
                    return os.path.join(self.server.video_dir, os.path.basename(video_key))
                else:
                    return super().translate_path(path)

            def do_GET(self) -> None:  # noqa: N802
                """Handle GET requests with HTTP Range support for video seeking."""
                range_header = self.headers.get("Range")
                if not range_header:
                    return super().do_GET()

                # Resolve the file path using the existing translate_path logic
                file_path = self.translate_path(self.path)
                if not os.path.isfile(file_path):
                    self.send_error(404, "File not found")
                    return

                file_size = os.path.getsize(file_path)

                # Parse "Range: bytes=<start>-<end>" header
                if not range_header.startswith("bytes="):
                    return super().do_GET()  # Malformed; fall back to full response

                range_spec = range_header[len("bytes=") :]
                parts = range_spec.split("-", 1)
                try:
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else file_size - 1
                except ValueError:
                    return super().do_GET()

                # Clamp and validate
                end = min(end, file_size - 1)
                if start < 0 or start > end or start >= file_size:
                    self.send_response(416)  # Range Not Satisfiable
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                content_length = end - start + 1
                content_type = self.guess_type(file_path)

                try:
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(content_length)

                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(content_length))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    self.send_error(500, f"Error serving range request: {e}")

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/save-annotations":
                    try:
                        content_length = int(self.headers["Content-Length"])
                        post_data = self.rfile.read(content_length)
                        payload = json.loads(post_data)

                        youtube_id = payload["youtube_id"]
                        new_annotations = payload["annotations"]
                        new_summary = payload.get("summary")

                        output_dir_path = os.path.join(self.server.output_dir, youtube_id)
                        output_ann_path = os.path.join(output_dir_path, self.server.annotation_file)

                        os.makedirs(output_dir_path, exist_ok=True)

                        updated_data = {
                            "summary": new_summary,
                            "summary_video_only": payload.get("summary_video_only", ""),
                            "summary_audio_only": payload.get("summary_audio_only", ""),
                            "summary_multimodal": payload.get("summary_multimodal", new_summary),
                            "edited_summary": payload.get("edited_summary", "No"),
                            "edited_summary_video_only": payload.get("edited_summary_video_only", "No"),
                            "edited_summary_audio_only": payload.get("edited_summary_audio_only", "No"),
                            "non_english": bool(payload.get("non_english", False)),
                            "video_descriptions": payload.get("video_descriptions", []),
                            "visual_queries": payload.get("visual_queries", []),
                            "audio_queries": payload.get("audio_queries", []),
                            # Backward-compat key
                            "annotations": new_annotations,
                        }

                        with open(output_ann_path, "w") as f:
                            json.dump(updated_data, f, indent=4)

                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"Annotations saved successfully")

                    except Exception as e:
                        self.send_error(500, f"Error saving annotations: {e}")
                else:
                    self.send_error(404, "Endpoint not found")

        port = self.port
        max_attempts = self.max_port_attempts

        for attempt in range(max_attempts):
            try:
                server_address = ("0.0.0.0", port)
                server = ThreadingHTTPServer(server_address, CustomHTTPRequestHandler)

                server.video_dir = self.video_dir
                server.ann_dir = self.ann_dir
                server.output_dir = self.output_dir
                server.annotation_file = self.annotation_file
                server.video_paths_map = self.video_paths_map

                ip_address = self.get_ip_address()
                local_url = f"http://localhost:{port}/video_annotation_visualizer.html"
                network_url = f"http://{ip_address}:{port}/video_annotation_visualizer.html"

                print("🚀 Server started!")
                print(f"   - To access on this machine: {local_url}")
                print(f"   - To share on your network: {network_url}")
                print("📄 HTML file generated: video_annotation_visualizer.html")
                print(f"🎬 Serving videos from: {self.video_dir}")
                print(f"💾 Saving annotations to: {self.output_dir}")

                try:
                    webbrowser.open(local_url)
                except Exception:
                    print("⚠️  Could not open browser automatically.")

                print("⏹️  Press Ctrl+C to stop server")

                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    print("\n🛑 Server stopped")
                    server.server_close()
                break

            except OSError as e:
                if "Address already in use" in str(e) and attempt < max_attempts - 1:
                    port += 1
                    print(f"⚠️  Port {self.port} in use, trying port {port}...")
                else:
                    print(f"❌ Failed to start server: {e}")
                    break


def main() -> None:
    """Parse arguments and run the visualizer."""
    parser = argparse.ArgumentParser(description="HTML Video Annotation Visualizer with Video Playback and Editing")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument(
        "--annotation_file", type=str, required=True, help="Name of the annotation file to load and save"
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save modified annotations. Defaults to ann_dir.")
    parser.add_argument(
        "--durations",
        type=str,
        default=None,
        help="JSON file containing video durations. If omitted, durations are computed from the video directory.",
    )
    parser.add_argument(
        "--mapping", type=str, help="JSON file mapping video filenames to YouTube IDs (optional with --auto_mapping)"
    )
    parser.add_argument(
        "--auto_mapping", action="store_true", help="Automatically generate mapping based on filename matching"
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively discover .mp4 files under --video_dir.")
    parser.add_argument(
        "--output-name-mode",
        type=str,
        choices=("auto", "stem", "level1_level2"),
        default="auto",
        help=(
            "How annotation folder IDs are inferred when mapping is missing. "
            "'auto' tries level1_level2 then stem. "
            "'stem' uses video filename stem; "
            "'level1_level2' mirrors mp_base naming (<level1>/<level2>/<file>.mp4 -> level1_level2)."
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server (default: 8000)")
    args = parser.parse_args()

    if args.output_dir and os.path.exists(args.output_dir):
        print(f"Output directory already exists: {args.output_dir}")
        choice = input("Do you want to overwrite it? [Y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("Aborting. Please remove it or choose a different output directory.")
            sys.exit(1)
        else:
            print("Continuing and overwriting existing files...")

    visualizer = HTMLVideoAnnotationVisualizer(
        args.video_dir,
        args.ann_dir,
        args.output_dir,
        args.durations,
        args.annotation_file,
        args.mapping,
        args.auto_mapping,
        args.recursive,
        args.output_name_mode,
        args.port,
    )
    visualizer.start_server()


if __name__ == "__main__":
    main()
