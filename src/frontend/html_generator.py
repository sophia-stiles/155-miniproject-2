# html_generator.py

import json
from typing import Any


def generate_video_options(videos: list[str], annotations_data: dict[str, Any]) -> str:
    """Generate HTML options for video selection."""
    options = []
    for i, video_file in enumerate(videos):
        selected = "selected" if i == 0 else ""
        # Safely get data for the video file
        data = annotations_data.get(video_file, {})
        ann_count = len(data.get("annotations", []))
        options.append(f'<option value="{video_file}" {selected}>{video_file} ({ann_count} annotations)</option>')
    return "\n".join(options)


def generate_html_content(
    videos: list[str],
    annotations_data: dict[str, Any],
    video_paths: dict[str, str],
    durations_data: dict[str, Any],
) -> str:
    """Generates the complete HTML content for the video annotation visualizer.

    Args:
        videos: A list of video filenames.
        annotations_data: A dictionary containing the annotation data for all videos.
        video_paths: A dictionary mapping video filenames to their server paths.
        durations_data: A dictionary mapping video ids to duration metadata.

    Returns:
        The complete HTML content as a string.
    """
    video_options_html = generate_video_options(videos, annotations_data)
    annotations_data_json = json.dumps(annotations_data, indent=2)
    video_paths_json = json.dumps(video_paths, indent=2)
    durations_data_json = json.dumps(durations_data, indent=2)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Annotation Visualizer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
                         Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
            color: #333;
            overflow: hidden; /* Prevent body scrollbars */
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 2em;
            color: #1a237e;
        }}
        .main-content {{
            display: flex;
            height: calc(100vh - 100px); /* Adjust height based on header */
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            overflow: auto; /* Allow individual panel scrolling */
        }}
        .video-panel {{
            flex: 0 0 24%; /* Initial width, smaller */
            min-width: 250px;
        }}
        .annotations-panel {{
            flex: 1 1 auto;
            min-width: 300px;
            display: flex;
            flex-direction: column;
        }}
        .resizer {{
            flex: 0 0 10px;
            background: #e0e0e0;
            cursor: col-resize;
            border-left: 1px solid #ccc;
            border-right: 1px solid #ccc;
        }}
        .resizer:hover {{
            background: #bdbdbd;
        }}
        h3, h4 {{
            color: #3f51b5;
            border-bottom: 2px solid #e8eaf6;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 15px;
        }}
        .video-selector select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            margin-bottom: 20px;
        }}
        .video-container {{
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 15px;
        }}
        .video-player {{
            width: 100%;
            height: auto;
            background: #000;
        }}
        .video-controls button, .nav-controls button, .save-btn,
        .ack-btn, .save-item-btn, .unack-btn, .add-btn {{
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            color: white;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        .video-controls {{ margin-top: 15px; }}

        .next-btn {{ background-color: #673ab7; }}
        .add-btn {{ background-color: #00bcd4; }}
        .ack-btn {{ background-color: #009688; padding: 4px 10px; font-size: 12px; }}
        .save-item-btn {{ background-color: #3f51b5; padding: 4px 10px; font-size: 12px; }}
        .unack-btn {{ background-color: #f44336; padding: 4px 10px; font-size: 12px; }}
        .annotation-controls {{ float: right; }}
        .summary {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        #summaryText {{
            font-size: 0.9em; /* Smaller font for summary */
            line-height: 1.4;
        }}
        #summaryText[contenteditable="true"],
        #summaryVideoOnlyText[contenteditable="true"],
        #summaryAudioOnlyText[contenteditable="true"] {{
            background-color: #fffde7;
            outline: 1px dashed #fbc02d;
        }}
        .stats-section-compact {{
            padding: 10px;
            background: #f1f3f5;
            border-radius: 5px;
            margin-bottom: 15px;
            text-align: center;
            font-size: 0.9em;
        }}
        .stats-section-compact span {{
            margin: 0 10px;
        }}
        .stats-section-compact b {{
            color: #3f51b5;
        }}
        .annotation-item {{
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #FF5722;
            overflow: hidden; /* Clearfix for floated buttons */
        }}
        .acknowledged-item {{
            border-left: 4px solid #4CAF50;
            opacity: 0.7;
            font-size: 0.8em;
            padding: 5px;
        }}
        .annotation-text, .annotation-audio {{
            width: 100%;
            min-height: 40px;
            border: 1px solid #ddd;
            padding: 5px;
            border-radius: 3px;
            box-sizing: border-box;
        }}
        .annotation-text[contenteditable="true"],
        .annotation-audio {{
            background-color: #fffde7;
            outline: 1px dashed #fbc02d;
        }}
        .annotation-audio {{
            background-color: #e3f2fd;
            display: block;
            resize: vertical;
            overflow-wrap: anywhere;
            min-height: 88px;
            line-height: 1.35;
        }}
        .summary-subsection {{
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .summary-subtext {{
            background: #f5f9ff;
            border: 1px solid #d7e6ff;
            border-radius: 4px;
            padding: 8px;
            white-space: pre-wrap;
        }}
        .summary-subheader {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .summary-subbuttons button {{
            margin: 0;
        }}
        .query-lists-section {{
            margin-top: 0;
            border-top: 1px solid #e0e0e0;
            padding-top: 8px;
            min-height: 0;
            overflow-y: auto;
        }}
        .query-group {{
            margin-bottom: 10px;
        }}
        .query-item {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            border-radius: 4px;
            padding: 6px;
            margin: 5px 0;
            font-size: 0.9em;
        }}
        .query-item.acknowledged-item {{
            border-left: 4px solid #4CAF50;
            background: #f4fbf4;
        }}
        .query-text {{
            width: 100%;
            min-height: 34px;
            border: 1px solid #ddd;
            padding: 5px;
            border-radius: 3px;
            background: #fffde7;
            outline: 1px dashed #fbc02d;
            box-sizing: border-box;
            overflow-wrap: anywhere;
        }}
        .query-time {{
            font-weight: bold;
            color: #ef6c00;
            font-size: 12px;
        }}
        .annotation-time {{
            font-weight: bold;
            color: #FF5722;
            font-size: 14px;
        }}
        .time-input-container input {{
            width: 80px;
            padding: 2px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .no-annotations {{ color: #777; font-style: italic; text-align: center; padding: 20px; }}
        .annotation-lists-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            gap: 10px;
        }}
        #active-list-wrapper {{
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            min-height: 80px;
            max-height: 50%;
            overflow: hidden;
        }}
        #queries-list-wrapper {{
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            min-height: 80px;
            max-height: 40%;
            overflow: hidden;
        }}
        #acknowledged-list-wrapper {{
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            min-height: 80px;
            border-top: 1px solid #e0e0e0;
            padding-top: 8px;
            overflow: hidden;
        }}
        #acknowledged-list-wrapper h4 {{
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        #annotationList, #queryList, #acknowledgedList {{
            flex: 1;
            overflow-y: auto;
        }}
        .nav-controls {{
            display: flex;
            gap: 20px;
            align-items: start;
        }}
        .nav-section {{
            flex: 1;
        }}
        .nav-controls button {{ background-color: #9c27b0; }}
        .time-slider {{ width: 100%; margin: 10px 0; }}
        .time-display {{ text-align: center; font-size: 18px; font-weight: bold; color: #333; margin: 10px 0; }}
        #saveStatus {{ text-align: center; margin-bottom: 10px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Video Annotation Visualizer</h1>
        </div>

        <div class="main-content">
            <div class="panel video-panel" id="leftPanel">
                <h3>Video Details</h3>
                <div class="video-selector">
                    <select id="videoSelect" onchange="changeVideo()">
                        {video_options_html}
                    </select>
                </div>

                <div class="video-container">
                    <video id="videoPlayer" class="video-player" controls>
                        <source id="videoSource" src="" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>

                <div class="nav-section">
                    <h4>Time Navigation</h4>
                    <div class="time-display" id="timeDisplay">Time: 00:00.000</div>
                    <input type="range" id="timeSlider" class="time-slider" min="0"
                           max="1000" value="0">
                    <div class="time-input" style="display: flex; gap: 10px; margin-bottom: 10px;">
                        <input type="number" id="timeInput" step="0.1" placeholder="Time (s)"
                               value="0" style="flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 5px;">
                        <button onclick="setTime()" style="background: #4CAF50;">Go</button>
                    </div>
                </div>

                <div class="stats-section-compact">
                    <span>Total Annotations: <b id="totalAnnotationsStat">0</b></span>
                    <span>Active: <b id="activeAnnotationsStat">0</b></span>
                    <span>Duration: <b id="videoDuration">--:--</b></span>
                </div>

                <div class="summary" id="summary">
                    <h4>📝 Video Summary
                        <button class="save-item-btn" onclick="saveAllAnnotations()">Save</button>
                    </h4>
                    <div class="summary-subsection" id="summaryMainSection">
                        <div class="summary-subheader">
                            <b>Multimodal Summary</b>
                            <div class="summary-subbuttons">
                                <button class="ack-btn" onclick="acknowledgeSummary()">Ack</button>
                            </div>
                        </div>
                        <div id="summaryText" class="summary-subtext" contenteditable="true"
                             oninput="updateSummaryText(this.innerText)">Select a video to see summary.</div>
                    </div>
                    <div class="summary-subsection" id="summaryVideoOnlySection">
                        <div class="summary-subheader">
                            <b>Video-only Summary</b>
                            <div class="summary-subbuttons">
                                <button class="ack-btn" onclick="acknowledgeSummaryVideoOnly()">Ack</button>
                            </div>
                        </div>
                        <div id="summaryVideoOnlyText" class="summary-subtext" contenteditable="true"
                             oninput="updateSummaryVideoOnlyText(this.innerText)">No video-only summary.</div>
                    </div>
                    <div class="summary-subsection" id="summaryAudioOnlySection">
                        <div class="summary-subheader">
                            <b>Audio-only Summary</b>
                            <div class="summary-subbuttons">
                                <button class="ack-btn" onclick="acknowledgeSummaryAudioOnly()">Ack</button>
                            </div>
                        </div>
                        <div id="summaryAudioOnlyText" class="summary-subtext" contenteditable="true"
                             oninput="updateSummaryAudioOnlyText(this.innerText)">No audio-only summary.</div>
                    </div>
                </div>

                <div class="video-controls">
                    <button onclick="nextVideo()" class="next-btn">Next Video ➡️</button>
                    <button onclick="addNewAnnotation()" class="add-btn">➕ New Annotation</button>
                </div>
            </div>

            <div class="resizer" id="resizer"></div>

            <div class="panel annotations-panel" id="rightPanel">
                <div class="annotation-lists-container">
                    <div id="active-list-wrapper">
                        <h3>Active Annotations</h3>
                        <div id="annotationList">
                            <div class="no-annotations">Select a video to see annotations.</div>
                        </div>
                    </div>
                    <div id="queries-list-wrapper">
                        <div class="query-lists-section">
                            <h4>Queries</h4>
                            <div id="queryList">
                                <div class="no-annotations">Select a video to see queries.</div>
                            </div>
                        </div>
                    </div>

                    <div id="acknowledged-list-wrapper">
                        <h4>Acknowledged Items</h4>
                        <div id="acknowledgedList">
                            <div class="no-annotations">No acknowledged items yet.</div>
                        </div>
                    </div>
                </div>

                <div id="saveStatus"></div>
            </div>
        </div>
    </div>

    <script>
        const annotationsData = {annotations_data_json};
        const videoPaths = {video_paths_json};
        const durationsData = {durations_data_json};
        let currentVideo = Object.keys(annotationsData)[0];
        let currentTime = 0;
        let videoPlayer = null;
        let annotationBoundaries = [];
        let acknowledgedIds = new Set();
        let acknowledgedQueryIds = new Set();
        let summaryAcknowledged = false;
        let summaryVideoOnlyAcknowledged = false;
        let summaryAudioOnlyAcknowledged = false;
        let originalAnnotations = {{}};
        let accurateDuration = 0;
        let timeSliderDragging = false;
        let ignoreTimeUpdateUntilSeeked = false;

        function seekToTimeSec(requestedTimeSec, source = 'unknown') {{
            const slider = document.getElementById('timeSlider');
            const durationSec = videoPlayer && Number.isFinite(videoPlayer.duration) ? videoPlayer.duration : null;
            let target = Number(requestedTimeSec);
            if (!Number.isFinite(target)) target = 0;
            if (target < 0) target = 0;
            if (durationSec !== null && target > durationSec) target = durationSec;

            currentTime = target;
            document.getElementById('timeInput').value = currentTime.toFixed(3);

            if (slider) {{
                slider.value = String(Math.round(currentTime * 1000));
            }}

            if (videoPlayer) {{
                ignoreTimeUpdateUntilSeeked = true;
                videoPlayer.currentTime = currentTime;
                videoPlayer.pause();
            }}
            updateDisplay();
        }}

        function escapeAnnotationHtml(s) {{
            if (s == null || s === undefined) return '';
            return String(s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            videoPlayer = document.getElementById('videoPlayer');
            if (videoPlayer) {{
                videoPlayer.addEventListener('timeupdate', updateTimeFromVideo);
                videoPlayer.addEventListener('loadedmetadata', updateVideoDuration);
                videoPlayer.addEventListener('seeked', function() {{
                    ignoreTimeUpdateUntilSeeked = false;
                    if (!timeSliderDragging) {{
                        document.getElementById('timeSlider').value = Math.round(videoPlayer.currentTime * 1000);
                        updateDisplay();
                    }}
                }});
            }}
            const timeSliderEl = document.getElementById('timeSlider');
            if (timeSliderEl) {{
                timeSliderEl.addEventListener('mousedown', function() {{
                    timeSliderDragging = true;
                }});
                timeSliderEl.addEventListener('mouseup', function() {{
                    timeSliderDragging = false;
                }});
                timeSliderEl.addEventListener('mouseleave', function() {{
                    timeSliderDragging = false;
                }});
                timeSliderEl.addEventListener(
                    'touchstart',
                    function() {{ timeSliderDragging = true; }},
                    {{ passive: true }}
                );
                timeSliderEl.addEventListener(
                    'touchend',
                    function() {{ timeSliderDragging = false; }},
                    {{ passive: true }}
                );

                // Robust slider wiring so clicks, drags, and keyboard changes always seek.
                timeSliderEl.addEventListener('input', function(e) {{
                    seekToTimeSec(Number(e.target.value) / 1000, 'slider input');
                }});
                timeSliderEl.addEventListener('change', function(e) {{
                    seekToTimeSec(Number(e.target.value) / 1000, 'slider change');
                }});
                timeSliderEl.addEventListener('click', function(e) {{
                    seekToTimeSec(Number(e.target.value) / 1000, 'slider click');
                }});
            }}
            if (Object.keys(annotationsData).length > 0) {{
                // Deep copy for original data
                originalAnnotations = JSON.parse(JSON.stringify(annotationsData));
                changeVideo();
            }}

            // Horizontal Resizer logic
            const resizer = document.getElementById('resizer');
            const leftPanel = document.getElementById('leftPanel');
            let isResizing = false;

            resizer.addEventListener('mousedown', function(e) {{
                e.preventDefault();
                isResizing = true;
                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', stopResize);
            }});

            function handleMouseMove(e) {{
                if (!isResizing) return;
                const container = resizer.parentElement;
                const containerRect = container.getBoundingClientRect();
                let newLeftWidth = e.clientX - containerRect.left;

                // Enforce minimum width
                const minWidth = 250;
                if (newLeftWidth < minWidth) newLeftWidth = minWidth;
                if (newLeftWidth > container.clientWidth - minWidth) {{
                    newLeftWidth = container.clientWidth - minWidth;
                }}

                leftPanel.style.flexBasis = newLeftWidth + 'px';
            }}

            function stopResize() {{
                isResizing = false;
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', stopResize);
            }}

        }});

        function changeVideo() {{
            currentVideo = document.getElementById('videoSelect').value;
            const videoPath = videoPaths[currentVideo];
            const data = annotationsData[currentVideo] || {{}};

            // --- MODIFIED DURATION LOGIC STARTS HERE ---
            const videoId = data.youtube_id || currentVideo.split('.').slice(0, -1).join('.');
            const durationInfo = durationsData[videoId];
            const durationDisplay = document.getElementById('videoDuration');
            const timeSlider = document.getElementById('timeSlider');
            const timeDisplay = document.getElementById('timeDisplay');

            if (durationInfo) {{
                const hours = parseInt(durationInfo.hr, 10);
                const minutes = parseInt(durationInfo.min, 10);
                const seconds = parseInt(durationInfo.sec, 10);
                const milliseconds = parseInt(durationInfo.ms, 10);
                const totalSeconds = (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000);

                accurateDuration = totalSeconds; // Store the accurate duration

                // Use your existing formatTime function for consistency
                const formattedDuration = formatTime(totalSeconds);

                durationDisplay.textContent = formattedDuration;
                timeSlider.max = Math.ceil(totalSeconds * 1000);
                timeDisplay.textContent = '/ ' + formattedDuration; // Update time navigation display
            }} else {{
                accurateDuration = 0; // Reset if no duration info is found

                // Fallback if duration is not in the JSON file
                console.warn(`Duration info not found for ${{videoId}}. Waiting for video metadata.`);
                durationDisplay.textContent = "--:--";
                timeDisplay.textContent = '/ --:--'; // Also reset time navigation display
            }}
            acknowledgedIds.clear(); // Clear acknowledged on video change
            acknowledgedQueryIds.clear();
            summaryAcknowledged = false;
            summaryVideoOnlyAcknowledged = false;
            summaryAudioOnlyAcknowledged = false;
            (data.visual_queries || []).forEach(q => {{
                if (q.acknowledged) acknowledgedQueryIds.add(q.id);
            }});
            (data.audio_queries || []).forEach(q => {{
                if (q.acknowledged) acknowledgedQueryIds.add(q.id);
            }});

            // Re-create the video element to ensure old event listeners are cleared
            const videoContainer = document.querySelector('.video-container');
            videoContainer.innerHTML = `
                <video id="videoPlayer" class="video-player" controls>
                    <source id="videoSource" src="${{videoPath}}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            videoPlayer = document.getElementById('videoPlayer');
            videoPlayer.addEventListener('timeupdate', updateTimeFromVideo);
            videoPlayer.addEventListener('loadedmetadata', updateVideoDuration);
            videoPlayer.addEventListener('seeked', function() {{
                ignoreTimeUpdateUntilSeeked = false;
                if (!timeSliderDragging) {{
                    document.getElementById('timeSlider').value = Math.round(videoPlayer.currentTime * 1000);
                    updateDisplay();
                }}
            }});

            // FIX: Check if metadata is already loaded (race condition fix)
            if (videoPlayer.readyState >= 1) {{
                updateVideoDuration();
            }}

            videoPlayer.load();

            currentTime = 0;
            ignoreTimeUpdateUntilSeeked = false;
            document.getElementById('timeInput').value = '0';
            timeSlider.value = 0;

            buildAnnotationBoundaries();
            updateDisplay();
        }}


        function nextVideo() {{
            const videoSelect = document.getElementById('videoSelect');
            let currentIndex = videoSelect.selectedIndex;
            let nextIndex = currentIndex + 1;
            if (nextIndex >= videoSelect.options.length) {{
                nextIndex = 0; // Loop back to the first video
            }}
            videoSelect.selectedIndex = nextIndex;
            changeVideo();
        }}

        function updateTimeFromVideo() {{
            if (!videoPlayer) return;
            if (timeSliderDragging) return;
            if (ignoreTimeUpdateUntilSeeked) return;

            let reportedTime = videoPlayer.currentTime;

            // If we have an accurate duration from our file and the video is almost over,
            // snap to the accurate duration. This corrects for browser inaccuracies.

            if (
                accurateDuration > 0 &&
                Math.abs(reportedTime - accurateDuration) < 0.06
            ) {{
                // Corrects cases where browser overshoots by ~60ms.
                currentTime = accurateDuration;
            }}else {{
                currentTime = reportedTime;
            }}

            const sliderVal = Math.round(currentTime * 1000);
            document.getElementById('timeInput').value = currentTime.toFixed(3);
            document.getElementById('timeSlider').value = sliderVal;
            updateDisplay();
        }}

        function updateVideoDuration() {{
            const data = annotationsData[currentVideo] || {{}};
            const videoId = data.youtube_id || currentVideo.split('.').slice(0, -1).join('.');
            if (!durationsData[videoId]) {{
                if (!videoPlayer || !videoPlayer.duration || isNaN(videoPlayer.duration)) return;
                const duration = videoPlayer.duration;
                const formattedDuration = formatTime(duration); // Use formatTime for consistency

                document.getElementById('videoDuration').textContent = formattedDuration;
                document.getElementById('timeSlider').max = Math.ceil(duration * 1000);
                // Update time navigation display.
                document.getElementById('timeDisplay').textContent =
                    '/ ' + formattedDuration;
                buildAnnotationBoundaries();
            }}
        }}

        function setTime() {{
            const timeInput = document.getElementById('timeInput');
            seekToTimeSec(parseFloat(timeInput.value) || 0, 'time input go button');
        }}

        function updateTimeFromSlider() {{
            const slider = document.getElementById('timeSlider');
            seekToTimeSec(Number(slider.value) / 1000, 'inline oninput');
        }}



        function addNewAnnotation() {{
            const data = annotationsData[currentVideo];
            let endTime = currentTime + 1;
            if (endTime > videoPlayer.duration) {{
                endTime = videoPlayer.duration;
            }}
            const newAnn = {{
                id: `ann-${{data.annotations.length}}`,
                object_id: `event_${{String(data.annotations.length + 1).padStart(3, '0')}}`,
                start: currentTime,
                end: endTime,
                text: "New annotation",
                audio: "",
                start_str: formatTime(currentTime),
                end_str: formatTime(endTime),
                edited_desc: true,
                edited_audio: true,
                edited_start_time: true,
                edited_end_time: true
            }};

            data.annotations.push(newAnn);
            // Also add to original annotations to track changes
            originalAnnotations[currentVideo].annotations.push(JSON.parse(JSON.stringify(newAnn)));
            buildAnnotationBoundaries();
            updateDisplay();
            videoPlayer.currentTime = newAnn.start;
        }}

        function acknowledgeAnnotation(annId) {{
            const data = annotationsData[currentVideo];
            const annotation = data.annotations.find(a => a.id === annId);
            if (annotation && !validateAnnotation(annotation)) {{
                return;
            }}
            acknowledgedIds.add(annId);
            updateDisplay();
            saveAllAnnotations();
        }}

        function unacknowledgeAnnotation(annId) {{
            acknowledgedIds.delete(annId);
            updateDisplay();
        }}

        function getQueryById(queryId) {{
            const data = annotationsData[currentVideo];
            if (!data) return null;
            const allQueries = [...(data.visual_queries || []), ...(data.audio_queries || [])];
            return allQueries.find(q => q.id === queryId) || null;
        }}

        function updateQueryText(queryId, newText) {{
            const query = getQueryById(queryId);
            if (!query) return;
            query.text = newText.trim();
            query.edited_text = true;
        }}

        function updateQueryTime(queryId, field, newTimeStr) {{
            const query = getQueryById(queryId);
            if (!query) return;
            const newTimeInSeconds = timeStrToSeconds(newTimeStr);
            if (isNaN(newTimeInSeconds)) {{
                alert("Invalid time format. Please use MM:SS.mmm");
                updateDisplay();
                return;
            }}
            if (field === 'start') {{
                query.start = newTimeInSeconds;
                query.start_str = newTimeStr;
            }} else if (field === 'end') {{
                query.end = newTimeInSeconds;
                query.end_str = newTimeStr;
            }}
            query.edited_time = true;
            updateDisplay();
        }}

        function validateQuery(query) {{
            if (!query) return false;
            if (query.end < query.start) {{
                alert(`Error in query "${{query.text || ''}}": End time cannot be before start time.`);
                return false;
            }}
            const startTime = parseFloat(query.start.toFixed(3));
            const endTime = parseFloat(query.end.toFixed(3));
            if (startTime > accurateDuration || endTime > accurateDuration) {{
                alert(
                    `Error in query "${{query.text || ''}}": Start or end time cannot be beyond the video duration.`
                );
                return false;
            }}
            return true;
        }}

        function acknowledgeQuery(queryId) {{
            const query = getQueryById(queryId);
            if (!validateQuery(query)) return;
            acknowledgedQueryIds.add(queryId);
            if (query) query.acknowledged = true;
            updateDisplay();
            saveAllAnnotations();
        }}

        function unacknowledgeQuery(queryId) {{
            const query = getQueryById(queryId);
            acknowledgedQueryIds.delete(queryId);
            if (query) query.acknowledged = false;
            updateDisplay();
        }}

        function acknowledgeSummary() {{
            summaryAcknowledged = true;
            updateDisplay();
            saveAllAnnotations();
        }}

        function unacknowledgeSummary() {{
            summaryAcknowledged = false;
            updateDisplay();
        }}

        function acknowledgeSummaryVideoOnly() {{
            summaryVideoOnlyAcknowledged = true;
            updateDisplay();
            saveAllAnnotations();
        }}

        function unacknowledgeSummaryVideoOnly() {{
            summaryVideoOnlyAcknowledged = false;
            updateDisplay();
        }}

        function acknowledgeSummaryAudioOnly() {{
            summaryAudioOnlyAcknowledged = true;
            updateDisplay();
            saveAllAnnotations();
        }}

        function unacknowledgeSummaryAudioOnly() {{
            summaryAudioOnlyAcknowledged = false;
            updateDisplay();
        }}

        function validateAnnotation(ann) {{
            const duration = videoPlayer.duration;
            if (ann.end < ann.start) {{
                alert(`Error in annotation "${{ann.text}}": End time cannot be before start time.`);
                return false;
            }}

            // Round to 3 decimal places to match display format and avoid floating point issues
            const startTime = parseFloat(ann.start.toFixed(3));
            const endTime = parseFloat(ann.end.toFixed(3));
            const videoDuration = parseFloat(duration.toFixed(3));

            // Compare against value from the durations file, not video metadata.
            if (startTime > accurateDuration || endTime > accurateDuration) {{
                alert(
                    `Error in annotation "${{ann.text}}": Start or end time cannot be beyond the video duration` +
                    ` (${{formatTime(duration)}}).`
                );
                return false;
            }}
            return true;
        }}

        function updateDisplay() {{
            const data = annotationsData[currentVideo];
            if (!data) return;

            // Show/hide summary
            const summaryEl = document.getElementById('summary');
            const mainSummarySectionEl = document.getElementById('summaryMainSection');
            const videoOnlySummarySectionEl = document.getElementById('summaryVideoOnlySection');
            const audioOnlySummarySectionEl = document.getElementById('summaryAudioOnlySection');
            mainSummarySectionEl.style.display = summaryAcknowledged ? 'none' : 'block';
            videoOnlySummarySectionEl.style.display = summaryVideoOnlyAcknowledged ? 'none' : 'block';
            audioOnlySummarySectionEl.style.display = summaryAudioOnlyAcknowledged ? 'none' : 'block';
            const allSummariesAcknowledged =
                summaryAcknowledged &&
                summaryVideoOnlyAcknowledged &&
                summaryAudioOnlyAcknowledged;
            summaryEl.style.display = allSummariesAcknowledged ? 'none' : 'block';
            document.getElementById('summaryText').innerText = data.summary;
            document.getElementById('summaryVideoOnlyText').innerText = data.summary_video_only || '';
            document.getElementById('summaryAudioOnlyText').innerText = data.summary_audio_only || '';

            // Update time display
            document.getElementById('timeDisplay').textContent = `Time: ${{formatTime(currentTime)}}`;

            // Update stats
            document.getElementById('totalAnnotationsStat').textContent = data.annotations.length;

            const listEl = document.getElementById('annotationList');
            const queryListEl = document.getElementById('queryList');
            const ackListEl = document.getElementById('acknowledgedList');
            let ackListHTML = '';

            // Active and unacknowledged annotations
            const activeUnacknowledged = data.annotations.filter(ann =>
                currentTime >= ann.start && currentTime <= ann.end && !acknowledgedIds.has(ann.id)
            );

            // All acknowledged annotations for the current video
            const allAcknowledged = data.annotations.filter(ann => acknowledgedIds.has(ann.id));

            // Update active annotations count
            document.getElementById('activeAnnotationsStat').textContent = activeUnacknowledged.length +
                allAcknowledged.filter(ann => currentTime >= ann.start && currentTime <= ann.end).length;

            if (activeUnacknowledged.length === 0) {{
                listEl.innerHTML = '<div class="no-annotations">No unacknowledged annotations at this time.</div>';
            }} else {{
                listEl.innerHTML = activeUnacknowledged.map(ann =>
                    `<div class="annotation-item">
                        <div class="annotation-controls">
                            <button class="save-item-btn" onclick="saveAllAnnotations()">Save</button>
                            <button class="ack-btn" onclick="acknowledgeAnnotation('${{ann.id}}')">Ack</button>
                        </div>
                        <div class="annotation-time time-input-container">
                            [<input type="text" value="${{escapeAnnotationHtml(ann.start_str)}}"
                                    onchange="updateAnnotationTime('${{ann.id}}', 'start', this.value)"> -
                             <input type="text" value="${{escapeAnnotationHtml(ann.end_str)}}"
                                    onchange="updateAnnotationTime('${{ann.id}}', 'end', this.value)">]
                        </div>
                        <div><b>Description:</b></div>
                        <div class="annotation-text"
                             id="desc-${{ann.id}}"
                             contenteditable="true"
                             oninput="updateAnnotationText('${{ann.id}}', this.innerText)">
                            ${{escapeAnnotationHtml(ann.text)}}
                        </div>
                        <div><b>Audio:</b></div>
                        <textarea
                            class="annotation-audio"
                            id="audio-${{ann.id}}"
                            rows="2"
                            oninput="updateAnnotationAudio('${{ann.id}}', this.value)"
                        >${{escapeAnnotationHtml(ann.audio || '')}}</textarea>
                    </div>`
                ).join('');
            }}

            const visualQueries = Array.isArray(data.visual_queries) ? data.visual_queries : [];
            const audioQueries = Array.isArray(data.audio_queries) ? data.audio_queries : [];
            const activeVisualQueries = visualQueries.filter(
                q => currentTime >= q.start && currentTime <= q.end && !acknowledgedQueryIds.has(q.id)
            );
            const activeAudioQueries = audioQueries.filter(
                q => currentTime >= q.start && currentTime <= q.end && !acknowledgedQueryIds.has(q.id)
            );
            const allAcknowledgedQueries = [...visualQueries, ...audioQueries].filter(
                q => acknowledgedQueryIds.has(q.id)
            );
            const renderQueryItems = (queries) => {{
                if (!queries.length) return '<div class="no-annotations">No queries.</div>';
                return queries.map(q => `
                    <div class="query-item">
                        <div class="annotation-controls">
                            <button class="save-item-btn" onclick="saveAllAnnotations()">Save</button>
                            <button class="ack-btn" onclick="acknowledgeQuery('${{q.id}}')">Ack</button>
                        </div>
                        <div class="query-time time-input-container">
                            [<input type="text" value="${{escapeAnnotationHtml(q.start_str || '')}}"
                                    onchange="updateQueryTime('${{q.id}}', 'start', this.value)"> -
                             <input type="text" value="${{escapeAnnotationHtml(q.end_str || '')}}"
                                    onchange="updateQueryTime('${{q.id}}', 'end', this.value)">]
                        </div>
                        <div class="query-text"
                             contenteditable="true"
                             oninput="updateQueryText('${{q.id}}', this.innerText)">
                            ${{escapeAnnotationHtml(q.text || '')}}
                        </div>
                    </div>
                `).join('');
            }};
            queryListEl.innerHTML = `
                <div class="query-group">
                    <div><b>Visual Queries (Active)</b></div>
                    ${{renderQueryItems(activeVisualQueries)}}
                </div>
                <div class="query-group">
                    <div><b>Audio Queries (Active)</b></div>
                    ${{renderQueryItems(activeAudioQueries)}}
                </div>
            `;

            ackListHTML += allAcknowledgedQueries.map(q =>
                `<div class="query-item acknowledged-item">
                    <div class="annotation-controls">
                        <button class="unack-btn" onclick="unacknowledgeQuery('${{q.id}}')">Un-Ack</button>
                    </div>
                    <div class="query-time">
                        [${{escapeAnnotationHtml(q.start_str || '')}} -
                        ${{escapeAnnotationHtml(q.end_str || '')}}]
                    </div>
                    <div class="query-text">${{escapeAnnotationHtml(q.text || '')}}</div>
                </div>`
            ).join('');

            if (summaryAcknowledged) {{
                ackListHTML += `<div class="annotation-item acknowledged-item">
                        <div class="annotation-controls">
                             <button class="unack-btn" onclick="unacknowledgeSummary()">Un-Ack Summary</button>
                        </div>
                        <b>Acknowledged Summary:</b>
                        <div class="annotation-text">${{escapeAnnotationHtml(data.summary)}}</div>
                    </div>`;
            }}
            if (summaryVideoOnlyAcknowledged) {{
                ackListHTML += `<div class="annotation-item acknowledged-item">
                        <div class="annotation-controls">
                             <button class="unack-btn"
                                     onclick="unacknowledgeSummaryVideoOnly()">
                                 Un-Ack Video-only Summary
                             </button>
                        </div>
                        <b>Acknowledged Video-only Summary:</b>
                        <div class="annotation-text">${{escapeAnnotationHtml(data.summary_video_only || '')}}</div>
                    </div>`;
            }}
            if (summaryAudioOnlyAcknowledged) {{
                ackListHTML += `<div class="annotation-item acknowledged-item">
                        <div class="annotation-controls">
                             <button class="unack-btn"
                                     onclick="unacknowledgeSummaryAudioOnly()">
                                 Un-Ack Audio-only Summary
                             </button>
                        </div>
                        <b>Acknowledged Audio-only Summary:</b>
                        <div class="annotation-text">${{escapeAnnotationHtml(data.summary_audio_only || '')}}</div>
                    </div>`;
            }}

            ackListHTML += allAcknowledged.map(ann =>
                `<div class="annotation-item acknowledged-item">
                    <div class="annotation-controls">
                         <button class="unack-btn" onclick="unacknowledgeAnnotation('${{ann.id}}')">Un-Ack</button>
                    </div>
                    <div class="annotation-time">[${{ann.start_str}} - ${{ann.end_str}}]</div>
                    <div><b>Description:</b></div>
                    <div class="annotation-text">${{escapeAnnotationHtml(ann.text)}}</div>
                    <div><b>Audio:</b></div>
                    <div class="annotation-audio">${{escapeAnnotationHtml(ann.audio || '')}}</div>
                </div>`
            ).join('');

            if (ackListHTML === '') {{
                ackListEl.innerHTML = '<div class="no-annotations">No acknowledged items yet.</div>';
            }} else {{
                ackListEl.innerHTML = ackListHTML;
            }}

            updateAnnotationSliderPosition();
        }}

        function updateAnnotationText(annId, newText) {{
            const data = annotationsData[currentVideo];
            const annotation = data.annotations.find(a => a.id === annId);
            const originalAnn = originalAnnotations[currentVideo].annotations.find(a => a.id === annId);
            if (annotation && originalAnn) {{
                const trimmedText = newText.trim();
                annotation.text = trimmedText;
                annotation.edited_desc = trimmedText !== (originalAnn.text || "").trim();
            }}
        }}

        function updateAnnotationAudio(annId, newText) {{
            const data = annotationsData[currentVideo];
            const annotation = data.annotations.find(a => a.id === annId);
            const originalAnn = originalAnnotations[currentVideo].annotations.find(a => a.id === annId);
            if (annotation && originalAnn) {{
                const trimmedText = newText.trim();
                annotation.audio = trimmedText;
                annotation.edited_audio = trimmedText !== (originalAnn.audio || "").trim();
            }}
        }}

        function updateAnnotationTime(annId, field, newTimeStr) {{
            const data = annotationsData[currentVideo];
            const annotation = data.annotations.find(a => a.id === annId);
            const originalAnn = originalAnnotations[currentVideo].annotations.find(a => a.id === annId);

            if (annotation && originalAnn) {{
                const newTimeInSeconds = timeStrToSeconds(newTimeStr);
                if (isNaN(newTimeInSeconds)) {{
                    alert("Invalid time format. Please use MM:SS.mmm");
                    updateDisplay(); // Re-render to show original value
                    return;
                }}

                if (field === 'start') {{
                    annotation.start = newTimeInSeconds;
                    annotation.start_str = newTimeStr;
                    annotation.edited_start_time = newTimeStr !== originalAnn.start_str;
                }} else if (field === 'end') {{
                    annotation.end = newTimeInSeconds;
                    annotation.end_str = newTimeStr;
                    annotation.edited_end_time = newTimeStr !== originalAnn.end_str;
                }}
                buildAnnotationBoundaries();
            }}
        }}

        function timeStrToSeconds(timeStr) {{
            const parts = timeStr.split(':');
            if (parts.length !== 2) return NaN;

            const secondsAndMs = parts[1].split('.');
            if (secondsAndMs.length !== 2) return NaN;

            const minutes = parseInt(parts[0], 10);
            const seconds = parseInt(secondsAndMs[0], 10);
            const milliseconds = parseInt(secondsAndMs[1], 10);

            if (isNaN(minutes) || isNaN(seconds) || isNaN(milliseconds)) return NaN;

            return minutes * 60 + seconds + milliseconds / 1000.0;
        }}


        function updateSummaryText(newText) {{
            const data = annotationsData[currentVideo];
            const originalData = originalAnnotations[currentVideo];
            if (data && originalData) {{
                const trimmedText = newText.trim();
                data.summary = trimmedText;
                data.edited_summary = trimmedText !== (originalData.summary || "").trim();
            }}
        }}

        function updateSummaryVideoOnlyText(newText) {{
            const data = annotationsData[currentVideo];
            const originalData = originalAnnotations[currentVideo];
            if (data && originalData) {{
                const trimmedText = newText.trim();
                data.summary_video_only = trimmedText;
                data.edited_summary_video_only = trimmedText !== (originalData.summary_video_only || "").trim();
            }}
        }}

        function updateSummaryAudioOnlyText(newText) {{
            const data = annotationsData[currentVideo];
            const originalData = originalAnnotations[currentVideo];
            if (data && originalData) {{
                const trimmedText = newText.trim();
                data.summary_audio_only = trimmedText;
                data.edited_summary_audio_only = trimmedText !== (originalData.summary_audio_only || "").trim();
            }}
        }}

        async function saveAllAnnotations() {{
            const data = annotationsData[currentVideo];
            for (const ann of data.annotations) {{
                if (!validateAnnotation(ann)) {{
                    return;
                }}
            }}
            for (const query of [...(data.visual_queries || []), ...(data.audio_queries || [])]) {{
                if (!validateQuery(query)) {{
                    return;
                }}
            }}

            const statusEl = document.getElementById('saveStatus');
            statusEl.textContent = 'Saving...';
            statusEl.style.color = '#ff9800';

            const payload = {{
                youtube_id: data.youtube_id,
                summary: data.summary,
                summary_video_only: data.summary_video_only || "",
                summary_audio_only: data.summary_audio_only || "",
                summary_multimodal: data.summary || "",
                non_english: data.non_english || false,
                edited_summary: data.edited_summary ? "Yes" : "No",
                edited_summary_video_only: data.edited_summary_video_only ? "Yes" : "No",
                edited_summary_audio_only: data.edited_summary_audio_only ? "Yes" : "No",
                video_descriptions: data.annotations.map(ann => {{
                    const newAnn = {{
                        object_id: ann.object_id || "",
                        t0: ann.start_str,
                        t1: ann.end_str,
                        text: ann.text,
                        audio: ann.audio || "",
                        edited_desc: ann.edited_desc ? "Yes" : "No",
                        edited_audio: ann.edited_audio ? "Yes" : "No",
                        edited_start_time: ann.edited_start_time ? "Yes" : "No",
                        edited_end_time: ann.edited_end_time ? "Yes" : "No"
                    }};
                    return newAnn;
                }}),
                visual_queries: (data.visual_queries || []).map(q => {{
                    return {{
                        t0: q.start_str,
                        t1: q.end_str,
                        text: q.text || "",
                        acknowledged: acknowledgedQueryIds.has(q.id),
                        edited_text: q.edited_text ? "Yes" : "No",
                        edited_time: q.edited_time ? "Yes" : "No"
                    }};
                }}),
                audio_queries: (data.audio_queries || []).map(q => {{
                    return {{
                        t0: q.start_str,
                        t1: q.end_str,
                        text: q.text || "",
                        acknowledged: acknowledgedQueryIds.has(q.id),
                        edited_text: q.edited_text ? "Yes" : "No",
                        edited_time: q.edited_time ? "Yes" : "No"
                    }};
                }}),
                annotations: data.annotations.map(ann => {{
                    const newAnn = {{
                        t0: ann.start_str,
                        t1: ann.end_str,
                        desc: ann.text,
                        audio: ann.audio || "",
                        edited_desc: ann.edited_desc ? "Yes" : "No",
                        edited_audio: ann.edited_audio ? "Yes" : "No",
                        edited_start_time: ann.edited_start_time ? "Yes" : "No",
                        edited_end_time: ann.edited_end_time ? "Yes" : "No"
                    }};
                    return newAnn;
                }})
            }};

            try {{
                const response = await fetch('/save-annotations', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                if (response.ok) {{
                    statusEl.textContent = '✅ Saved successfully!';
                    statusEl.style.color = '#4CAF50';
                }} else {{
                    const error = await response.text();
                    statusEl.textContent = `❌ Error: ${{error}}`;
                    statusEl.style.color = '#f44336';
                }}
            }} catch (e) {{
                statusEl.textContent = `❌ Network Error: ${{e.message}}`;
                statusEl.style.color = '#f44336';
            }}

            setTimeout(() => {{ statusEl.textContent = ''; }}, 5000);
        }}

        function buildAnnotationBoundaries() {{
            const data = annotationsData[currentVideo];
            annotationBoundaries = [];
            if (data && data.annotations.length > 0) {{
                const boundaries = new Set([0]);
                data.annotations.forEach(ann => {{
                    boundaries.add(ann.start);
                    boundaries.add(ann.end);
                }});
                if (videoPlayer && videoPlayer.duration) {{
                    boundaries.add(videoPlayer.duration);
                }}
                annotationBoundaries = [...boundaries].sort((a, b) => a - b);
            }}
            const slider = document.getElementById('annotationSlider');
            if (slider) {{
                slider.max = annotationBoundaries.length > 0 ? annotationBoundaries.length - 1 : 1;
                slider.value = 0;
            }}
        }}

        function updateFromAnnotationSlider() {{
            const slider = document.getElementById('annotationSlider');
            if (!slider) return;
            const index = parseInt(slider.value);
            if (annotationBoundaries.length > 0 && index < annotationBoundaries.length) {{
                currentTime = annotationBoundaries[index];
                if (videoPlayer) {{
                    videoPlayer.currentTime = currentTime;
                    videoPlayer.pause();
                }}
                updateDisplay();
            }}
        }}

        function goToPreviousAnnotation() {{
            const slider = document.getElementById('annotationSlider');
            if (!slider || annotationBoundaries.length === 0) return;
            slider.value = Math.max(0, parseInt(slider.value) - 1);
            updateFromAnnotationSlider();
        }}

        function goToNextAnnotation() {{
            const slider = document.getElementById('annotationSlider');
            if (!slider || annotationBoundaries.length === 0) return;
            slider.value = Math.min(annotationBoundaries.length - 1, parseInt(slider.value) + 1);
            updateFromAnnotationSlider();
        }}

        function updateAnnotationSliderPosition() {{
            const slider = document.getElementById('annotationSlider');
            if (!slider || annotationBoundaries.length === 0) return;
            let closestIndex = 0;
            let minDistance = Infinity;
            for (let i = 0; i < annotationBoundaries.length; i++) {{
                const distance = Math.abs(annotationBoundaries[i] - currentTime);
                if (distance < minDistance) {{
                    minDistance = distance;
                    closestIndex = i;
                }}
            }}
            slider.value = closestIndex;
            updateAnnotationInfo(closestIndex);
        }}

        function updateAnnotationInfo(index) {{
            const infoEl = document.getElementById('annotationInfo');
            if (!infoEl) return;
            if (annotationBoundaries.length > 0 && index < annotationBoundaries.length) {{
                const time = annotationBoundaries[index];
                infoEl.textContent = `Jump to: ${{formatTime(time)}} (${{index + 1}}/${{annotationBoundaries.length}})`;
            }} else {{
                infoEl.textContent = "No annotation timeline.";
            }}
        }}

        function formatTime(seconds) {{
            if (isNaN(seconds)) return "00:00.000";
            const minutes = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${{String(minutes).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}` +
                   `.${{String(ms).padStart(3, '0')}}`;
        }}

        function secondsToTimeObject(totalSeconds) {{
            const hours = Math.floor(totalSeconds / 3600);
            totalSeconds %= 3600;
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = Math.floor(totalSeconds % 60);
            const milliseconds = Math.round((totalSeconds % 1) * 1000);

            return {{
                hr: String(hours).padStart(2, '0'),
                min: String(minutes).padStart(2, '0'),
                sec: String(seconds).padStart(2, '0'),
                ms: String(milliseconds).padStart(3, '0')
            }};
        }}
    </script>
</body>
</html>"""
    return html_content
