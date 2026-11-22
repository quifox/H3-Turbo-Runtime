# ComfyUI Integration: Native Audiovisual Interface for H3 Turbo

[Home](../../README.md) · [English Home](../../README.en.md) · [CozyUI Integration](../cozyui/README.md)

The default workflow runs a **5-second, 832×480, 24 FPS, 4-step, Fast-path FL2VA** configuration without requiring external media. Submitting a text prompt generates a native synchronized audiovisual MP4 on consumer hardware.

---

## 1. Ready-to-Use Workflows

| Workflow File | Purpose |
|---|---|
| [`complete-5s-workflow.json`](complete-5s-workflow.json) | Full canvas workflow containing generation settings, optional media condition nodes, and the final AV exporter. |
| [`complete-5s-api.json`](complete-5s-api.json) | ComfyUI API JSON format ready for programmatic `POST /prompt` submissions. |
| [`text-to-video.json`](text-to-video.json) | Clean standalone text-to-audiovisual workflow. |
| [`first-last-frame.json`](first-last-frame.json) | Dual keyframe guidance (first frame, last frame, or both). |
| [`timeline-guides.json`](timeline-guides.json) | Arbitrary temporal keyframes anchored along the video duration. |
| [`reference-to-video.json`](reference-to-video.json) | Multimodal Ref2VA workflow with character images, reference video, and voice tracks. |

### Quick Installation

```bash
# Link the custom node to your ComfyUI installation
python deploy.py configure-comfy --comfy /path/to/ComfyUI
export H3_RUNTIME_MODELS=/path/to/weights
```

Drag and drop `complete-5s-workflow.json` into the ComfyUI web interface, adjust your prompt, and click Queue Prompt. Input images and audio should be placed in ComfyUI's `input/` directory.

---

## 2. Task Modes and Model Variant Selection

| Task Mode | Target Variant | Input Specification |
|---|---|---|
| **Text to Video (T2VA)** | FL2VA | Text prompt only |
| **First Frame to Video (I2VA)** | FL2VA | Optional start image |
| **Last Frame to Video (L2VA)** | FL2VA | Optional end image |
| **First and Last Frame (FL2VA)** | FL2VA | Both start and end images |
| **Timeline Guides** | FL2VA | Arbitrary intermediate image, video, or audio guide anchors |
| **Reference to Video (Ref2VA)** | Ref2VA | Identity image, video, and audio references combined with temporal guides |

*Note:* Setting `variant=auto` automatically switches between FL2VA and Ref2VA based on whether reference media inputs are present.

---

## 3. Advanced Parameters and JSON Guides

### Timeline Guides JSON Format
```json
[
  {"at_seconds": 2.0, "image": "middle.png"},
  {"frame_idx": 72, "video": "short-clip.mp4", "audio": "guide.wav"},
  {"frame_idx": -1, "image": "ending.png"}
]
```

### Multimodal References JSON Format
```json
{
  "images": ["character.png", "angle2.png"],
  "videos": [{"path": "reference.mp4", "audio": true}],
  "audios": ["voice.wav"],
  "image_size": "max"
}
```

Prompts can reference inputs using `<Picture 1>`, `<Video 1>`, or `<Audio 1>`. Chinese dialogues should use the native `<d>[Chinese]... </d>` tag structure.

Generated video files are saved to ComfyUI's `output/h3-turbo/` directory as standard H.264 / AAC MP4 files.
