"""
data/dataset.py — OpenVID-1M dataset and motion complexity stratification.

Paper reference: Section 4.1 — "We conducted our experiments using the OpenVID-1M
dataset, a large-scale open-domain video dataset containing diverse motion patterns."

Motion complexity stratification (Sec 4.2): prompts are classified as 'simple'
(dominant motion ≈ single SIM(2) transform) or 'complex' (otherwise), using
an LLM rubric (GPT-5 in the paper).

WARNING: The exact LLM rubric is not published (SIR conf 0.70).
MotionComplexityFilter provides a keyword-based fallback and an LLM subclass.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Simple motion keywords (Sec 4.2 rubric: single rigid transform)
_SIMPLE_KEYWORDS = [
    "translat", "slide", "move", "drift",
    "rotat", "spin", "turn", "clockwise", "counterclockwise",
    "zoom", "scale", "approach", "recede", "far away to nearby",
]


class OpenVIDDataset(Dataset):
    """PyTorch Dataset wrapper for the OpenVID-1M video dataset.

    Paper reference: Section 4.1.

    Expected directory layout::

        {data_root}/
            videos/
                video_00000001.mp4
                ...
            metadata.json      # [{"video": "...", "prompt": "...", ...}, ...]

    Args:
        data_root:    Path to OpenVID-1M root directory.
        split:        Dataset split: 'train' or 'val'.
        T:            Number of frames to sample per clip.
        H, W:         Spatial resolution to resize to.
        motion_split: If not None, filter to 'simple' or 'complex' prompts.
        max_samples:  If set, truncate dataset (useful for debug/quick runs).
        classifier:   Optional MotionComplexityFilter for split filtering.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        T: int = 16,
        H: int = 256,
        W: int = 256,
        motion_split: Optional[Literal["all", "simple", "complex"]] = "all",
        max_samples: Optional[int] = None,
        classifier: Optional["MotionComplexityFilter"] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.T = T
        self.H = H
        self.W = W
        self.motion_split = motion_split

        # Load metadata
        meta_path = self.data_root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {meta_path}. "
                "See data/README_data.md for download instructions."
            )

        with open(meta_path) as f:
            all_meta: List[Dict] = json.load(f)

        # Filter by split key if present
        if split in ("train", "val"):
            all_meta = [m for m in all_meta if m.get("split", "train") == split]

        # Filter by motion complexity
        if motion_split and motion_split != "all":
            if classifier is None:
                classifier = KeywordMotionClassifier()
            logger.info(f"Filtering to '{motion_split}' motion prompts...")
            all_meta = [
                m for m in all_meta
                if classifier.classify(m.get("prompt", "")) == motion_split
            ]
            logger.info(f"After filter: {len(all_meta)} samples")

        # Optionally truncate
        if max_samples is not None:
            all_meta = all_meta[:max_samples]

        self.metadata = all_meta
        logger.info(f"OpenVIDDataset ({split}): {len(self)} samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return a dict with 'video' tensor [C, T, H, W] and 'prompt' string."""
        item = self.metadata[idx]
        video_path = self.data_root / "videos" / item["video"]

        video = self._load_video(str(video_path))  # [C, T, H, W]

        return {
            "video":  video,
            "prompt": item.get("prompt", ""),
            "path":   str(video_path),
        }

    def _load_video(self, path: str) -> Tensor:
        """Load video as float32 tensor [C, T, H, W] in [0, 1]."""
        try:
            import torchvision.io as tvio
            from torchvision.transforms.functional import resize
        except ImportError:
            raise ImportError("torchvision is required for video loading.")

        # Read all frames
        frames, _, _ = tvio.read_video(path, pts_unit="sec", output_format="TCHW")
        # frames: [T_all, C, H_orig, W_orig], uint8

        total_frames = frames.shape[0]
        if total_frames == 0:
            raise RuntimeError(f"Could not read any frames from {path}")

        # Uniform temporal sampling
        if total_frames >= self.T:
            indices = torch.linspace(0, total_frames - 1, self.T).long()
        else:
            # Repeat last frame if video is shorter than T
            indices = torch.cat([
                torch.arange(total_frames),
                torch.full((self.T - total_frames,), total_frames - 1),
            ])
        frames = frames[indices]  # [T, C, H_orig, W_orig]

        # Spatial resize
        frames = resize(frames, [self.H, self.W])  # [T, C, H, W]

        # Rearrange to [C, T, H, W] and normalise to [0, 1]
        video = frames.permute(1, 0, 2, 3).float() / 255.0

        return video


# ---------------------------------------------------------------------------
# Motion Complexity Classifiers
# ---------------------------------------------------------------------------

class MotionComplexityFilter(ABC):
    """Abstract base class for motion complexity classification.

    Paper reference: Section 4.2 — LLM (GPT-5) rubric for 'simple' vs 'complex'.
    WARNING: Exact rubric not published (SIR conf 0.70). Subclass and override
    classify() with your preferred implementation.
    """

    @abstractmethod
    def classify(self, prompt: str) -> Literal["simple", "complex"]:
        """Classify a text prompt as 'simple' or 'complex' motion."""
        ...


class KeywordMotionClassifier(MotionComplexityFilter):
    """Keyword-based fallback classifier (no LLM required).

    A prompt is 'simple' if it contains at least one keyword associated with
    a single rigid transform (translation, rotation, scaling).
    Otherwise it is 'complex'.

    This is a heuristic approximation of the LLM rubric used in the paper.
    Replace with LLMMotionClassifier for higher fidelity.
    """

    def classify(self, prompt: str) -> Literal["simple", "complex"]:
        prompt_lower = prompt.lower()
        for kw in _SIMPLE_KEYWORDS:
            if kw in prompt_lower:
                return "simple"
        return "complex"

    def __repr__(self) -> str:
        return "KeywordMotionClassifier()"


class LLMMotionClassifier(MotionComplexityFilter):
    """LLM-based classifier approximating the paper's GPT-5 rubric.

    Paper reference: Section 4.2 — "we stratify test prompts into simple vs.
    complex motion using an LLM (GPT-5) following our rubric: a prompt is simple
    when the dominant motion is well-approximated by a single rigid transform."

    WARNING: Exact rubric/system prompt not published (SIR conf 0.70).
    The system prompt below is our best approximation.

    Args:
        api_key:    OpenAI API key (or set OPENAI_API_KEY env var).
        model:      LLM model name. Paper used GPT-5 — use latest available.
    """

    SYSTEM_PROMPT = (
        "You are a motion analysis assistant. Given a video description, classify "
        "the dominant motion as 'simple' or 'complex'.\n\n"
        "SIMPLE: The primary motion can be well-approximated by a SINGLE rigid "
        "transformation — one of: (a) constant-velocity translation, "
        "(b) in-plane rotation, or (c) uniform scaling/zoom.\n\n"
        "COMPLEX: The motion involves multiple, non-rigid, or articulated motions "
        "that cannot be captured by a single rigid transformation.\n\n"
        "Respond with exactly one word: 'simple' or 'complex'."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        self.model = model
        self._api_key = api_key

    def classify(self, prompt: str) -> Literal["simple", "complex"]:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai to use LLMMotionClassifier")

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().lower()
        if "simple" in result:
            return "simple"
        return "complex"

    def __repr__(self) -> str:
        return f"LLMMotionClassifier(model='{self.model}')"
