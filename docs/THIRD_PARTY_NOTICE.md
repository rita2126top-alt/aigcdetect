# Sources, provenance, and third-party notices

## Original PPM-CLIP snapshot

Repository: <https://github.com/bandaidssssss/PPM_CLIP>

Pinned commit: `09d05b9fc4be6a2b079356bbf337a05e193db0be`.

The 47 original files are preserved byte-for-byte, including files that the upstream repository itself tracked under `__pycache__`. Their SHA256 digests are in `provenance/upstream.json`. New implementation code imports the original CLIP implementation and DCT patch selector rather than pretending those components were newly invented.

The imported upstream tree did not contain a standalone LICENSE file. This extension does not invent, replace, or broaden the upstream authors' permission terms. Review the upstream source and associated notices before broader redistribution or use. The user-provided method description is the specification for the new extension, not an independently verified published paper.

Original OpenAI CLIP project and model source: <https://github.com/openai/CLIP>. The local default checkpoint URL and expected SHA256 come from the model registry bundled in the pinned PPM-CLIP snapshot, `clip/clip.py`.

## Dataset source

Official GenImage repository / download instructions:
<https://github.com/GenImage-Dataset/GenImage/blob/main/Readme.md>

The default Google Drive folder is the publisher's listed folder, not an asserted equivalent third-party mirror. Dataset availability, quotas, access and upstream data licensing are outside the software's control. The dataset is not included in the Git repository or deliverable archive. GenImage's real-image source and any underlying dataset access terms must be considered by the operator.

## Environment references

Verified installation combination used for code compatibility: torch 2.10.0 and torchvision 0.25.0. This is a version pin, not a claim that these are the newest packages.

PyTorch official version/wheel matrix: <https://pytorch.org/get-started/previous-versions/>

Miniconda official installation guidance: <https://www.anaconda.com/docs/getting-started/installation>

Miniconda distribution index/checksums: <https://repo.anaconda.com/miniconda/>

## Experimental claims

No pretrained CADP detector weights, real-dataset training scores, GPU performance numbers or paper acceptance claims are supplied. The toy random-image fixtures are generated solely for automated software tests, clearly marked as such, and kept outside the source tree.
