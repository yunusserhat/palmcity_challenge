# Data and upstream model terms

The project code is GPLv3. Data, checkpoints and dependencies retain their original licenses. Downloading a checkpoint does not relicense it as GPLv3. This release distributes training code and small measurement records, and obtains data and weights from the official sources.

PalmCity is supplied for research and challenge use. Consult the [dataset repository](https://github.com/PalmCityDataset/palmcity) and [challenge conditions](https://www.codabench.org/competitions/18192/) before use. The experiment used pretrained weights under the author’s confirmed challenge permission. This is not a claim that all third party artifacts have a GPL license.

| Upstream checkpoint | Recorded terms | Source |
|---|---|---|
| `smp-hub/resnet50.imagenet` | BSD-3-Clause code, artifact card other | [license or model card](https://github.com/pytorch/vision/blob/main/LICENSE) |
| `smp-hub/mit_b2.imagenet` | NVIDIA research/noncommercial terms | [license or model card](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |
| `nvidia/segformer-b5-finetuned-ade-640-640` | NVIDIA research/noncommercial terms | [license or model card](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |
| `openmmlab/upernet-convnext-large` | MIT declared by official model card | [license or model card](https://huggingface.co/openmmlab/upernet-convnext-large) |
| `openmmlab/upernet-swin-large` | MIT declared by official model card | [license or model card](https://huggingface.co/openmmlab/upernet-swin-large) |
| `facebook/mask2former-swin-large-ade-semantic` | CC-BY-NC-4.0 artifact, MIT upstream code | [license or model card](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#license) |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | DINOv3 custom license; gated access requires the requesting user’s own authorization | [license or model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/main/LICENSE.md) |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 custom license; gated access requires the requesting user’s own authorization | [license or model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) |
| `tue-mps/eomt-dinov3-ade-semantic-large-512` | MIT model-card declaration; DINOv3 underlying pretrained license also applies | [license or model card](https://huggingface.co/tue-mps/eomt-dinov3-ade-semantic-large-512) |

The DINOv3 base and large encoder repositories require access authorization. Obtain that authorization from Meta and authenticate with your own Hugging Face account if reproducing those candidates. The selected EoMT ADE checkpoint is publicly downloadable, but its underlying DINOv3 terms still apply. Never place a token in a config, command history or versioned file.

The final three PalmCity EoMT checkpoints are available separately in the [PalmCity model release](https://huggingface.co/yunusserhat/palmcity-eomt-dinov3-large). The derived weights are distributed under the DINOv3 License with the complete agreement and EoMT attribution supplied in that model repository. The [inference guide](inference.md) uses these weights directly. Their license is separate from the GPLv3 project code.

Dependency implementations are installed from their pinned upstream packages rather than copied into this repository. PyTorch and torchvision use BSD style licenses. Transformers uses Apache 2.0. timm and segmentation_models_pytorch use permissive upstream licenses. Package metadata and upstream license files are authoritative for their respective versions.
