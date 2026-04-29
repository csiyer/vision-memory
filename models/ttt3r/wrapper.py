from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch


@dataclass
class FrameReadout:
    frame_index: int
    beta_t: torch.Tensor
    beta_mean: torch.Tensor
    delta_s_norm: torch.Tensor
    mean_conf_self: torch.Tensor
    conf_self: torch.Tensor
    output: dict[str, Any]


@dataclass
class TTT3RMemoryState:
    state_feat: torch.Tensor
    state_pos: torch.Tensor
    init_state_feat: torch.Tensor
    mem: torch.Tensor
    init_mem: torch.Tensor
    step_index: int = 0

    def clone(self) -> "TTT3RMemoryState":
        return TTT3RMemoryState(
            state_feat=self.state_feat.clone(),
            state_pos=self.state_pos.clone(),
            init_state_feat=self.init_state_feat.clone(),
            mem=self.mem.clone(),
            init_mem=self.init_mem.clone(),
            step_index=self.step_index,
        )


class TTT3RMemoryWrapper:
    """Thin wrapper that exposes TTT3R memory readouts on image sequences.

    The three primary probes are:
    - beta_t: per-state-token update gate
    - delta_s_norm: norm of the accepted write delta S_t = beta_t * g_t
    - mean_conf_self: average conf_self over all image pixels
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
        image_size: int = 512,
        verbose: bool = False,
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.model_path = Path(model_path) if model_path else self.root / "src" / "cut3r_512_dpt_4_64.pth"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.verbose = verbose

        src_path = self.model_path.parent
        src_path_str = str(src_path)
        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)

        from dust3r.model import ARCroco3DStereo  # pylint: disable=import-error

        self.model = ARCroco3DStereo.from_pretrained(str(self.model_path)).to(self.device)
        self.model.config.model_update_type = "ttt3r"
        self.model.eval()

        from dust3r.utils.device import to_gpu  # pylint: disable=import-error
        from dust3r.utils.image import load_images  # pylint: disable=import-error
        from einops import rearrange

        self._to_gpu = to_gpu
        self._load_images = load_images
        self._rearrange = rearrange

    def prepare_views(
        self,
        image_paths: Sequence[str | Path],
        update: bool = True,
        reset_interval: int | None = None,
    ) -> list[dict[str, Any]]:
        image_paths = [str(path) for path in image_paths]
        images = self._load_images(image_paths, size=self.image_size, verbose=self.verbose)
        views: list[dict[str, Any]] = []
        interval = reset_interval or 10**9
        for i, image in enumerate(images):
            view = {
                "img": image["img"],
                "ray_map": torch.full(
                    (image["img"].shape[0], 6, image["img"].shape[-2], image["img"].shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(update).unsqueeze(0),
                "reset": torch.tensor((i + 1) % interval == 0).unsqueeze(0),
            }
            views.append(view)
        return views

    @torch.no_grad()
    def run_sequence(
        self,
        image_paths: Sequence[str | Path],
        update: bool = True,
        reset_interval: int | None = None,
        keep_outputs: bool = True,
    ) -> list[FrameReadout]:
        return self.run_views(
            self.prepare_views(image_paths=image_paths, update=update, reset_interval=reset_interval),
            keep_outputs=keep_outputs,
        )

    @torch.no_grad()
    def run_views(
        self,
        views: Sequence[dict[str, Any]],
        keep_outputs: bool = True,
    ) -> list[FrameReadout]:
        frame_readouts, _ = self.study_views(views, keep_outputs=keep_outputs)
        return frame_readouts

    @torch.no_grad()
    def study_views(
        self,
        views: Sequence[dict[str, Any]],
        keep_outputs: bool = True,
    ) -> tuple[list[FrameReadout], TTT3RMemoryState]:
        frame_readouts: list[FrameReadout] = []
        state: TTT3RMemoryState | None = None
        for raw_view in views:
            readout, state = self.step(raw_view, state=state, keep_output=keep_outputs)
            frame_readouts.append(readout)
        assert state is not None
        return frame_readouts, state

    @torch.no_grad()
    def encode_views(self, views: Sequence[dict[str, Any]]) -> TTT3RMemoryState:
        _, state = self.study_views(views, keep_outputs=False)
        return state

    @torch.no_grad()
    def encode_sequence(
        self,
        image_paths: Sequence[str | Path],
        update: bool = True,
        reset_interval: int | None = None,
    ) -> TTT3RMemoryState:
        views = self.prepare_views(image_paths=image_paths, update=update, reset_interval=reset_interval)
        return self.encode_views(views)

    @torch.no_grad()
    def probe_view(
        self,
        view: dict[str, Any],
        state: TTT3RMemoryState,
        keep_output: bool = True,
    ) -> FrameReadout:
        readout, _ = self.step(view, state=state, keep_output=keep_output)
        return readout

    @torch.no_grad()
    def probe_image(
        self,
        image_path: str | Path,
        state: TTT3RMemoryState,
        keep_output: bool = True,
    ) -> FrameReadout:
        view = self.prepare_views([image_path], update=True)[0]
        return self.probe_view(view, state=state, keep_output=keep_output)

    @torch.no_grad()
    def step(
        self,
        raw_view: dict[str, Any],
        state: TTT3RMemoryState | None = None,
        keep_output: bool = True,
    ) -> tuple[FrameReadout, TTT3RMemoryState]:
        model = self.model
        view = self._to_gpu(raw_view, self.device)
        if "true_shape" not in view and "true_shape" in raw_view:
            true_shape = raw_view["true_shape"]
            if isinstance(true_shape, np.ndarray):
                true_shape = torch.from_numpy(true_shape)
            view["true_shape"] = true_shape.clone().to(self.device)
        selected_imgs = view["img"]
        selected_shapes = view["true_shape"].to(view["img"].device)
        img_out, img_pos, _ = model._encode_image(selected_imgs, selected_shapes)
        feat_i = img_out[-1]
        pos_i = img_pos
        shapes = selected_shapes

        if state is None:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            state = TTT3RMemoryState(
                state_feat=state_feat,
                state_pos=state_pos,
                init_state_feat=state_feat.clone(),
                mem=mem,
                init_mem=mem.clone(),
                step_index=0,
            )

        prev_state_feat = state.state_feat.clone()
        reset_flag = bool(view["reset"].any().item())

        if model.pose_head_flag:
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if state.step_index == 0 or reset_flag:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, state.mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            global_img_feat_i = None
            pose_feat_i = None
            pose_pos_i = None

        new_state_feat, dec, _, cross_attn_state, _, _ = model._recurrent_rollout(
            state.state_feat,
            state.state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            state.init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=True,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        if global_img_feat_i is not None:
            new_mem = model.pose_retriever.update_mem(state.mem, global_img_feat_i, out_pose_feat_i)
        else:
            new_mem = state.mem

        head_input = [
            dec[0].float(),
            dec[model.dec_depth * 2 // 4][:, 1:].float(),
            dec[model.dec_depth * 3 // 4][:, 1:].float(),
            dec[model.dec_depth].float(),
        ]
        output = model._downstream_head(head_input, shapes, pos=pos_i)

        update_mask = view["img_mask"]
        update = view.get("update", None)
        if update is not None:
            update_mask = update_mask & update
        update_mask = update_mask[:, None, None].float()

        if state.step_index == 0 or reset_flag:
            beta_t = update_mask.expand(-1, prev_state_feat.shape[1], -1)
        else:
            cross_attn_state = self._rearrange(
                torch.cat(cross_attn_state, dim=0),
                "l h nstate nimg -> 1 nstate nimg (l h)",
            )
            state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
            beta_t = update_mask * torch.sigmoid(state_query_img_key)[..., None]

        g_t = new_state_feat - prev_state_feat
        delta_s = beta_t * g_t

        next_state_feat = prev_state_feat + delta_s
        next_mem = new_mem * update_mask + state.mem * (1 - update_mask)

        if reset_flag:
            reset_mask = view["reset"][:, None, None].float()
            next_state_feat = state.init_state_feat * reset_mask + next_state_feat * (1 - reset_mask)
            next_mem = state.init_mem * reset_mask + next_mem * (1 - reset_mask)

        next_state = TTT3RMemoryState(
            state_feat=next_state_feat,
            state_pos=state.state_pos,
            init_state_feat=state.init_state_feat,
            mem=next_mem,
            init_mem=state.init_mem,
            step_index=state.step_index + 1,
        )

        conf_self = output["conf_self"].detach()
        beta_mean = beta_t.squeeze(-1).mean(dim=1).detach().cpu()
        delta_s_norm = torch.linalg.vector_norm(delta_s.flatten(1), dim=1).detach().cpu()
        mean_conf_self = conf_self.flatten(1).mean(dim=1).detach().cpu()

        if keep_output:
            beta_cpu = beta_t.squeeze(-1).detach().cpu()
            conf_self_cpu = conf_self.detach().cpu()
            output_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in output.items()}
        else:
            beta_cpu = torch.empty(0)
            conf_self_cpu = torch.empty(0)
            output_cpu = {}

        readout = FrameReadout(
            frame_index=state.step_index,
            beta_t=beta_cpu,
            beta_mean=beta_mean,
            delta_s_norm=delta_s_norm,
            mean_conf_self=mean_conf_self,
            conf_self=conf_self_cpu,
            output=output_cpu,
        )
        return readout, next_state

    @staticmethod
    def summarize(readouts: Iterable[FrameReadout]) -> list[dict[str, float]]:
        summaries: list[dict[str, float]] = []
        for readout in readouts:
            summaries.append(
                {
                    "frame_index": float(readout.frame_index),
                    "beta_mean": float(readout.beta_mean.mean().item()),
                    "delta_s_norm": float(readout.delta_s_norm.mean().item()),
                    "mean_conf_self": float(readout.mean_conf_self.mean().item()),
                }
            )
        return summaries
