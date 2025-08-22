# torchtitan_ext/datasets/blendcorpus_builder.py
import math
from types import SimpleNamespace
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader

# BlendCorpus public API (from README)
from blendcorpus import (
    get_config as bc_get_config,
    set_config as bc_set_config,
    mpu as bc_mpu,
    build_gpt_datasets,
    build_pretraining_data_loader,
)
from blendcorpus.utils import get_ltor_masks_and_position_ids as bc_get_masks
def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def _shift_tokens_to_labels(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # tokens: [B, T]
    input_ids = tokens[:, :-1].contiguous()
    labels    = tokens[:, 1:].contiguous()
    return input_ids, labels


def _maybe_attention_mask(
    input_ids: torch.Tensor,
    eod_token_id: Optional[int],
    reset_pos_ids: bool,
    reset_attn_mask: bool,
    eod_mask_loss: bool,
) -> torch.Tensor:
    # You can return None if TorchTitan model builds its own causal mask internally.
    # If you prefer to supply one, BlendCorpus utility builds LTR masks for Megatron-style flows.
    attn_mask, _, _ = bc_get_masks(
        input_ids,
        eod_token_id if eod_token_id is not None else -1,
        reset_pos_ids,
        reset_attn_mask,
        eod_mask_loss,
    )
    return attn_mask  # [B, 1, T, T] causal mask


def build_blendcorpus_dataloader(cfg, global_batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    TorchTitan will call this to obtain three dataloaders.
    Expected return: (train_loader, valid_loader, test_loader)

    Assumes the following fields exist in your TOML (see sample below):
      [data]
      input_dir = "/path/to/tokenized"   # BlendCorpus output (tokenized)
      seq_length = 8192
      micro_batch_size = 4
      num_workers = 8
      split = "98,1,1"
      dataloader_type = "single"  # or "cyclic"
      append_eod = true           # if you appended eod during tokenization

    """
    # --- Map TorchTitan config → BlendCorpus config namespace ---
    # Feel free to adjust/extend as needed – BlendCorpus' set_config/get_config
    # take an args-like namespace in current releases.
    cfg.blendcorpus.seq_length = getattr(cfg.training, "seq_len")
    cfg.blendcorpus.data_file_list = getattr(cfg.training, "dataset_path")
    cfg.blendcorpus.train_iters = int(getattr(cfg.training, "steps"))
    cfg.blendcorpus.micro_batch_size = int(getattr(cfg.training, "local_batch_size"))
    cfg.blendcorpus.global_batch_size = global_batch_size 
    # Initialize BlendCorpus' (Megatron-like) model-parallel to no-op sizes.
    cfg.blendcorpus.tensor_model_parallel_size = cfg.parallelism.tensor_parallel_degree
    cfg.blendcorpus.pipeline_model_parallel_size = cfg.parallelism.pipeline_parallel_degree
    cfg.blendcorpus.sequence_parallel_size = cfg.parallelism.context_parallel_degree
    bc_mpu.initialize_model_parallel(
        tensor_model_parallel_size=cfg.blendcorpus.tensor_model_parallel_size ,
        pipeline_model_parallel_size=cfg.blendcorpus.pipeline_model_parallel_size,
        sequence_parallel_size=cfg.blendcorpus.sequence_parallel_size,
    )

    bc_set_config(cfg.blendcorpus)
    bc_cfg = bc_get_config()

    # Build BC datasets
    train_ds, valid_ds, test_ds = build_gpt_datasets(bc_cfg)

    # Compute consumed samples to enable dataloader restart from checkpoints
    consumed_train = 0
    consumed_valid = 0
    consumed_test =  0

    # Wrap BC dataloaders (stateful/checkpointable)
    train_loader = build_pretraining_data_loader(train_ds, consumed_train, bc_cfg)
    valid_loader = build_pretraining_data_loader(valid_ds, consumed_valid, bc_cfg) if valid_ds is not None else None
    test_loader  = build_pretraining_data_loader(test_ds,  consumed_test,  bc_cfg) if test_ds  is not None else None

    # TorchTitan expects each batch to be dict-like with "input_ids" and "labels".
    # We'll adapt BC's batches on-the-fly via a generator wrapper.

    def adapt(_iter):
        iterator = iter(_iter)
        for batch in iterator:
            # BC returns {"dataset_idx": [B], "text": Long[B,T]}
            tokens = batch["text"].long()
            input_ids, labels = _shift_tokens_to_labels(tokens)
            yield {'input': input_ids}, labels

    class _AdapterDL:
        """Lightweight wrapper to keep TorchTitan happy (len/iter/epoch size).
        Also allows retargeting (advancing) the underlying BlendCorpus loader
        after Trainer restores the current step from a checkpoint.
        """
        def __init__(self, dl: DataLoader, *, ds, bc_cfg):
            self.dl = dl
            self._ds = ds
            self._bc_cfg = bc_cfg
            try:
                self._len = len(dl)  # type: ignore
            except TypeError:
                self._len = int(1e12)

        def __len__(self):
            return self._len

        def __iter__(self):
            return adapt(iter(self.dl))

        def set_consumed_by_global_step(self, global_step: int, global_batch_size: int):
            """Rebuild the underlying BC dataloader to reflect already-consumed samples.

            This is called *after* Trainer.checkpointer.load() restored the
            current step. BlendCorpus expects consumed in units of samples
            (sequences), so we use step * global_batch_size.
            """
            consumed = int(global_step) * int(global_batch_size)
            # rebuild the loader at the new consumed point
            new_dl = build_pretraining_data_loader(self._ds, consumed, self._bc_cfg)
            self.dl = new_dl

    return (
        _AdapterDL(train_loader, ds=train_ds, bc_cfg=bc_cfg) if train_loader is not None else None,
        _AdapterDL(valid_loader, ds=valid_ds, bc_cfg=bc_cfg) if valid_loader is not None else None,
        _AdapterDL(test_loader,  ds=test_ds,  bc_cfg=bc_cfg) if test_loader  is not None else None,
    )
