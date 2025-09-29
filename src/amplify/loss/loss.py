import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from typing import Tuple

from ..tokenizer import ProteinTokenizer


def get_loss(
    device: torch.device,
    strategy: str, 
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None = None,
    label_smoothing: float = 0.0,
    weights: (dict | None) = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.modules.loss._Loss:
    """Public wrapper for constructing the loss function.

    Args:
        device (torch.device): Device.
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id (int): <PAD> token index.
        mask_token_id (int): <MASK> token index.
        bos_token_id (int): <BOS> token index.
        eos_token_id (int): <EOS> token index.
        unk_token_id (int): <UNK> token index.
        other_special_token_ids (list | None, optional): Indices of the special other tokens. Defaults to None.
        label_smoothing (float, optional): Label smoothing coefficient. Defaults to 0.0.
        weights (dict  |  None, optional): Class weights. Defaults to None.
        strategy_name (str, optional): Strategy for loss manipulation. Defaults to constrained learning.
        dtype (torch.dtype, optional): Dtype of the class_weights. Defaults to torch.float32.

    Returns:
        torch.nn.modules.loss._Loss: A cross-entropy loss function with mean or none reduction.
    """
    tokenizer = ProteinTokenizer(
        vocab_path,
        pad_token_id,
        mask_token_id,
        bos_token_id,
        eos_token_id,
        unk_token_id,
        other_special_token_ids,
    )

    # Class weights
    class_weights = None
    if weights is not None and any(w != 1 for w in weights.values()):
        class_weights = [weights.get(tokenizer.id_to_token(i), 1) for i in range(len(tokenizer))]
        class_weights = Tensor(class_weights).to(device, dtype, non_blocking=True)

    if strategy == 'ally':
        return CrossEntropyLoss(weight=class_weights, reduction="none", label_smoothing=label_smoothing)
    else:
        return CrossEntropyLoss(weight=class_weights, reduction="mean", ignore_index=-100, label_smoothing=label_smoothing)


def get_lagrangian(
    device: torch.device,
    train_loss_seq: torch.Tensor,
    lambdas_current: torch.Tensor,
    slacks_current: torch.Tensor,
    dual_lr: float = 0.1,
    epsilon: float = 2.4,
    alpha: float = 0.1,
    **kwargs,
) -> torch.Tensor:

    lagrangian = (train_loss_seq*(1+lambdas_current.to(device)) - \
                    lambdas_current.to(device)*(epsilon+slacks_current.to(device))).nanmean() + \
                    0.5*alpha*torch.linalg.norm(slacks_current)**2

    return lagrangian


def update_dual_variables(
    train_loss_seq: torch.Tensor,
    lambdas_current: torch.Tensor,
    slacks_current: torch.Tensor,
    epsilon: float,
    dual_lr: float,
    slack_lr: float,
    alpha: float,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:

    train_loss_seq = train_loss_seq.detach().cpu()
    nan_mask = torch.isnan(train_loss_seq)
    nan_idxs = torch.nonzero(nan_mask, as_tuple=True)
    
    train_loss_seq[nan_idxs] = epsilon + slacks_current[nan_idxs] # skip nan when updating dual variables, replace epsilon with epsilon+slacks

    lambdas_tmp = lambdas_current
    lambdas_current += dual_lr*(train_loss_seq-(epsilon+slacks_current))
    slacks_current -= slack_lr*(0.5*alpha*slacks_current-lambdas_tmp) 

    lambdas_current.data.clamp_(min=0)
    slacks_current.data.clamp_(min=0)

    return lambdas_current, slacks_current