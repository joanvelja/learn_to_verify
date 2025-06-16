deepspeed.utils.safe_get_full_fp32_param(param)[source]
Assemble and return the fp32 parameter of a low-precision (e.g., fp16) parameter.

Parameters
param (torch.nn.Parameter) – A model parameter

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

deepspeed.utils.safe_get_full_grad(param)[source]
Assemble and return the fp32 gradient of a low-precision (e.g., fp16) parameter. The return data type is that used for gradient accumulation. This is usually the param data type, but could also be different (e.g., bf16 param training with fp32 gradient accumulation).

Parameters
param (torch.nn.Parameter) – A model parameter

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

deepspeed.utils.safe_get_full_optimizer_state(param, optim_state_key)[source]
Assemble and return the fp32 optimizer state of a low-precision (e.g., fp16) parameter.

Parameters
param (torch.nn.Parameter) – A model parameter

optim_state_key (string) – Key value of optimizer state (e.g., exp_avg in Adam optimizer)

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

deepspeed.utils.safe_get_local_fp32_param(param)[source]
Get the local partition of a ZeRO-3 partitioned parameter in fp32 precision.

Parameters
param (torch.nn.Parameter) – A model parameter.

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

deepspeed.utils.safe_get_local_grad(param)[source]
Get the local gradient partition of a ZeRO-3 partitioned parameter. The return data type is that used for gradient accumulation. This is usually the param data type, but could also be different (e.g., bf16 param training with fp32 gradient accumulation).

Parameters
param (torch.nn.Parameter) – A model parameter

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

deepspeed.utils.safe_get_local_optimizer_state(param, optim_state_key)[source]
Get the local optimizer state partition of ZeRO-3 partitioned parameter in fp32 precision.

Parameters
param (torch.nn.Parameter) – A model parameter

optim_state_key (string) – Key value of optimizer state (e.g., exp_avg in Adam optimizer)

Returns
A tensor on accelerator device

Return type
Union[torch.Tensor, None]

These routines can be used in a training loop as shown in the following snippet.

backward(loss)
[...]
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
for n, lp in model.named_parameters():
    # 1. Access the full states
    #  1.1) gradient lookup
    # For zero1 and zero2, gradient lookup must be called after `backward` and before `step`
    # For zero3, gradient lookup must be called after `backward`
    hp_grad = safe_get_full_grad(lp)


    # 1.2) fp32 and optim states can probably be called anywhere in the training loop, but will be updated after `step`
    hp = safe_get_full_fp32_param(lp)
    exp_avg = safe_get_full_optimizer_state(lp, "exp_avg")
    exp_avg_sq = safe_get_full_optimizer_state(lp, "exp_avg_sq")

    # 2. Access the local states (zero3)
    # For zero3, all of the parameters, gradients, and optimizer states are partitioned,
    # and each process can access its corresponding local state.
    local_hp = safe_get_local_fp32_param(lp)
    local_hp_grad = safe_get_local_grad(lp)
    local_exp_avg = safe_get_local_optimizer_state(lp, "exp_avg")
    local_exp_avg_sq = safe_get_local_optimizer_state(lp, "exp_avg_sq")

[...]
optimizer.step()