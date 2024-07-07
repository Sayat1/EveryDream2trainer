from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_rex_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    d = 0.9,
    last_epoch: int = -1
) -> LambdaLR:

    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return 0
        else:
            decay_steps = num_training_steps - num_warmup_steps
            progress = (current_step / decay_steps)
            div = (1 - d) + (d * (1 - progress))
            return 0.0 + (1 - 0.0) * ((1 - progress) / div)
        
    return LambdaLR(optimizer, lr_lambda, last_epoch)