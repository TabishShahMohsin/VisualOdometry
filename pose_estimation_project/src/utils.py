import torch

def core_loss(log_pred_periods, log_true_periods):
    """
    Core loss based on Huber loss on log-periods.
    """
    return torch.nn.functional.huber_loss(log_pred_periods, log_true_periods, delta=0.1)

def swap_loss(pred_periods, true_periods):
    """
    Calculates the min loss over two possible assignments (swapped or not).
    """
    log_pred = torch.log(pred_periods)
    log_true = torch.log(true_periods)
    
    loss1_per_item = torch.nn.functional.huber_loss(log_pred, log_true, delta=0.1, reduction='none').mean(dim=1)
    
    swapped_log_pred = torch.stack([log_pred[:, 1], log_pred[:, 0]], dim=1)
    loss2_per_item = torch.nn.functional.huber_loss(swapped_log_pred, log_true, delta=0.1, reduction='none').mean(dim=1)
    
    min_loss_per_item = torch.min(loss1_per_item, loss2_per_item)
    
    return min_loss_per_item.mean()

def freq_loss(pred_periods, true_periods):
    """
    Penalty on frequency error to handle harmonic ambiguity.
    """
    pred_freq = 1.0 / pred_periods
    true_freq = 1.0 / true_periods
    return torch.nn.functional.huber_loss(pred_freq, true_freq, delta=0.001)

def total_supervised_loss(pred_periods, true_periods, lambda_f=0.2):
    """
    Combined supervised loss.
    """
    l_swap = swap_loss(pred_periods, true_periods)
    l_freq = freq_loss(pred_periods, true_periods)
    return l_swap + lambda_f * l_freq