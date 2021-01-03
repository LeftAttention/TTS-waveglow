import torch

class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output, clean_audio):
        # clean_audio is unused;
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(
            z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total  # noqa: E501
        return loss / (z.size(0) * z.size(1) * z.size(2))
