import torch


class GAP(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)

        output = torch.mean(x, [2, 3])

        return output

    def backward(self, grad_output):
        x = self.saved_tensors
        x = x[0]

        # 1/n * local gradient
        grad_input = grad_output / x.size(2) * x.size(3)

        # (batch, channel) -> (batch, channel, 1, 1)
        grad_input = grad_input.unsqueeze(2).unsqueeze(3)

        # (batch, channel, 1, 1) -> (batch, channel, 4, 4)
        grad_input = grad_input.repeat(1, 1, x.size(2), x.size(3))

        return grad_input