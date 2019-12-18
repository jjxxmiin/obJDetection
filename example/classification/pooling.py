import torch


class GVAP(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)

        batch = x.size(0)
        depth = x.size(1)



        return x

    def backward(self, grad_output):
        x = self.saved_tensors

        return x

gvap = GVAP()

test = torch.ones((1, 1024, 4, 4))

print(test)

a = gvap(test)