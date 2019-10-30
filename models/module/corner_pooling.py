'''
REFERENCE : https://github.com/feifeiwei/Pytorch-CornerNet/blob/master/module/corner_pooling.py
'''
import torch


def comp(a, b, A, B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).contiguous().view(batch, 1, -1)
    b_ = b.unsqueeze(1).contiguous().view(batch, 1, -1)
    c_ = torch.cat((a_, b_), 1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_ == m).float()
    m1 = m.permute(0, 2, 1)
    k = m1[..., 0]
    j = m1[..., 1]
    z = ((k * j) != 1).float()
    j = z * j
    m1 = torch.cat((k, j), 1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).contiguous().view(batch, 1, -1)
    B_ = B.unsqueeze(1).contiguous().view(batch, 1, -1)
    C_ = torch.cat((A_, B_), 1).permute(0, 2, 1)
    m1 = m1.long().permute(0, 2, 1)
    res = C_[m1.long() == 1].view_as(a)

    return res


# customize grad function
class left_pool(torch.autograd.Function):

    def forward(self, x):
        self.save_for_backward(x.clone())
        output = torch.zeros_like(x)
        batch = x.size(0)
        width = x.size(3)

        input_tmp = x.select(3, width - 1)
        output.select(3, width - 1).copy_(input_tmp)

        for idx in range(1, width):
            input_tmp = x.select(3, (width - 1) - idx)
            output_tmp = output.select(3, width - idx)

            in_ = input_tmp.view(batch, 1, -1)
            out_ = output_tmp.view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(3, width - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        x = self.saved_tensors
        output = torch.zeros_like(x)
        batch = x.size(0)
        width = x.size(3)

        input_tmp = x.select(3, width - 1)
        output.select(3, width - 1).copy_(input_tmp)

        grad = grad_output.clone()
        result = torch.zeros_like(grad)

        grad_tmp = grad.select(3, width - 1)
        result.select(3, width - 1).copy_(grad_tmp)

        for idx in range(1, width):
            input_tmp = x.select(3, (width - 1) - idx)
            output_tmp = output.select(3, width - idx)

            in_ = input_tmp.view(batch, 1, -1)
            out_ = output_tmp.view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(3, width - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

            # grad update
            grad_tmp = grad.select(3, (width - 1) - idx)
            result_tmp = result.select(3, width - idx)

            comp_tmp = comp(input_tmp, output_tmp, grad_tmp, result_tmp)

            result.select(3, width - idx - 1).copy_(comp_tmp)

        return result


class right_pool(torch.autograd.Function):

    def forward(self, x):
        self.save_for_backward(x.clone())
        output = torch.zeros_like(x)
        batch = x.size(0)
        width = x.size(3)

        input_tmp = x.select(3, 0)
        output.select(3, 0).copy_(input_tmp)

        for idx in range(1, width):
            input_tmp = x.select(3, idx)
            output_tmp = output.select(3, idx - 1)

            in_ = input_tmp.view(batch, 1, -1)
            out_ = output_tmp.view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(3, idx).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        x = self.saved_tensors
        output = torch.zeros_like(x)
        batch = x.size(0)
        width = x.size(3)

        input_tmp = x.select(3, 0)
        output.select(3, 0).copy_(input_tmp)

        grad = grad_output.clone()
        result = torch.zeros_like(grad)

        grad_tmp = grad.select(3, 0)
        result.select(3, 0).copy_(grad_tmp)

        for idx in range(1, width):
            input_tmp = x.select(3, idx)
            output_tmp = output.select(3, idx - 1)

            in_ = input_tmp.view(batch, 1, -1)
            out_ = output_tmp.view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(3, idx).copy_(cmp_tmp.view_as(input_tmp))

            # grad
            grad_tmp = grad.select(3, idx)
            result_tmp = result.select(3, idx - 1)

            comp_tmp = comp(input_tmp, output_tmp, grad_tmp, result_tmp)

            result.select(3, idx).copy_(comp_tmp)

        return result


class top_pool(torch.autograd.Function):

    def forward(self, x):
        self.save_for_backward(x.clone())
        output = torch.zeros_like(x)
        batch = x.size(0)
        height = x.size(2)

        input_tmp = x.select(2, height - 1)
        output.select(2, height - 1).copy_(input_tmp)

        for idx in range(1, height):
            input_tmp = x.select(2, (height - 1) - idx)
            output_tmp = output.select(2, height - idx)

            in_ = input_tmp.contiguous().view(batch, 1, -1)
            out_ = output_tmp.contiguous().view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(2, height - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        x = self.saved_tensors
        output = torch.zeros_like(x)
        batch = x.size(0)
        height = x.size(2)

        input_tmp = x.select(2, height - 1)
        output.select(2, height - 1).copy_(input_tmp)

        grad = grad_output.clone()
        result = torch.zeros_like(grad)

        grad_tmp = grad.select(2, height - 1)
        result.select(2, height - 1).copy_(grad_tmp)

        for idx in range(1, height):
            input_tmp = x.select(2, (height - 1) - idx)
            output_tmp = output.select(2, height - idx)

            in_ = input_tmp.contiguous().view(batch, 1, -1)
            out_ = output_tmp.contiguous().view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(2, height - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

            # grad
            grad_tmp = grad.select(2, (height - 1) - idx)
            result_tmp = result.select(2, height - idx)

            comp_tmp = comp(input_tmp, output_tmp, grad_tmp, result_tmp)

            result.select(2, height - idx - 1).copy_(comp_tmp)

        return result


class bottom_pool(torch.autograd.Function):

    def forward(self, x):
        self.save_for_backward(x.clone())
        output = torch.zeros_like(x)
        batch = x.size(0)
        height = x.size(2)

        input_tmp = x.select(2, 0)
        output.select(2, 0).copy_(input_tmp)

        for idx in range(1, height):
            input_tmp = x.select(2, idx)
            output_tmp = output.select(2, idx - 1)

            in_ = input_tmp.contiguous().view(batch, 1, -1)
            out_ = output_tmp.contiguous().view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        x = self.saved_tensors
        output = torch.zeros_like(x)
        batch = x.size(0)
        height = x.size(2)

        input_tmp = x.select(2, 0)
        output.select(2, 0).copy_(input_tmp)

        grad = grad_output.clone()
        result = torch.zeros_like(grad)

        grad_tmp = grad.select(2, 0)
        result.select(2, 0).copy_(grad_tmp)

        for idx in range(1, height):
            input_tmp = x.select(2, idx)
            output_tmp = output.select(2, idx - 1)

            in_ = input_tmp.contiguous().view(batch, 1, -1)
            out_ = output_tmp.contiguous().view(batch, 1, -1)

            # row max value
            # width * channel
            cmp_tmp = torch.cat((in_, out_), 1).max(1)[0]

            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))

            # grad
            grad_tmp = grad.select(2, idx)
            result_tmp = result.select(2, idx - 1)

            comp_tmp = comp(input_tmp, output_tmp, grad_tmp, result_tmp)

            result.select(2, idx).copy_(comp_tmp)

        return result
