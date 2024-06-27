import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
from collections import OrderedDict
import json


class GSAttention(nn.Module):
    """global spectral attention (GSA), SST

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """
    def __init__(self, dim, num_heads, bias):
       
        super(GSAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def flops(self,patchresolution):
        flops = 0
        H, W,C = patchresolution
        flops +=  H* C *W* C
        flops +=  C *C*H*W
        return flops

class ConvAutoencoder(nn.Module):
    def __init__(self, in_dim):
        super(ConvAutoencoder, self).__init__()
        # encoder layers ##
        # conv layer (depth from 28 --> 28), 3x3 kernels
        self.in_dim = in_dim
        self.conv1 = nn.Conv2d(self.in_dim, self.in_dim, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_out = nn.Conv2d(self.in_dim, self.in_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        # self.fc = nn.Linear(in_features=8 * 3 * 256, out_features=28 * 3)  # Adjust in_features and out_features
        self.upsample = nn.Upsample(size=(28, 3), mode='bilinear', align_corners=False)

    def forward(self, x):
        # encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # compressed representation
        # decode ##
        # add transpose conv layers, with relu activation function
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.upsample(x)
        x = x.view(-1, 1, 28, 3)
        # print(x.shape)
        # exit()
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.shape)
        # exit()
        # x = self.fc(x)
        # x = x.view(-1, 28, 3)  # Reshape to match desired output size
        # output should have a sigmoid applied
        output = torch.sigmoid(x)

        return torch.squeeze(output)

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(28, 32, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will
        # increase the spatial dims by 2
        # kernel_size=3 to get to a 7x7 image output
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 28, 3, padding=1)

    def forward(self, x):
        # encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation

        # decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))

        return x

class Spectral_Conv3D_Block(nn.Module):
    def __init__(self, inC, outC):
        super(Spectral_Conv3D_Block, self).__init__()
        self.conv = nn.Conv3d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.cov_block = nn.Sequential(
            nn.Conv3d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(in_channels=outC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(in_channels=outC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        residual = self.conv(residual)
        residual = self.relu(residual)
        output_f = output + residual
        return output_f


def thre(inputs, threshold):
    '''
    Soft thresholding.

    Args:
        inputs: input tensor
        threshold: threshold value >=0
    Output:
        out: soft thresholding outputs
    '''
    out = torch.sign(inputs) * torch.relu(torch.abs(inputs) - threshold)
    return out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            # h,w = 256//self.heads,320//self.heads
            h, w = 256 // self.heads, 256 // self.heads # haijin
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out


class HS_MSA_direct(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            # h,w = 256//self.heads,320//self.heads
            h, w = 256 // self.heads, 256 // self.heads # haijin
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out


class S2S_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            # h,w = 256//self.heads,320//self.heads
            h, w = 256 // self.heads, 256 // self.heads # haijin
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        self.conv1_1 = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=True)
        self.Spectral_Conv3D_Block = Spectral_Conv3D_Block(inC=self.dim//2, outC=self.dim//2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim//2, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim//2, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            x1, x2 = torch.split(x, (c/2, c/2), dim=-1)
            print(x1.shape)
            # spectral branch
            x2 = x2.view(b, 1, h, w, c/2)
            x2 = x2.permute(0, 4, 1, 2, 3)
            X2S = self.Spectral_Conv3D_Block(x2)
            X2S = X2S.permute(0, 2, 3, 4, 1)
            X2S = X2S.view(b, h, w, c/2)

            #
            c = c//2
            x = x1
            print(x.shape)
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])

            out = torch.cat([out, X2S], dim=-1).contiguous()

        return out


class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

        self.Spectral_Conv3D_Block = Spectral_Conv3D_Block(inC=dim, outC=dim)
        self.conv1_1 = nn.Conv2d(dim, dim*2, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=True)
        self.GSAttention = GSAttention(dim, num_heads=4, bias=False)
        self.conv_block = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
                )

    def forward(self, x):
        x2 = x
        x2_out = self.GSAttention(x2) + x2

        return x2_out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# this U-shape architecture is based on DAUHST
class HGSA(nn.Module):
    def __init__(self, in_dim, out_dim, dim=28, num_blocks=[1,1,1]):
        super(HGSA, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:,:28,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea)

        # Mapping
        out = self.mapping(fea) #+ x
        return out[:, :, :h_inp, :w_inp]

# Spectral unmixing module only with spectral attention
class LR_decompose(nn.Module):

    def __init__(self):
        super(LR_decompose, self).__init__()
        self.decom_A = HGSA(in_dim=28, out_dim=3, dim=28, num_blocks=[1,1,1])
        self.decom_E = ConvAutoencoder(in_dim=28)
        
    def forward(self, y, x):
        """
        :param y: [28,256,256]
        :return: A: [3,256,256]; E:[28,3]
        """
        # decompose noisy image
        b, c, h, w = y.shape
        X_x = torch.zeros(b, c, h, w).cuda().float()
        X_y = torch.zeros(b, c, h, w).cuda().float()

        A_y = self.decom_A(y)
        E_y = self.decom_E(y)
        if E_y.shape[0]==28:
            E_y = torch.unsqueeze(E_y, 0)
            
        for i in range(b):
            A_ym = torch.reshape(A_y[i,:,:,:], [3, 256*256])
            X_ym = torch.mm(E_y[i,:,:], A_ym)
            X_y[i,:,:,:]   = torch.reshape(X_ym, [28, 256, 256])        
            
        A_x = self.decom_A(x)
        E_x = self.decom_E(x)
        if E_x.shape[0]==28:
            E_x = torch.unsqueeze(E_x, 0)   

        for i in range(b):
            A_xm = torch.reshape(A_x[i,:,:,:], [3, 256*256])
            X_xm = torch.mm(E_x[i,:,:], A_xm)
            X_x[i,:,:,:]   = torch.reshape(X_xm, [28, 256, 256])


        return X_y, X_x, A_y, A_x, E_y, E_x


