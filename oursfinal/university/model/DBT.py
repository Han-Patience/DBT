import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SiT(nn.Module):
    def __init__(self,
                 *,
                 signal_heigh,
                 signal_width,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool="cls",
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(nn.Linear(signal_width, dim), )

        self.pos_embedding = nn.Parameter(torch.randn(1, signal_heigh + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
#         print(img.shape)
        x = self.to_patch_embedding(img)
#         print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
#         print(x.shape)
        x += self.pos_embedding[:, :(n + 1)]
#         print(x.shape)
        x = self.dropout(x)
#         print('before trans',x.shape)
        x = self.transformer(x)
#         print('after trans',x.shape)
        x = x[:, 0]
#         print('after pool',x.shape)
        x = self.to_latent(x)
#         print(self.mlp_head(x).shape)
        return self.mlp_head(x)


class DataBlock(nn.Module):
    def __init__(self,
                 shift=3,
                 israwdata=False,
                 layerNorm=False,
                 max_len: int = None,
                 isMs=False,
                 isDBT=False,
                 isAVGi=False,
                 isSOD=False,
                 isDd=False):
        super(DataBlock, self).__init__()

        assert shift > 0, "shift must >=1"

        self.derivConv2d = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=(2, 1),
                                     stride=(2, 1),
                                     padding=0,
                                     bias=False)
        self.derivConv2d.apply(self.conv_init)
        
        self.derivConv2d_avg = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=(5, 1),
                                     stride=(5, 1),
                                     padding=0,
                                     bias=False)
        self.derivConv2d_avg.apply(self.conv_init_avg)
        
        self.derivConv2d_2jiecahfen = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=(4, 1),
                                     stride=(4, 1),
                                     padding=0,
                                     bias=False)
        self.derivConv2d_2jiecahfen.apply(self.conv_init_2jiecahfen)
        
        self.isDBT = isDBT
        self.shift = shift
        self.layerNorm =layerNorm
        self.isMs = isMs
        self.israwdata = israwdata
        self.isAVGi = isAVGi
        self.isSOD = isSOD
        self.isDd = isDd

        self.latent = nn.Identity()
        for param in self.parameters():
            param.requires_grad = False

        self.width =  (max_len - (1 <<(shift - 1)))
        self.ln = nn.LayerNorm(self.width)#逐行平滑

        self.ms = MultiScale(3, self.width)#参数3为poolkernal可调整，多尺度特征

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.weight.data.fill_(0.618)  # = 初始化
        self.sm = nn.Hardtanh()
#---------------------------------------------------#
    def forward(self, x):
        x = x[:, None, None, :]  #bz,1,1,len
        
        x1 = x.repeat(1, 1, 5, 1)  #跃变均值
        
        x2 = x.repeat(1, 1, 4, 1)  #二阶差分
        
        x3 = x.repeat(1, 1, 2, 1)  #指定行数的错位
        
        x4 = x.repeat(1, 1, 1, 1)  #原始数据
              

        x = x.repeat(1, 1, self.shift << 1, 1)  #bz,1,shift*2,len
        
        for d1 in range(0, self.shift):
            x[:, :, d1 << 1, :] = x.roll(1 << d1, -1)[:, :, d1 << 1, :]

        
        x3[:, :, 1, :] = x3.roll(4, -1)[:, :, 1, :]
        
        x1[:, :, 1, :] = x1.roll(2, -1)[:, :, 1, :]
        x1[:, :, 2, :] = x1.roll(4, -1)[:, :, 2, :]
        x1[:, :, 3, :] = x1.roll(6, -1)[:, :, 3, :]
        x1[:, :, 4, :] = x1.roll(8, -1)[:, :, 4, :]

        
        x2[:, :, 1, :] = x2.roll(1, -1)[:, :, 1, :]
        x2[:, :, 2, :] = x2.roll(1, -1)[:, :, 2, :]
        x2[:, :, 3, :] = x2.roll(2, -1)[:, :, 3, :]
        
        deriv1 = self.derivConv2d(x)

        deriv1_1 = self.derivConv2d_avg(x1)

        deriv1_2 = self.derivConv2d_2jiecahfen(x2)
        
        deriv1_3 = self.derivConv2d(x3)


#跃变均值    
        restemp1 = deriv1_1[:, :, :, (1 << (d1)):]
        restemp1 = restemp1.squeeze(1)
#一阶差分
        restemp2 = deriv1[:, :, :, (1 << (d1)):]
        restemp2 = restemp2.squeeze(1)
#原始数据
        restemp3 = x4[:, :, :, (1 << (d1)):]
        restemp3 = restemp3.squeeze(1)
#二阶差分
        restemp4 = deriv1_2[:, :, :, (1 << (d1)):]
        restemp4 = restemp4.squeeze(1)
#远距离依赖关系
        restemp5 = deriv1_3[:, :, :, (1 << (d1)):]
        restemp5 = restemp5.squeeze(1)
        
        deriv1[:,:,0,:] = x4[:,:,0,:]

        res = deriv1[:, :, :, (1 << (d1)):]
        res = res.squeeze(1)        
#-------------------------------------------------#

#Data Block
        if self.isDBT:
            res = res.repeat(1, 4, 1)#数据的逐行堆叠
            res = self.ln(res)
            res[:,1,:] = restemp1[:,0,:]
            res[:,2,:] = restemp2[:,0,:]
            res[:,3,:] = restemp3[:,0,:]
            return res

#多尺度特征
        elif self.isMs:
            res = self.ms(res)
            return res
        
#逐行平滑
        elif self.layerNorm:
            res = res.repeat(1, 4, 1)
            res = self.ln(res)
            return res

#二阶差分
        elif self.isSOD:
            res = restemp4
            res = res.repeat(1, 4, 1)
            return res
#远距离依赖
        elif self.isDd:
            res = restemp5
            res = res.repeat(1, 4, 1)
            return res
#跃变均值
        elif self.isAVGi:
            res = restemp1
            res = res.repeat(1, 4, 1)
            return res
#原始数据        
        elif self.israwdata:
            res = res.repeat(1, 4, 1)
            return res
#一阶差分
        else:
            res = restemp2
            res = res.repeat(1, 4, 1)
            return res


    def conv_init(self, conv):
        weight = torch.tensor(
            [[[[-1.0], [1.0]]]],
            requires_grad=False)  # .to(torch.device("cuda:7" if torch.cuda.is_available() else "cpu"))

        conv.weight.data = weight
        
    def conv_init_avg(self, conv):
        weight = torch.tensor(
            [[[[1/5],[1/5],[1/5],[1/5], [1/5]]]],
            requires_grad=False)  # .to(torch.device("cuda:7" if torch.cuda.is_available() else "cpu"))

        conv.weight.data = weight
        
    def conv_init_2jiecahfen(self, conv):
        weight = torch.tensor(
            [[[[1.0], [-1.0], [-1.0], [1.0]]]],
            requires_grad=False)  # .to(torch.device("cuda:7" if torch.cuda.is_available() else "cpu"))

        conv.weight.data = weight 

class MultiScale(nn.Module):
    def __init__(self, poolKernel=9, size: int = None) -> None:
        super(MultiScale, self).__init__()
        poolKernel = 9
        self.avgPool1 = nn.AvgPool1d(kernel_size=poolKernel, stride=1)
        self.avgPool2 = nn.AvgPool1d(kernel_size=poolKernel, )
        self.maxPool1 = nn.MaxPool1d(kernel_size=poolKernel, stride=1)
        self.maxPool2 = nn.MaxPool1d(kernel_size=poolKernel, )
        self.upSample = nn.Upsample(size=(size), mode="nearest")
        self.latent = nn.Identity()

    def forward(self, x):
        res = [
            self.upSample(self.avgPool1(x)) + x,
            self.upSample(self.avgPool2(x)) + x,
            self.upSample(self.maxPool1(x)) + x,
            self.upSample(self.maxPool2(x)) + x,
        ]
        return torch.cat(res, dim=1)


class DBT(nn.Module):
    def __init__(self,
                 *,
                 max_len,
                 num_class,
                 shift,
                 dim=1024,
                 depth=6,
                 heads=4,
                 mlp_dim=1024,
                 dropout,
                 emb_dropout,
                 layerNorm,
                 isMs,
                 isDBT,
                 israwdata,
                 isAVGi,
                 isSOD,
                 isDd):
        super(DBT, self).__init__()

        self.DB = DataBlock(shift=shift,
                                    layerNorm=layerNorm,
                                    max_len=max_len,
                                    isMs=isMs,
                                    isDBT=isDBT,
                                    israwdata=israwdata,
                                    isAVGi=isAVGi,
                                    isSOD=isSOD,
                                    isDd=isDd)

        self.signal_heigh = shift * 4 + 1  # ~ 4 represent ms 4 times,and 1 is embed raw data
        self.signal_width = (max_len - (1 << (shift - 1)))
        self.signal_heigh -= 1
        self.Sit = SiT(
            signal_heigh=self.signal_heigh,
            signal_width=self.signal_width,
            num_classes=num_class,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.weight.data.fill_(0.02)  # = 初始化 Pavia0.02 Paviadis 0.99
        self.sm = nn.Hardtanh()
        # self.sm = nn.Tanh()
        self.latent = nn.Identity()
        self.embed = nn.Sequential(nn.Linear(max_len, self.signal_width), nn.Dropout(0.1))

    def forward(self, x):
        
        x = self.DB(x)
            
        x = self.Sit(x)

        return x


if __name__ == "__main__":
    model = DBT(max_len=200,
                    num_class=15,
                    dropout=0,
                    emb_dropout=0,
                    shift=1,
                    layerNorm=True,
                    isMs=True,
                    isDBT=True,
                    israwdata=True,
                    isAVGi=True,
                    isSOD=True,
                    isDd=True)
    img = torch.rand(4, 200)
    res = model(img)

    print(res.shape)
