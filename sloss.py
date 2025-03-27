import h5py
import torch
from torch import nn
import torch.nn.functional as F

class SearcherLoss(nn.Module):

    def forward(self, Paba, y):
        equality_matrix = torch.eq(y.clone().view(-1,1), y).float()
        u = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        u.requires_grad = False

        L_searcher = F.kl_div(torch.log(1e-8 + Paba), u, size_average=False)
        L_searcher /= u.size()[0]

        return L_searcher

class SupervisorLoss(nn.Module):

    def forward(self, Pt):
        v = torch.ones([1, Pt.size()[1]]) / float(Pt.size()[1])
        v.requires_grad = False
        if Pt.is_cuda: v = v.cuda()
        L_supervisor = F.kl_div(torch.log(1e-8 + Pt), v, size_average=False)
        L_supervisor /= v.size()[0]

        return L_supervisor

class Matrix(nn.Module):

    def __init__(self):
        super(Matrix, self).__init__()

    def forward(self, xs, xt):

        Bs = xs.size()[0] #batch size
        Bt = xt.size()[0]

        xs = xs.clone().view(Bs, -1)
        xt = xt.clone().view(Bt, -1)

        W = torch.mm(xs, xt.transpose(1,0)) #点积

        Pab = F.softmax(W, dim=1)  #公式Pab

        Pba = F.softmax(W.transpose(1,0), dim=1) #Pba

        Paba = Pab.mm(Pba) #Paba

        Pt = torch.mean(Pab, dim=0, keepdim=True) #(4)

        return Paba, Pt

class LOSS(nn.Module):

    def __init__(self, α = 0.1, β = 0.01):
        super(LOSS, self).__init__()

        self.matrix = Matrix()
        self.searcher = SearcherLoss()
        self.supervisor  = SupervisorLoss()

        self.searcher_weight = α
        self.supervisor_weight  = β

    def forward(self, xs, xt, y):

        Paba, Pt = self.matrix(xs, xt)
        L_searcher = self.searcher(Paba, y)
        L_supervisor  = self.supervisor(Pt)

        return self.supervisor_weight*L_supervisor + self.searcher_weight*L_searcher
