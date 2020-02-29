
import torch
from torch import nn
from sympy import *
import numpy as np

def vector_args(fun):
    def wrapper(args):
        return fun(*args)
    return wrapper

class FuncVector:
    def __init__(self, F=None):
        self.F = F
        return

    @classmethod
    def from_sympy(cls, F, L):
        FVec = cls()
        FVec.F = np.zeros(len(F), dtype=np.object)
        for i, _ in enumerate(F):
            FVec.F[i] = vector_args(lambdify(L, F[i]))
        return FVec

    def __call__(self, x):
        res = torch.zeros_like(x)
        for i,_ in enumerate(x):
            res[i] = self.F[i](x)
        return res

class noise(nn.Module):
    def __init__(self, eps, delta):
        super(noise, self).__init__()
        self.eps = eps
        self.delta = nn.Parameter(delta)
        return

    def __call__(self):
        return self.eps * (abs(self.eps) > 0) + self.delta * (self.eps == 0)


if __name__=='__main__':
    L = symbols('x1, x2, x3, x4, x5')
    N = len(L)
    [x1, x2, x3, x4, x5] = L

    f1 = x1 ** 2 + x2 ** 2 - x3
    f2 = x2 ** 2 + x3 ** 2 - 5 * x1
    f3 = x3 ** 2 + x4 ** 2 - 8 * x2
    f4 = x4 ** 2 + x5 ** 2 - 8 * x1
    f5 = x5 ** 2 + x1 ** 2 - 5 * x1

    X = Matrix([x1, x2, x3, x4, x5])
    F = Matrix([f1, f2, f3, f4, f5])
    J = F.jacobian(X)
    H = J.diff(X).as_immutable().reshape(N, N, N)


    x_0 = torch.FloatTensor([1, 1, 2, 2, 2])[:, None]
    eps = torch.FloatTensor([.1, .1, 0, 0, 0])[:, None]
    e = noise(eps, torch.zeros_like(x_0))

    xe_star = x_0 + e.eps

    values = dict(zip(X, xe_star))

    JV = torch.FloatTensor(J.subs(values).tolist())
    HV = torch.FloatTensor(H.subs(values).tolist()).permute(1, 2, 0)

    F = FuncVector.from_sympy(F, L)
    J = JV
    H = HV

    def find_L(G, noise):
        L_main = torch.sum(G.T @ G)
        L_reg = 1e6*torch.sum((noise.delta * (abs(noise.eps) > 0)) ** 2)
        return L_main + L_reg

    opt = torch.optim.Adam(e.parameters(), lr=1e-2)
    for step in range(1000):
        g1 = F(xe_star)
        g2 = (e.delta - xe_star).T @ J
        g3 = (e.delta - xe_star).T @ H @ (e.delta - xe_star)
        G = g1 + g2.T + g3[:,:,0]
        L = find_L(G, e)
        if step % 10 == 0:
            print('Step #{}, loss = {}\n\tdelta: {}'.format(step, L, e.delta.data))
        L.backward()
        opt.step()
        opt.zero_grad()

    print(F(x_0 + e()))