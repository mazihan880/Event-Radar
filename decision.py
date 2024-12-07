import torch
import torch.nn as nn
import torch.nn.functional as F


# loss function
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, alpha.device)

    return (A + B)


class fushion_decision(nn.Module):

    def __init__(self, views, feature_out, lambda_epochs=50):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(fushion_decision, self).__init__()
        self.views = views
        self.lambda_epochs = lambda_epochs
        #print(self.views)
        self.Classifiers = nn.ModuleList([Classifier(feature_out) for i in range(self.views)])

       
    def DSuncertain(self, alpha):
           
        b, S, E, u = dict(), dict(), dict(), dict()
        
        for v in range(3):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = 2/S[v]

        return torch.stack([u[0],u[1],u[2]],dim = 1)

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], 2, global_step, self.lambda_epochs)
        uncertaincof = self.DSuncertain(alpha)
        #print(uncertaincof.shape)
        loss = torch.mean(loss)
        return evidence, uncertaincof, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Module):
    def __init__(self, feature_out):
        super(Classifier, self).__init__()
        
        self.fc = nn.Sequential(
                nn.Linear(feature_out, feature_out),
                nn.ReLU(),
                nn.Linear(feature_out, feature_out//2)
                )
                
        self.fcclass = nn.Linear(feature_out//2, 2)
        self.fcevd = nn.Softplus()

    def forward(self, x):
        h = self.fc(x)
        h = self.fcclass(h)
        h = self.fcevd(h)
        return h