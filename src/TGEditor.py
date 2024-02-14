import argparse
import matplotlib.pyplot as plt
import torch
import warnings
import config
import os
from HTNE import *
from torch.optim import SparseAdam
from torch.utils.data import DataLoader, RandomSampler
from concurrent.futures import ProcessPoolExecutor
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import copy
import torch.optim as optim
from torch.autograd import grad
from torch.nn.functional import one_hot
import torch.nn.functional as F
import time


debug = True

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        self.l2 = nn.Linear(hidden_size, output_size).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)

    def forward(self, h, sigmoid=False):
        h = self.l1(h)
        if sigmoid:
            h = self.l2(torch.sigmoid(h))
        else:
            h = self.l2(F.relu(h))
        return h

class BaseGenerator(nn.Module):
    def __init__(self, device='cpu'):
        super(BaseGenerator, self).__init__()
        self.device = device

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim)).type(torch.float64).to(self.device)

    def sample(self, num_samples):
        noise = self.sample_latent(num_samples)
        labels = torch.ones(noise.shape[0], 1).to(self.device)
        noise = torch.hstack((noise, labels))
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(self.device)
        rw, ts, _ = self(noise, input_zeros, self.device)                              # (n, rw_len, N), (n, rw_len, 1)
        return rw, ts

    def sample_discrete(self, num_samples):
        with torch.no_grad():
            rw, ts = self.sample(num_samples)                          # (n, rw_len, N), (n, rw_len, 1)
        rw = np.argmax(rw.cpu().numpy(), axis=2)                               # (n, rw_len)
        ts = torch.squeeze(ts, dim=2).cpu().numpy()                            # (n, rw_len)
        return rw, ts
    
    def graph_edit(self, num_samples, walks):
    
        noise = self.sample_latent(num_samples)
        labels = torch.ones(noise.shape[0], 1).to(self.device)
        noise = torch.hstack((noise, labels))
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(self.device)
        rm_rw, add_rw = self(noise, input_zeros, input_rws=walks)                           # (n, rw_len)
        
        return rm_rw, add_rw
    
    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand(logits.shape, dtype=torch.float64)                      # gumbel_noise = uniform noise [0, 1]
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(self.device)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=1)
        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)





class DyGANGenerator(BaseGenerator):
    def __init__(self, H_inputs, batch_size, H, z_dim, H_t, N, rw_len, temp, disten=False, k=100, rm_th=0.0002, add_th=0.98, device='cpu'):
        ''' The original DyGAN generator
        H_inputs: input dimension
        H:        hidden state dimension
        z_dim:    dimension of the latent code z
        H_t:      dimension of the time hidden embedding
        N:        Number of nodes in the graph to generate
        rw_len:   Length of random walks to generate
        add_th:   threshold for adding node
        rm_th:   threshold for removing node
        temp:     temperature for the gumbel softmax
        k:        number of verification done during evaluation
        '''
        BaseGenerator.__init__(self)
        self.device = device
        self.compressor =  nn.Linear(z_dim+1, z_dim).type(torch.float64)
        self.shared_intermediate = nn.Linear(z_dim, H).type(torch.float64)     # (z_dim, H)
        torch.nn.init.xavier_uniform_(self.shared_intermediate.weight)
        torch.nn.init.zeros_(self.shared_intermediate.bias)
        self.c_intermediate = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_intermediate.weight)
        torch.nn.init.zeros_(self.c_intermediate.bias)
        self.h_intermediate = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_intermediate.weight)
        torch.nn.init.zeros_(self.h_intermediate.bias)
        self.lstmcell = nn.LSTMCell(H_inputs, H).type(torch.float64)
        if disten:
            self.time_adapter = MLP(H, H, H).type(torch.float64)  
        self.time_decoder = TimeDecoder(H, H_t).type(torch.float64)
        self.Wt_up = nn.Linear(1, H_t).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_up.weight)
        torch.nn.init.zeros_(self.Wt_up.bias)
        self.W_up = nn.Linear(H, N).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_up.weight)
        torch.nn.init.zeros_(self.W_up.bias)

        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.W_vt = nn.Linear(H_inputs + H_t, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_vt.weight)
        self.prob = nn.Linear(H_inputs, 1, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.prob.weight)
        self.disten = disten
        self.rw_len = rw_len
        self.temp = temp
        self.H_inputs = H_inputs
        self.batch_size=batch_size,
        self.H = H
        self.latent_dim = z_dim
        self.H_t = H_t
        self.N = N
        self.k = k
        self.rm_th = rm_th
        self.add_th = add_th
    
    '''
        Training:
            graph editor module works just like a generator by producing the best sequence infered
        Evaluating:
            graph editor will produce the best action based on the joint probability of v1, v2, t
    '''
    def forward(self, latent, inputs, induced=True, input_rws=None):
        if induced:
            latent = torch.tanh(self.compressor(latent))
        shared_intermediate = torch.tanh(self.shared_intermediate(latent))
        hc = (torch.tanh(self.h_intermediate(shared_intermediate)),
            torch.tanh(self.c_intermediate(shared_intermediate)))              # stack h and c
        hc_copy = (torch.clone(hc[0]), torch.clone(hc[1]))
        last_t = torch.zeros(inputs.shape[0], 1).type(torch.float64).to(self.device)
        last_t_copy = torch.zeros(inputs.shape[0], 1).type(torch.float64).to(self.device)
        if input_rws == None:
            rw = []
            ts = []
            for _ in range(self.rw_len):
                hh, cc = self.lstmcell(inputs, hc)                                 # start with a fake input
                hc = (hh, cc)                                                      # for the next cell step
                if self.disten:
                    attn_map = torch.sigmoid(self.time_adapter(hh))
                    H_graph = attn_map * hh
                    H_time = (1 - attn_map) * hh
                else:
                    H_graph, H_time = hh, hh
                p = self.W_up(H_graph)                                             # (n, N)
                #we also draw the probability distribution to determine graph editing procedure
#                 v = self.gumbel_softmax_sample(p, self.temp, self.device, hard=False)               # (n, N), batch of one-hot vectors
                v = self.gumbel_softmax_sample(p, self.temp)
                H_graph = self.W_down(v)                                           # (n, H)
                t = self.time_decoder(H_time)                                      # (n, 1)
                t = self.time_decoder.constraints(t, last_t)                       # (n, 1)
                H_time = self.Wt_up(t)                                             # (n, H_t)
                vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                inputs = self.W_vt(vt)
                last_t = t
                rw.append(v)
                ts.append(t)
                
            return  torch.stack(rw, dim=1), torch.stack(ts, dim=1), self.prob(inputs)                  # (n, rw_len, N), (n, rw_len, 1)
        else:
            '''
                we perform k step verifications
            '''
            nodes,times = input_rws
            selections = torch.multinomial(torch.arange(self.batch_size[0], dtype=torch.float64), self.k).to(self.device)
            cur_walk_a = torch.index_select(nodes, 0, selections)  #to compute the p(rw) agains p(rw')
            cur_time_a = torch.index_select(times, 0, selections)
            '''
                we compute the initial rw score
            '''
            rs = []
            ts = []
            for rw_step in torch.arange(self.rw_len):
                hh, cc = self.lstmcell(inputs, hc)                                 # start with a fake input
                hc = (hh, cc)                                                      # for the next cell step
                if self.disten:
                    attn_map = torch.sigmoid(self.time_adapter(hh))
                    H_graph = attn_map * hh
                    H_time = (1 - attn_map) * hh
                else:
                    H_graph, H_time = hh, hh
                p = self.W_up(H_graph)                                             # (n, N)             # (n, N), batch of one-hot vectors
                v = nodes[:,rw_step,:]
                H_graph = self.W_down(v)                                           # (n, H)
                t = times[:,rw_step,:].view(-1,1)
                H_time = self.Wt_up(t)                                             # (n, H_t)
                vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                inputs = self.W_vt(vt)
                last_t = t
                rs.append(v)
                ts.append(t)
            rs = torch.stack(rs, dim=1)
            ts =  torch.stack(ts, dim=1)
            '''
                we compute the score for real random walk
            '''
            real_rw_prob = torch.sigmoid(self.prob(inputs))
            
            overall_rm_rw = []
            overall_add_rw = []
            
#             overall_rm_ts = []
#             overall_add_ts = []
            for step in torch.arange(self.k):
                
                rm_rw = []
                add_rw = []

                rm_ts = []
                add_ts = []
                
                step = step.to(self.device)
                indices = torch.argmax(cur_walk_a[step], dim=-1)
                indices_cpu = indices.clone().detach().cpu().numpy()
                time_indices = cur_time_a[step]
                time_indices_cpu = time_indices.clone().detach().cpu().numpy()
                hc = (hc_copy[0][step].view(1,-1), hc_copy[1][step].view(1,-1))
                current_input = inputs[step].view(1,-1)
                cur_last_t = last_t_copy[step]
                for rw_step in torch.arange(self.rw_len):
                    rw_step = rw_step.to(self.device)
                    hh, cc = self.lstmcell(current_input, hc)
                    hc = (hh, cc)
                    H_graph, H_time = hh, hh
                    p = self.W_up(H_graph)  
                    v = self.gumbel_softmax_sample(p, self.temp)
                    v_hard = torch.max(v, 1, keepdim=True)[0].eq(v).type(torch.float64).to(self.device)
                    v_hard = (v_hard - v).detach() + v
                    v = v.squeeze()
                    v_true = F.one_hot(indices[rw_step], num_classes=self.N).to(self.device).double().view(1,-1)
                    
                    t = self.time_decoder(H_time)                                     # (n, 1)
                    t = self.time_decoder.constraints(t, cur_last_t)                  # (n, 1)
                    t_true = time_indices[rw_step].double().view(1, -1)
                    
                    rm_th = v.sort().values[int(len(v) * 0.05)]
                    top_candidate = torch.argmax(v).detach().item()  # index
                    top_candidate_prob = v[top_candidate]
                    if top_candidate_prob > self.add_th:
                        v_true = v_hard
                        t_true = t
                        if debug:
                            print("added an edge", v[top_candidate])
                        if rw_step > 0:
                            add_rw.append((indices_cpu[rw_step-1], top_candidate, time_indices_cpu[rw_step-1].item()))
                        if rw_step < self.rw_len - 1:
                            add_rw.append((top_candidate, indices_cpu[rw_step + 1], time_indices_cpu[rw_step + 1].item()))
                                
                    if v[indices[rw_step]] < rm_th:
                        if debug:
                            print("removed an edge", v[indices[rw_step]], time_indices_cpu[rw_step].item())
            
                        rw_step = rw_step.cpu().item()

                        if rw_step > 0:
                            rm_rw.append((indices_cpu[rw_step - 1], indices_cpu[rw_step], time_indices_cpu[rw_step - 1].item()))
                        if rw_step < self.rw_len -1:
                            rm_rw.append((indices_cpu[rw_step], indices_cpu[rw_step + 1], time_indices_cpu[rw_step].item()))
                    else:
                        v_true[0][indices[rw_step]] = 1
#                             if debug:
#                                 print("skipping becuase the current node is removed")
#                             continue
                    H_graph = self.W_down(v_true) 
                    H_time = self.Wt_up(t_true) 
                    vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                    current_input = self.W_vt(vt)
                    cur_last_t = time_indices[rw_step].view(-1,1)
                verified_rw_prob = torch.sigmoid(self.prob(current_input))
#                 if verified_rw_prob > real_rw_prob[step]:
                if verified_rw_prob > real_rw_prob[step]:
                    if debug:
                        print("accepted for the better prob:", verified_rw_prob, " vs ", real_rw_prob[step])
                    
                    overall_rm_rw = overall_rm_rw + rm_rw
                    overall_add_rw = overall_add_rw + add_rw
            return  overall_rm_rw, overall_add_rw

class Discriminator(nn.Module):
    def __init__(self, H_inputs, H, H_t, N):
        '''
            H_inputs: input dimension
            H:        hidden state dimension
            H_t:      dimension of the time hidden embedding
            N:        Number of nodes in the graph to generate
            rw_len:   Length of random walks to generate
        '''
        super(Discriminator, self).__init__()
        self.W_vt = nn.Linear(H_inputs + H_t, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_vt.weight)
        self.lstmcell = nn.LSTMCell(H_inputs, H).type(torch.float64)
        self.lin_out = nn.Linear(H, 1, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        torch.nn.init.zeros_(self.lin_out.bias)
        self.Wt_up = nn.Linear(1, H_t).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_up.weight)
        torch.nn.init.zeros_(self.Wt_up.bias)

        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.H = H
        self.H_t = H_t
        self.N = N
        self.H_inputs = H_inputs

    def forward(self, v, t):
        rw_len = v.shape[1]
        v = v.view(-1, self.N)                             # [n*rw_len, N]
        v = self.W_down(v)                                 # [n*rw_len, H_inputs]
        v = v.view(-1, rw_len, self.H_inputs)              # [n, rw_len, H_inputs]
        t = t.view(-1, rw_len, 1)                          # [n, rw_len, 1]
        t = self.Wt_up(t)                                  # [n, rw_len, H_t]
        vt = torch.cat((v, t), dim=2)                      # [n, rw_len, H_inputs + H_t]
        inputs = self.W_vt(vt)                             # [n, rw_len, H_inputs]
        hc = self.init_hidden(v.size(0))
        for i in range(rw_len):
            hc = self.lstmcell(inputs[:, i, :], hc)
        pred = self.lin_out(hc[0])
        return pred

    def init_inputs(self, num_samples):
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        weight = next(self.parameters()).data
        return (
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64),
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64))
        


class TimeDecoder(nn.Module):
    def __init__(self, H, H_t, dropout_p=0.2):
        super(TimeDecoder, self).__init__()
        self.Wt_down = nn.Linear(H, H_t).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_down.weight)
        torch.nn.init.zeros_(self.Wt_down.bias)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.Wt_pred = nn.Linear(H_t, 1, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_pred.weight)
        self.dropout_p = dropout_p

    def forward(self, x):                                  # (n, H)
        self.dropout.train()
        x = torch.tanh(self.Wt_down(x))                    # (n, H_t)
        x = self.dropout(x)                                # (n, H_t)
        x = self.Wt_pred(x)                                # (n, 1)
        max_t = torch.ones(x.shape[0], 1).type(torch.float64).to(x.device)
        x = torch.clamp(x, max=max_t)                      # make sure x is less than 1
        return x

    def constraints(self, x, last_t, epsilon=1e-1):
        epsilon = torch.tensor(epsilon).type(torch.float64).to(x.device)
        max_t = torch.ones(x.shape[0], 1).type(torch.float64).to(x.device)
        min_ = torch.min(x)
        x = torch.where(min_ < epsilon, x - min_, x)      # (n, 1)
        max_ = torch.max(x)
        x = torch.where(1. < max_, x / max_, x)
        x = torch.clamp(x, min=last_t, max=max_t)
        return x

class DyGAN_trainer():
    def __init__(
        self, data, node_embs=None, max_iterations=20000, rw_len=16, induced=True, batch_size=128, H_gen=40, H_disc=30, H_node=128,
        H_t=12, disten=False, H_inp=128, z_dim=16, lr=0.0003, n_critic=2, gp_weight=10.0,
        betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0,
        temp_decay=1-5e-5, min_temp=0.5, baselines_stats=None, device='cpu'):
        """Initialize DyGAN.
        :param data: CTDNE temporal data
        :param N: Number of nodes in the graph to generate
        :param max_iterations: Maximal iterations if the stopping_criterion is not fulfilled, defaults to 20000
        :param rw_len: Length of random walks to generate, defaults to 16
        :param batch_size: The batch size, defaults to 128
        :param H_gen: The hidden_size of the generator, defaults to 40
        :param H_disc: The hidden_size of the discriminator, defaults to 30
        :param H_t: The hidden_size of the time embedding, defaults to 12
        :param H_inp: Input size of the LSTM, defaults to 128
        :param z_dim: The dimension of the random noise that is used as input to the generator, defaults to 16
        :param lr: Learning rate for both generator and discriminator, defaults to 0.0003
        :param n_critic:  The number of discriminator iterations per generator training iteration, defaults to 3
        :param gp_weight: Gradient penalty weight for the Wasserstein GAN, defaults to 10.0
        :param betas: Decay rates of the Adam Optimizers, defaults to (.5, .9)
        :param l2_penalty_disc:  L2 penalty for the generator weights, defaults to 5e-5
        :param l2_penalty_gen: L2 penalty on the di10scriminator weights, defaults to 1e-7
        :param temp_start: The initial temperature for the Gumbel softmax, defaults to 5.0
        :param temp_decay: After each evaluation, the current temperature is updated as
                        `current_temp := max(temp_decay * current_temp, min_temp)`, defaults to 1-5e-5
        :param min_temp: The minimal temperature for the Gumbel softmax, defaults to 0.5
        """ 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.max_iterations = max_iterations
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.disten = disten
        # self.node_emb = node_embs
        """
        " change N to feature dimension for decoding
        """
        self.N = data.num_nodes
        self.E = data.num_edges
        self.generator = DyGANGenerator(
            H_inputs=H_inp, batch_size=batch_size,H=H_gen, H_t=H_t, N=self.N, rw_len=rw_len,
            z_dim=z_dim, temp=temp_start, disten=disten, device=device).to(self.device)
#         self.discriminator = Discriminator(
#             H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N,
#             rw_len=rw_len).to(self.device)
        self.discriminator = Discriminator(
            H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N).to(self.device)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.l2_penalty_disc = l2_penalty_disc
        self.l2_penalty_gen =l2_penalty_gen
        self.temp_start = temp_start
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self.data = data
#         self.data.df_data = self.data.df_data.drop_duplicates(['src','tar', 't'], keep='last').drop(['label'], axis=1)
        self.data.df_data = self.data.df_data.drop_duplicates(['src','tar', 't'], keep='last')
        self.walker = CTDNE_random_walker(self.data, rw_len, batch_size)
        self.induced = induced
        self.critic_loss = []
        self.generator_loss = []
        self.running = True
        # self.gold_stats = utils.eval_temp_graph(self.data.df_data, self.N, self.data.unique_ts)
        self.baselines_stats = baselines_stats
#         for name in self.baselines_stats:
#             print(f"{name} score: {utils.temp_score(self.gold_stats, self.baselines_stats[name]):.4f}")
        self.best_gen = DyGANGenerator(
            H_inputs=H_inp,batch_size=batch_size, H=H_gen, H_t=H_t, N=self.N, rw_len=rw_len,
            z_dim=z_dim, temp=temp_start, disten=disten).to(self.device)
#         self.best_disc = Discriminator(
#             H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N,
#             rw_len=rw_len).to(self.device)
        self.best_disc = Discriminator(
            H_inputs=H_inp,H=H_disc, H_t=H_t, N=self.N).to(self.device)
        self.best_gen.eval()
        self.best_disc.eval()
        self.best_score = 1000000

    def l2_regularization_G(self, G):
        # regularizaation for the generator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in G.W_down.weight]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.weight]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.bias]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in G.shared_intermediate.weight]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in G.shared_intermediate.bias]) ** 2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in G.h_intermediate.weight]) ** 2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in G.h_intermediate.bias]) ** 2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in G.c_intermediate.weight]) ** 2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in G.c_intermediate.bias]) ** 2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.weight_ih]) ** 2 / 2)
        l2_11 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.weight_hh]) ** 2 / 2)
        l2_12 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.bias_ih]) ** 2 / 2)
        l2_13 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.bias_hh]) ** 2 / 2)
        l2_14 = torch.sum(torch.cat([x.view(-1) for x in G.W_vt.weight]) ** 2 / 2)
        l2_15 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_down.weight]) ** 2 / 2)
        l2_16 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_down.bias]) ** 2 / 2)
        l2_17 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_pred.weight]) ** 2 / 2)
        l2_18 = torch.sum(torch.cat([x.view(-1) for x in G.Wt_up.weight]) ** 2 / 2)
        l2_19 = torch.sum(torch.cat([x.view(-1) for x in G.Wt_up.bias]) ** 2 / 2)
        # l2_20 = torch.sum(torch.cat([x.view(-1) for x in G.node_decoder.linear.weight]) ** 2 / 2)
        # l2_21 = torch.sum(torch.cat([x.view(-1) for x in G.node_decoder.linear.bias]) ** 2 / 2)
        l_rw = l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10 + l2_11 + l2_12 + l2_13
        l_gen = l2_14 + l2_15 + l2_16 + l2_17 + l2_18 + l2_19
        if self.disten:
            l2_20 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l1.weight]) ** 2 / 2)
            l2_21 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l1.bias]) ** 2 / 2)
            l2_22 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l2.weight]) ** 2 / 2)
            l2_23 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l2.bias]) ** 2 / 2)
            l_gen += l2_20 + l2_21 + l2_22 + l2_23
        l2 = self.l2_penalty_gen * (l_rw + l_gen)
        return l2

    def l2_regularization_D(self, D):
        # regularizaation for the discriminator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in D.W_down.weight]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.weight_ih]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.weight_hh]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.bias_ih]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.bias_hh]) ** 2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.weight]) ** 2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.bias]) ** 2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in D.W_vt.weight]) ** 2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in D.Wt_up.weight]) ** 2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in D.Wt_up.bias]) ** 2 / 2)
        l2 = self.l2_penalty_disc * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10)
        return l2

    def calc_gp(self, fake_inputs, real_inputs):
        # calculate the gradient penalty. For more details see the paper 'Improved Training of Wasserstein GANs'.
        alpha = torch.rand((self.batch_size, 1, 1), dtype=torch.float64).to(self.device)
        fake_ns, fake_ts= fake_inputs
        real_ns, real_ts = real_inputs
        # print(fake_inputs, real_inputs)
        ns_difference = fake_ns - real_ns
        ts_difference = fake_ts - real_ts
#         fs_difference = fake_labels - real_labels
        ns_interpolates = real_ns + alpha * ns_difference
        ts_interpolates = real_ts + alpha * ts_difference
#         fs_interpolates = real_labels + alpha * fs_difference
        interpolates = (ns_interpolates, ts_interpolates)

        # for tensor in interpolates:
            # tensor.requires_grad = True

        y_pred_interpolates = self.discriminator(*interpolates)
        gradients = grad(outputs=y_pred_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(y_pred_interpolates),
            create_graph=True,
            retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
        gradient_penalty = torch.mean((slopes - 1) ** 2)
        gradient_penalty = gradient_penalty * self.gp_weight
        return gradient_penalty

    def critic_train_iteration(self):
        start = time.time()
        self.D_optimizer.zero_grad()
        #we generate walks first for label info and feed such info to the generator matching cgan
        real_ns, real_ts, real_labels = self.walker.walk()
        # create fake and real inputs
        real_ns = one_hot(torch.tensor(real_ns).type(torch.long), num_classes=self.N).type(torch.float64).to(self.device)
        real_ts = torch.tensor(real_ts)
        real_ts = real_ts.unsqueeze(dim=2).type(torch.float64).to(self.device)
        real_labels = torch.tensor(real_labels).to(self.device)
        real_inputs = (real_ns, real_ts)
#         fake_inputs = self.generator. ample(self.batch_size, self.device) # (n, rw_len, N), (n, rw_len, 1)
        fake_inputs = self.generator.sample(self.batch_size)
        # print(fake_inputs, real_inputs)
        # raise
         # (n, rw_len), # (n, rw_len)
        y_pred_fake = self.discriminator(*fake_inputs)
        y_pred_real = self.discriminator(*real_inputs)
        gp = self.calc_gp(fake_inputs, real_inputs)                  # gradient penalty
        disc_cost = torch.mean(y_pred_fake) - torch.mean(y_pred_real) + gp + self.l2_regularization_D(self.discriminator)
        disc_cost.backward()
        self.D_optimizer.step()
        end = time.time()
        print(f"took {end-start} seconds for one iteration")
        return disc_cost.item()
        

    def generator_train_iteration(self):
        self.generator.train()
        self.G_optimizer.zero_grad()
        fake_inputs = self.generator.sample(self.batch_size)
        
        y_pred_fake = self.discriminator(*fake_inputs)
        gen_cost = -torch.mean(y_pred_fake) + self.l2_regularization_G(self.generator)

        gen_cost.backward()
        self.G_optimizer.step()
        return gen_cost.item()

    def save_model(self):
        self.best_gen.load_state_dict(copy.deepcopy(self.generator.state_dict()))
        self.best_disc.load_state_dict(copy.deepcopy(self.discriminator.state_dict()))
        self.best_gen.eval()
        self.best_disc.eval()

    def load_model_from_best(self):
        self.generator.load_state_dict(copy.deepcopy(self.best_gen.state_dict()))
        self.discriminator.load_state_dict(copy.deepcopy(self.best_disc.state_dict()))

    def create_graph(self, data, i=0, visualize=False, update=False, edges_only=False, num_iterations=3):
        self.generator.eval()
        self.generator.temp = 0.5
        rm_nodes = []
        rm_times = []
        
        add_nodes = []
        add_times = []
        
        # edges_per_batch = self.batch_size * (self.rw_len - 1)
        # num_iterations = int(int(num_edges / 2) / edges_per_batch) + 1
        edges = data.df_data.drop(columns=['label'])
        ori_len = len(edges)
        for _ in range(num_iterations):
            
            rw, ts, _ = self.walker.walk()
            rw = one_hot(torch.tensor(rw).type(torch.long), num_classes=self.N).type(torch.float64).to(self.device)
            ts = torch.tensor(ts)
            ts = ts.unsqueeze(dim=2).type(torch.float64).to(self.device)
            rm_rw, add_rw = self.generator.graph_edit(self.batch_size,(rw, ts))
           
            edges_add = pd.DataFrame(rm_rw, columns=['src', 'tar', 't'])
            if len(rm_rw) > 0:
                edges_rm = pd.DataFrame(rm_rw, columns=['src', 'tar', 't'])
                edges = pd.concat([edges, edges_rm, edges_rm]).drop_duplicates(keep=False)
                print("removed ", len(edges_rm), "edges")
                edges.reset_index(drop=True, inplace=True)
            if len(add_rw) > 0:
                edges_add = pd.DataFrame(add_rw, columns=['src', 'tar', 't'])
                edges = pd.concat([edges, edges_add]).drop_duplicates(keep=False)
                print("added ", len(edges_add), "edges")
                edges.reset_index(drop=True, inplace=True)
        
        if visualize:
            self.data.visualize(edges)
        edges_copy = edges.copy()
        edges_copy['src'] = edges['tar']
        edges_copy['tar'] = edges['src']
        edges = pd.concat([edges, edges_copy], axis=0).drop_duplicates(keep='last')
        edges.reset_index(drop=True, inplace=True)
        if len(edges) < ori_len:
            dummy = [(0,0,0)]*(ori_len - len(edges))
            dummy = pd.DataFrame(dummy, columns=['src', 'tar', 't'])
            edges = pd.concat([edges, dummy])
#         elif len(edges) > ori_len:
#             dummy = [(0,0,0)]*(ori_len - len(edges))
#             dummy = pd.DataFrame(dummy, columns=['src', 'tar', 't'])
#             edges = pd.concat([edges, dummy])
        if edges_only:
            return edges
        # if update:
        #     self.generator.temp = np.maximum(self.temp_start * np.exp(-(1 - self.temp_decay) * i), self.min_temp)
        #     if generated_score < self.best_score:
        #         self.best_score = generated_score
        #         self.save_model()
        # return edges
    
    def convert_edges(self, rw, ts):
        
        edges = utils.temp_walk2edge(rw, ts)
        edges = edges[:int(num_edges / 2), :]
        
        
        edges = pd.DataFrame(edges, columns=['src', 'tar', 't'])
        edges_copy = edges.copy()
        edges_copy['src'] = edges['tar']
        edges_copy['tar'] = edges['src']
        edges = pd.concat([edges, edges_copy], axis=0)
#         assert len(edges) == num_edges
        edges.reset_index(drop=True, inplace=True)
    
    
        edges = pd.DataFrame(edges, columns=['src', 'tar', 't'])
        edges_copy = edges_rm.copy()
        edges_copy['src'] = edges['tar']
        edges_copy['tar'] = edges['src']
        edges = pd.concat([edges, edges_copy], axis=0)
#         assert len(edges) == num_edges
        edges.reset_index(drop=True, inplace=True)
        return edges

    def eval_model(self, num_eval=20):
        self.load_model_from_best()
        scores = []
        for _ in range(num_eval):
            temp_stats, edges = self.create_graph(num_edges=self.E, i=0, visualize=False, update=False)
            score = utils.temp_score(self.gold_stats, temp_stats)
            scores.append(score)
        print(f"Average: {np.mean(scores):.4f}")
        print(f"Max    : {np.max(scores):.4f}")
        print(f"Min    : {np.min(scores):.4f}")
        print(f"Median : {np.median(scores):.4f}")
        return scores

    def plot_graph(self):
        if len(self.critic_loss) > 10:
            plt.plot(self.critic_loss[9::], label="Critic loss")
            plt.plot(self.generator_loss[9::], label="Generator loss")
        else:
            plt.plot(self.critic_loss, label="Critic loss")
            plt.plot(self.generator_loss, label="Generator loss")
        plt.legend()
        plt.show()

    
    def train(self, create_graph_every=2000, plot_graph_every=200):
        """
        create_graph_every: int, default: 2000
            Creates every nth iteration a graph from randomwalks.
        plot_graph_every: int, default: 2000
            Plots the lost functions of the generator and discriminator.
        """
        starting_time = time.time()
        # Start Training
        print("start training")
        for i in range(self.max_iterations):
            if self.running:
                self.critic_loss.append(np.mean([self.critic_train_iteration() for _ in range(self.n_critic)]))
                self.generator_loss.append(self.generator_train_iteration())
                if i % 100 == 1:
                    print(f'iteration: {i}    critic: {self.critic_loss[-1]:.5f}    gen: {self.generator_loss[-1]:.5f}')
                # if i % create_graph_every == create_graph_every - 1:
                #     self.create_graph(self.E,i , visualize=True, update=True)
                #     print(f'Took {(time.time() - starting_time)/60} minutes so far..')
                if plot_graph_every > 0 and (i + 1) % plot_graph_every == 0:
                    self.plot_graph()
