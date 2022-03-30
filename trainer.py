# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

from network import Flashback

'''定义损失函数'''
class My_loss(nn.Module):
    def __init__(self):
        super.__init__()
        
    def forward(self, x1, x2):
        print('hello world!')
        

class FlashbackTrainer():
    
    def __init__(self, lambda_t, lambda_s, poicatg_matrix):
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.poicatg_matrix = poicatg_matrix
        
    def __str__(self):
        return 'Using training!'
    
    def js_loss(self, poicatg_matrix, y_pred_poi, y_pred_catgLayer):
        '''poicatg_matrix——（poi_num, category_num）
           y_pred_poi——（batch_size, poi_num）
           y_pred_catgLayer——（batch_size, category_num）
        '''
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        category_temp = torch.matmul(y_pred_poi, poicatg_matrix)
        poi2category_softmax = torch.nn.functional.softmax(category_temp, dim=1)
        catgLayer_softmax = torch.nn.functional.softmax(y_pred_catgLayer, dim=1)
        log_mean_output = ((poi2category_softmax + catgLayer_softmax)/2).log()
        similarity_loss = (KLDivLoss(log_mean_output, poi2category_softmax) + KLDivLoss(log_mean_output, catgLayer_softmax))/2
        return similarity_loss
        
    
    def parameters(self):
        return self.model.parameters()
    
    def prepare(self, loc_count, user_count, catg_count, catgLayer_count, timeslot_count, poi2coord, hidden_size, gru_factory, device):
        f_t = lambda delta_t : ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t))
        f_s = lambda delta_s : torch.exp(-(delta_s*self.lambda_s)) 
        f_t1 = lambda delta_t : ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, catg_count, catgLayer_count, timeslot_count, hidden_size, f_t, f_t1, f_s, gru_factory).to(device)
        
    def loss(self, h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer):
        self.model.train()
        y_pred_poi, y_pred_catgLayer = self.model(h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg)
        
        '''可以进行调整'''
        # loss = 0.5 * self.cross_entropy_loss(y_pred_poi, y_poi) + 0.3 * self.cross_entropy_loss(y_pred_catg, y_catg) + 0.2 * self.cross_entropy_loss(y_pred_catgLayer, y_catgLayer)
        lambda_p = 0.5
        loss = lambda_p * self.cross_entropy_loss(y_pred_poi, y_poi) + (1-lambda_p) * self.cross_entropy_loss(y_pred_catgLayer, y_catgLayer)
        
        similarity_loss = self.js_loss(self.poicatg_matrix, y_pred_poi, y_pred_catgLayer)
        loss = (1-similarity_loss)*loss
        return loss
    
    def evaluate(self,h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer):
        self.model.eval()
        y_pred_poi, y_pred_catgLayer = self.model(h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg)
        return y_pred_poi, y_pred_catgLayer
    
    def loss1(self, h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg, y_catgLayer):
        self.model.train()
        y_pred_poi, y_pred_catgLayer = self.model(h, x_user, x_tf, x_tb, x_tsf, x_tsb, x_cof, x_cob, x_poi_f, x_poi_b, x_catg_f, x_catg_b, x_catgLayer_f, x_catgLayer_b, y_tsecond, y_tslot, y_coord, y_poi, y_catg)
        return y_pred_poi, y_pred_catgLayer