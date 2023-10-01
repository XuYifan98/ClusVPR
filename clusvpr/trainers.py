from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from .utils.meters import AverageMeter
from torch.cuda.amp import autocast as autocast
from clusvpr.evaluators import extract_cnn_feature


class ClusVPRTrainer(object):
    def __init__(self, model, model_cache, margin=0.3,
                    neg_num=10, gpu=None, temp=[0.07,]):
        super(ClusVPRTrainer, self).__init__()
        self.model = model
        self.model_cache = model_cache

        self.margin = margin
        self.gpu = gpu
        self.neg_num = neg_num
        self.temp = temp

    def train(self, gen, epoch, sub_id, data_loader, optimizer, train_iters,
                    print_freq=1, lambda_soft=0.5, loss_type='sare_ind'):
        self.model.train()
        self.model_cache.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_hard = AverageMeter()
        losses_soft = AverageMeter()
        end = time.time()

        data_loader.new_epoch()

        for i in range(train_iters):

            inputs_easy, inputs_diff = self._parse_data(data_loader.next())
            data_time.update(time.time() - end)

            loss_hard, loss_soft = self._forward(inputs_easy, inputs_diff, loss_type, gen)
            loss = loss_hard + loss_soft*lambda_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_hard.update(loss_hard.item())
            losses_soft.update(loss_soft.item())

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}-{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_hard {:.3f} ({:.3f})\t'
                      'Loss_soft {:.3f} ({:.3f})'
                      .format(epoch, sub_id, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_hard.val, losses_hard.avg,
                              losses_soft.val, losses_soft.avg))

    def _parse_data(self, inputs):
        imgs = [input[0] for input in inputs]
        imgs = torch.stack(imgs).permute(1,0,2,3,4)
        imgs_easy = imgs[:,:self.neg_num+2]
        imgs_diff = torch.cat((imgs[:,0].unsqueeze(1).contiguous(), imgs[:,self.neg_num+2:]), dim=1)
        return imgs_easy.cuda(self.gpu), imgs_diff.cuda(self.gpu)

    def _forward(self, inputs_easy, inputs_diff, loss_type, gen):
        B, _, C, H, W = inputs_easy.size()
        inputs_easy = inputs_easy.view(-1, C, H, W)
        inputs_diff = inputs_diff.view(-1, C, H, W)

        sim_easy, vlad_anchors, vlad_pairs = self.model(inputs_easy)
        # vlad_anchors: B*1*9*L
        # vlad_pairs: B*(1+neg_num)*9*L
        with torch.no_grad():
            sim_diff_label, _, _ = self.model_cache(inputs_diff) # B*diff_pos_num*9*9
        sim_diff, _, _ = self.model(inputs_diff)

        if (gen==0):
            loss_hard = self._get_loss(vlad_anchors[:,0,0], vlad_pairs[:,0,0], vlad_pairs[:,1:,0], B, loss_type)
        else:
            loss_hard = 0
            for tri_idx in range(B):
                loss_hard += self._get_hard_loss(vlad_anchors[tri_idx,0,0].contiguous(), vlad_pairs[tri_idx,0,0].contiguous(), \
                                                vlad_pairs[tri_idx,1:], sim_easy[tri_idx,1:,0].contiguous().detach(), loss_type)
            loss_hard /= B

        log_sim_diff = F.log_softmax(sim_diff[:,:,0].contiguous().view(B,-1)/self.temp[0], dim=1)
        loss_soft = (- F.softmax(sim_diff_label[:,:,0].contiguous().view(B,-1)/self.temp[gen], dim=1).detach() * log_sim_diff).mean(0).sum()

        return loss_hard, loss_soft

    def _get_hard_loss(self, anchors, positives, negatives, score_neg, loss_type):
        # select the most difficult patches for negatives
        score_arg = score_neg.view(self.neg_num,-1).argmax(1)
        score_arg = score_arg.unsqueeze(-1).unsqueeze(-1).expand_as(negatives).contiguous()
        select_negatives = torch.gather(negatives,1,score_arg)
        select_negatives = select_negatives[:,0]

        return self._get_loss(anchors.unsqueeze(0).contiguous(), \
                            positives.unsqueeze(0).contiguous(), \
                            select_negatives.unsqueeze(0).contiguous(), 1, loss_type)

    def _get_loss(self, output_anchors, output_positives, output_negatives, B, loss_type):
        L = output_anchors.size(-1)

        if (loss_type=='triplet'):
            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_positives = output_positives.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            loss = F.triplet_margin_loss(output_anchors, output_positives, output_negatives,
                                            margin=self.margin, p=2, reduction='mean')

        elif (loss_type=='sare_joint'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)
            dist_neg = dist_neg.view(B, -1)

            # joint optimize
            dist = torch.cat((dist_pos, dist_neg), 1)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        elif (loss_type=='sare_ind'):
            dist_pos = torch.mm(output_anchors, output_positives.transpose(0,1)) # B*B
            dist_pos = dist_pos.diagonal(0)
            dist_pos = dist_pos.view(B, 1)

            output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
            output_negatives = output_negatives.contiguous().view(-1, L)
            dist_neg = torch.mm(output_anchors, output_negatives.transpose(0,1)) # B*B
            dist_neg = dist_neg.diagonal(0)

            dist_neg = dist_neg.view(B, -1)

            # indivial optimize
            dist_neg = dist_neg.unsqueeze(2)
            dist_pos = dist_pos.view(B, 1, 1).expand_as(dist_neg)
            dist = torch.cat((dist_pos, dist_neg), 2).view(-1, 2)/self.temp[0]
            dist = F.log_softmax(dist, 1)
            loss = (- dist[:, 0]).mean()

        else:
            assert ("Unknown loss function")

        return loss
