import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from .data_loader import SentenceRELoader
from .utils import AverageMeter, display_progress
torch.manual_seed(1219)
torch.cuda.manual_seed_all(1219)
torch.backends.cudnn.deterministic = True

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 batch_size=32, 
                 max_epoch=10000, 
                 max_global_step=5000,
                 lr=0.1, 
                 weight_decay=0.00001, 
                 opt='sgd',
                 model_name = 'default'):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.max_global_step = max_global_step
        self.model_name = model_name
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True) 
        if val_path != None:
            self.dev_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )      
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.consistent_creation = nn.MSELoss()
        # Params and optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        self.lr = lr
        self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt


    def train_model(self, metric='micro_f1'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.model(*args)
                #logits = self.parallel_model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                if global_step % 500 == 0 and global_step != 0 :
                    print("=== Step %d val ===" % global_step)
                    result = self.eval_model(self.val_loader)
                    self.F1_record.append(result['micro_f1'])
                    self.acc_record.append(result['acc'])
                    print(result)
                    print('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
                    if result[metric] > best_metric:
                        print("Best ckpt and saved.")
                        best_metric = result[metric]
                        torch.save(self.model.state_dict(), open(os.path.join(self.save_model_path, self.model_name), "wb"))
            # Val
            ''' 
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print(result)
            print('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                best_metric = result[metric]
                torch.save(self.model.state_dict(), open(os.path.join(self.save_model_path, self.model_name), "wb"))
            '''

    def train_model_overfitting(self) :
        best_metric = 0
        global_step = 0
        End_signal = True
        while End_signal:
            self.train()
            for iter, data in enumerate(self.train_loader):
                avg_loss = AverageMeter()
                avg_acc = AverageMeter()
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.model(*args)
                #logits = self.parallel_model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                # Optimize
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                display_progress("Training step: avg loss:%f avg acc:%f"%(avg_loss.avg, avg_acc.avg), global_step, self.max_global_step)
                global_step += 1
                if global_step >= self.max_global_step :
                    End_signal = False
                    break
        #torch.save(self.model.state_dict(), open(os.path.join(self.save_model_path, self.model_name), "wb"))
    def train_model_overfitting_with_ema(self, ema_model, gamma) :
        best_metric = 0
        global_step = 0
        End_signal = True
        while End_signal:
            self.train()
            for iter, data in enumerate(self.train_loader):
                avg_loss_classfication = AverageMeter()
                avg_loss_consistency = AverageMeter()
                avg_acc = AverageMeter()
                avg_ema_acc = AverageMeter()
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits1 = self.model(*args)
                logits2 = ema_model(*args).detach()
                #logits = self.parallel_model(*args)
                loss_classfication = self.criterion(logits1, label)
                loss_consistency = self.softmax_mse_loss(logits1, logits2)/self.batch_size
                #gamma = get_consistency_weight(global_step)
                loss = loss_classfication + gamma*loss_consistency
                score, pred = logits1.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                score, pred = logits2.max(-1) # (B)
                ema_acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss_classfication.update(loss_classfication.item(), 1)
                avg_loss_consistency.update(loss_consistency.item(), 1)
                avg_acc.update(acc, 1)
                avg_ema_acc.update(ema_acc, 1)
                # Optimize
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                display_progress("Training step:avg loss1:%f avg acc1:%f avg loss2:%f avg acc2:%f"%(avg_loss_classfication.avg, avg_acc.avg, avg_loss_consistency.avg, avg_ema_acc.avg), global_step, self.max_global_step)
                global_step += 1
                if global_step >= self.max_global_step :
                    End_signal = False
                    break
    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                loss = self.criterion(logits, label)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
            self.eval_loss_record.append(avg_loss.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def load_model(self, path):
        state_dict = torch.load(open(path, "rb"),map_location = lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)


    def softmax_mse_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        num_classes = input_logits.size()[1]
        return self.consistent_creation(input_softmax, target_softmax) / num_classes


def get_current_consistency_weight(epoch):   
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
