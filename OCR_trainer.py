import os
import tqdm
import numpy as np
import torch
import torch.nn as nn

from models.recognition import HTRNet

import torch.nn.functional as F
import argparse
from utils.metrics import CER, WER
from data_loader.builder import build_dataloader
from data_loader.data_sampler import AspectRatioBatchSampler
from data_loader.HWT_dataset import HWTDataset, letters
from torch.utils.data import RandomSampler
from utils.misc import read_config

device = "cuda"

class HTRTrainer(nn.Module):
    def __init__(self):
        super(HTRTrainer, self).__init__()
        self.resume = "model_zoo/vae_HTR138.pth"
        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()

    def prepare_dataloaders(self):
        train_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "train",
                               load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
        print('number of training images: ', len(train_dataset))
        #####
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset), dataset=train_dataset,
                                                batch_size=config.train_batch_size,
                                                aspect_ratios=train_dataset.aspect_ratio,
                                                drop_last=False,
                                                ratio_nums=train_dataset.ratio_nums, valid_num=config.valid_num)
        train_loader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=8,
                                        collate_fn=train_dataset.collate_fn_)
        test_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "test",
                                load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
        print('number of testing images: ', len(test_dataset))

        #####
        test_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(test_dataset), dataset=test_dataset,
                                                    batch_size=config.test_batch_size,
                                                    aspect_ratios=test_dataset.aspect_ratio,
                                                    drop_last=False,
                                                    ratio_nums=test_dataset.ratio_nums,
                                                    valid_num=config.valid_num)
        
        test_loader = build_dataloader(test_dataset, batch_sampler=test_batch_sampler, num_workers=8,
                                    collate_fn=test_dataset.collate_fn_)


        self.loaders = {'train': train_loader, 'val': test_loader}

        classes = letters

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i) for i,c in enumerate(classes)}
        icdict = {(i):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):
        net = HTRNet(nclasses=len(letters), vae=True)
        if self.resume is not None:
            print('resuming from checkpoint: {}'.format(self.resume))
            load_dict = torch.load(self.resume)
            load_status = net.load_state_dict(load_dict, strict=False)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / config.train_batch_size

    def prepare_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), 0.0001, weight_decay=0.00005)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])

    def decode(self, tdec, tdict, blank_id=0):
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        return dec_transcr
                
    def sample_decoding(self):
        test_loader_iter = iter(self.loaders['val'])
        test_data = next(test_loader_iter)
        data = test_data[np.random.randint(0, len(test_data))]
        img, target = data['img'].to(device), \
                    data['target'].to(device), \
        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])
        transcr = self.decode(target.cpu().numpy(), self.classes['i2c'])
        print('orig:: ' + transcr.strip())
        print('pred:: ' + dec_transcr.strip())


    def train(self, epoch):
        self.net.to(device)
        self.net.train()
        t = tqdm.tqdm(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, data in enumerate(t):
            img, target, target_lengths = data['img'].to(device), \
                data['target'].to(device), \
                data['target_lengths'].to(device)

            self.optimizer.zero_grad()
            output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)])

            loss_val = self.ctc_loss(output, target, act_lens, target_lengths)
            tloss_val = loss_val.item()
        
            loss_val.backward()
            self.optimizer.step()    

            t.set_postfix(values='loss : {:.2f}'.format(tloss_val))

        self.sample_decoding()
    
    def test(self, epoch, tset='test'):
        self.net.eval()
        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode="tokenizer")
        for data in tqdm.tqdm(loader):
            imgs, targets = data['img'].to(device), \
                data['target'].to(device)
            with torch.no_grad():
                o = self.net(imgs)
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy() #.squeeze()

            for tdec, target in zip(tdecs, targets):
                transcr = self.decode(target.cpu().numpy(), self.classes['i2c']).strip()
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        self.net.train()

    def save(self, epoch):
        print('####################### Saving model at epoch {} #######################'.format(epoch))
        if not os.path.exists('./saved_models'):
            os.makedirs('saved_models')

        torch.save(self.net.cpu().state_dict(), './saved_models/htrnet_{}.pt'.format(epoch))
        self.net.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    args = parse_args()
    config = read_config(args.config)
    max_epochs = 200

    htr_trainer = HTRTrainer()

    cnt = 1
    print('Training Started!')
    for epoch in range(1, max_epochs + 1):
        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()
        # save and evaluate the current model
        if epoch % 50 == 0:
            htr_trainer.save(epoch)
            htr_trainer.test(epoch, 'val')

    # save the final model
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    torch.save(htr_trainer.net.cpu().state_dict(), './saved_models/last.pt')
    