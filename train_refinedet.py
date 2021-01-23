import os
import logging
import time
import torch
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import argparse
from tensorboardX import SummaryWriter
from utils.augmentations import SSDAugment
from utils.refinedet_multibox_loss import RefineDetMultiBoxLoss
from utils.voc0712 import VOC_ROOT, VOCDetection
from utils.config import voc_refinedet, MEANS


class Trainer:
    def __init__(self):
        self.args = self.init_parser()
        self.set_tensor_type()
        self.cfg, self.dataset, self.dataloader = self.init_dataset()
        self.net, self.optimizer, self.arm_cri, self.odm_cri, self.writer = self.build_net()

    def set_tensor_type(self):
        if torch.cuda.is_available() and self.args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    def adjust_learning_rate(self, gamma, step):
        lr = self.args.lr * (gamma ** step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def init_parser():
        def str2bool(v: str):
            return v.lower() in ("yes", "true", "t", "1")

        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'], type=str, help='VOC or COCO')
        parser.add_argument('--input_size', default='320', choices=['320', '512'],
                            type=str, help='RefineDet320 or RefineDet512')
        parser.add_argument('--dataset_root', default=VOC_ROOT)
        parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth', help='Pretrained base model')
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--resume', default=None, type=str)
        parser.add_argument('--start_iter', default=0, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--cuda', default=True, type=str2bool)
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
        parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
        parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
        parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
        parser.add_argument('--save_folder', default='weights')
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--model_type', default='refinedet')
        args = parser.parse_args()
        return args

    @staticmethod
    def detection_collate(batch):
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
        return torch.stack(imgs, 0), targets

    def init_dataset(self):
        if self.args.dataset == 'COCO':
            raise NotImplementedError
        elif self.args.dataset == 'VOC':
            cfg = voc_refinedet[self.args.input_size]
            dataset = VOCDetection(root=self.args.dataset_root, transform=SSDAugment(cfg['min_dim'], MEANS), use_buf=False)
        else:
            raise NotImplementedError
        data_loader = data.DataLoader(dataset, self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      shuffle=True, collate_fn=self.detection_collate,
                                      pin_memory=True)
        return cfg, dataset, data_loader

    def build_net(self):
        if self.args.model_type == 'refinedet':
            from models.refinedet import build_refinedet
            net = build_refinedet('train', self.cfg['min_dim'], self.cfg['num_classes'])
            print('The model type is refinedet')
        elif self.args.model_type == 'deform':
            from models.refinedet_deform import build_refinedet_deform
            net = build_refinedet_deform('train', self.cfg['min_dim'], self.cfg['num_classes'])
            print('The model type is refinedet-deformable1')
        elif self.args.model_type == 'modulation':
            from models.refinedet_modulation import build_refinedet_modulation
            net = build_refinedet_modulation('train', self.cfg['min_dim'], self.cfg['num_classes'])
            print('The model type is refinedet-modulation1')
        else:
            from models.refinedet import build_refinedet
            net = build_refinedet('train', self.cfg['min_dim'], self.cfg['num_classes'])
            print('The model type is refinedet')

        if self.args.cuda:
            # net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if self.args.resume:
            logging.info(f'Resuming training, loading {self.args.resume}...')
            net.load_weights(self.args.resume)
        else:
            vgg_weights = torch.load(self.args.basenet)
            logging.info('Loading base network...')
            net.vgg.load_state_dict(vgg_weights)

        if self.args.cuda:
            net = net.cuda()

        if not self.args.resume:
            logging.info('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            net.extras.apply(self.weights_init)
            net.arm_loc.apply(self.weights_init)
            net.arm_conf.apply(self.weights_init)
            net.odm_loc.apply(self.weights_init)
            net.odm_conf.apply(self.weights_init)

            net.tcb0.apply(self.weights_init)
            net.tcb1.apply(self.weights_init)
            net.tcb2.apply(self.weights_init)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        arm_cri = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, self.args.cuda)
        odm_cri = RefineDetMultiBoxLoss(self.cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False,
                                        self.args.cuda, use_arm=True)
        return net, optimizer, arm_cri, odm_cri, SummaryWriter()

    @staticmethod
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()
        else:
            pass

    def train(self):
        if not os.path.exists(self.args.save_folder):
            os.mkdir(self.args.save_folder)

        self.net.train()
        arm_loc_loss, arm_conf_loss, odm_loc_loss, odm_conf_loss = 0.0, 0.0, 0.0, 0.0
        logging.info(f'Training RefineDet on: {self.dataset.name}')
        logging.info(f'Using the specified args: {self.args}')

        step_index = 0
        batch_iterator = iter(self.dataloader)
#         for iteration in range(self.args.start_iter, self.cfg['max_iter']):
        for iteration in range(self.args.start_iter, 200000):  
            if iteration in self.cfg['lr_steps']:
                step_index += 1
                self.adjust_learning_rate(self.args.gamma, step_index)

            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self.dataloader)
                images, targets = next(batch_iterator)

            if self.args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]

            t0 = time.time()
            out = self.net(images)
            # backprop
            self.optimizer.zero_grad()
            arm_loss_l, arm_loss_c = self.arm_cri(out, targets)
            odm_loss_l, odm_loss_c = self.odm_cri(out, targets)

            arm_loss = arm_loss_l + arm_loss_c
            odm_loss = odm_loss_l + odm_loss_c
            loss = arm_loss + odm_loss
            loss.backward()
            self.optimizer.step()
            t1 = time.time()
            arm_loc_loss += arm_loss_l.item()
            arm_conf_loss += arm_loss_c.item()
            odm_loc_loss += odm_loss_l.item()
            odm_conf_loss += odm_loss_c.item()

            if iteration % self.args.print_freq == 0:
                logging.info("iter %d. ARM_L = %.4f, ARM_C = %.4f, ODM_L = %.4f, ODM_C = %.4f, timer: %.4f sec" % (iteration,
                             arm_loc_loss / 10, arm_conf_loss / 10, odm_loc_loss / 10, odm_conf_loss / 10, (t1 - t0)))
                self.writer.add_scalar("Loss", (arm_loc_loss + arm_conf_loss + odm_loc_loss + odm_conf_loss) / 10, iteration)
                self.writer.add_scalar("ARM_L", arm_loc_loss / 10, iteration)
                self.writer.add_scalar("ARM_C", arm_conf_loss / 10, iteration)
                self.writer.add_scalar("ODM_L", odm_loc_loss / 10, iteration)
                self.writer.add_scalar("ODM_C", odm_conf_loss / 10, iteration)
                arm_loc_loss, arm_conf_loss, odm_loc_loss, odm_conf_loss = 0.0, 0.0, 0.0, 0.0

            if iteration != 0 and iteration % 5000 == 0:
                logging.info('Saving state, iter:', iteration)
                torch.save(self.net.state_dict(),
                           os.path.join(self.args.save_folder,
                                        f'RefineDet{self.args.input_size}_{self.args.dataset}_{self.args.model_type}_{iteration}.pth'))

        self.writer.close()
        torch.save(self.net.state_dict(),
                   os.path.join(self.args.save_folder,
                                f'RefineDet{self.args.input_size}_{self.args.dataset}_{self.args.model_type}_final.pth'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    trainer = Trainer()
    trainer.train()
