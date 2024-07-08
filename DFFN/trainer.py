from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import cv2
import pdb

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':

            if self.args.begin_test_epoch == 1:
                self.optimizer.load(ckp.dir, epoch=len(ckp.log))
            else:
                self.optimizer.load(ckp.dir, epoch=self.args.load_epoch)

        # self.early_drop = np.linspace(args.drop_rate, 0, args.cutoff_epoch)
        # self.cutoff_epoch = args.cutoff_epoch
        self.error_last = 1e8
        self.ckp.write_log(str(self.model), print_log = False)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def train(self):

        self.loss.step()
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch + 1, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        # if epoch < self.cutoff_epoch:
        #     self.model.update_dropout(self.early_drop[epoch])
        # else:
        #     self.model.update_dropout(0.)

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        # pdb.set_trace()
        if not self.begin_test():
            # pdb.set_trace()
            self.ckp.save(self, epoch, 10, is_best=False)
            torch.set_grad_enabled(True)
            return;
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                eval_acc = 0
                eval_acc_ssim = 0
                PSNR_values = []
                SSIM_values = []
                image_names = []
                time_list = []
                for lr, hr, filename in tqdm(d, ncols=80):
                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)
                    start.record()
                    sr= self.model(lr, idx_scale)
                    end.record()
                    torch.cuda.synchronize()
                    time_list.append(start.elapsed_time(end))  # milliseconds
                    # pdb.set_trace()
                    if filename[0] in ['1'] and self.args.rgb_range == 1:
                        output = sr
                        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if output.ndim == 3:
                            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                        cv2.imwrite(self.args.save+'/'+ filename[0]+'.png', output)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    eval_acc += psnr
                    PSNR_values.append(psnr)
                    ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)
                    eval_acc_ssim += ssim
                    SSIM_values.append(ssim)
                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    # self.ckp.log[-1, idx_data, idx_scale] += ssim
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results and self.args.rgb_range == 255:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                # pdb.set_trace()
                image_names.append('average')
                PSNR_values.append(eval_acc / len(d))
                SSIM_values.append(eval_acc_ssim / len(d))
                if self.args.load == '.' or self.args.load == '':
                    xlsx_file_name = self.args.save + '/results-' + d.dataset.name + '/' + d.dataset.name + '.xlsx'
                else:
                    xlsx_file_name = self.args.load + '/results-' + d.dataset.name + '/' + d.dataset.name + '.xlsx'
                image_names = np.array(image_names,dtype=object)
                PSNR_values = np.array(PSNR_values,dtype=object)
                SSIM_values = np.array(SSIM_values,dtype=object)
                PSNR_values = [round(i, 2) for i in PSNR_values]
                SSIM_values = [round(i, 4) for i in SSIM_values]
                data = {
                    'image': image_names,
                    'psnr': PSNR_values,
                    'ssim': SSIM_values
                }
                df = DataFrame(data)
                df.to_excel(xlsx_file_name)
                # pdb.set_trace()
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + self.args.begin_test_epoch
                    )
                )
                # self.ckp.write_log(
                #     '[{} x{}]\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                #         d.dataset.name,
                #         scale,
                #         self.ckp.log[-1, idx_data, idx_scale],
                #         best[0][idx_data, idx_scale],
                #         best[1][idx_data, idx_scale] + 1
                #     )
                # )

            if not self.args.test_only:
                self.ckp.save(self, epoch, idx_data, is_best=(best[1][idx_data, 0] + self.args.begin_test_epoch == epoch))

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log('Forward: {:.2f}s, meanRunTime: {:.2f}ms\n'.format(timer_test.toc(),np.mean(time_list)))
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            
            if epoch == self.args.switch_sgd_epoch and self.args.optimizer == 'ADAM':
                self.args.optimizer = 'SGD'
            return epoch >= self.args.epochs

    def begin_test(self):
        if self.args.test_only:
            # self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.begin_test_epoch