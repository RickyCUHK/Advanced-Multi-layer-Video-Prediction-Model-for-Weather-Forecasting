import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import time
from models.models import EncoderRNN
from data1.moving_mnist import MovingMNIST
import argparse
import os
import cv2
# from dataset import KITTI_train,KITTI_test
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
import preprocess
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help='device for training and testing')
parser.add_argument('--root', type=str, default='data', help='folder for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--checkpoint_path', type=str, default='', help='folder for dataset')
parser.add_argument('--lr', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--n_epochs', type=int, default=700, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=5, help='')
parser.add_argument('--save_dir', type=str, default='/home/huangriqi/Multi_layer_model_checkpoints')
parser.add_argument('--gen_frm_dir', type=str, default='/home/huangriqi/Multi_layer_model_results')
parser.add_argument('--patch_size', type=int, default=4, help='')
parser.add_argument('--train_data_paths', type=str, default='/data/data1/shuliang/train_radar')
parser.add_argument('--valid_data_paths', type=str, default='/data/data1/shuliang/test_radar/train000.npy')



# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=1 / 50000)
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_height', type=int, default=64)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=16)

args = parser.parse_args()

test_Datase = TestDataset(args.valid_data_paths)
test_loader = DataLoader(dataset=test_Datase, batch_size=args.batch_size, shuffle= False,
                          num_workers=4, pin_memory=False, drop_last=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((16, 64, 64))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.contiguous().view(np.prod((args.batch_size, 10)), -1)
        validity = self.model(img_flat)
        
        return validity


# Initialize
discriminator = Discriminator()
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
adversarial_loss = torch.nn.MSELoss()
discriminator.cuda()
adversarial_loss.cuda()
Tensor = torch.cuda.FloatTensor


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length-1,
                      args.img_height, args.img_width,
                      args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length-1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height, args.img_width, args.img_channel))
    zeros = np.zeros((args.img_height, args.img_width, args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length-1,
                                  args.img_height, args.img_width,
                                  args.img_channel))
    return eta, real_input_flag


def train_on_batch(input_tensor, target_tensor, mask, encoder, encoder_optimizer, criterion):
    encoder.train()
    mask = torch.FloatTensor(mask).permute(0, 1, 4, 2, 3).contiguous().to(args.device)
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    
    # Adversarial ground truths
    valid = Variable(Tensor(np.prod((args.batch_size, input_length)) , 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(np.prod((args.batch_size, input_length)) , 1).fill_(0.0), requires_grad=False)

    encoder_optimizer.zero_grad()
    loss = 0.0


    H_queue = [] 
    C_queue = [] 
    channel =16*2
    shape = 64

    for i in range(4):
        in_channel = channel*2**(i)
        in_shape = shape//2**(i)
        H_queue.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
        C_queue.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))


    encoder_H = [] 
    encoder_C = [] 
    channel =16*2
    shape = 64

    for i in range(4):
        in_channel = channel*2**(i)
        in_shape = shape//2**(i)
        encoder_H.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
        encoder_C.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))

    decoder_H = [] 
    decoder_C = [] 
    channel =16*2
    shape = 64

    for i in range(4):
        in_shape = shape//2**(i)
        in_channel = channel*2**(i)
        
        decoder_H.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
        decoder_C.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))


    for ei in range(input_length - 1):
        output_image,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C = encoder(input_tensor[:, ei, :, :, :], H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C)
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence

    for di in range(target_length):
        output_image,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C = encoder(decoder_input,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C)
        # print(output_image.shape)

        target = target_tensor[:, di, :, :, :]
        loss += criterion(output_image, target)
        if di !=9:
            decoder_input = target * mask[:, di] + output_image * (1 - mask[:, di])
            # print(decoder_input.shape)

    g_loss = adversarial_loss(discriminator(torch.stack((output_image,)*10, dim = 0)), valid)
    loss += 0.001 * g_loss;  

    loss.backward()
    encoder_optimizer.step()

    optimizer_D.zero_grad()
    real_imgs = Variable(target_tensor.type(Tensor))
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(torch.stack((output_image.detach(),)*10, dim = 0)), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    return loss.item() / target_length, g_loss, d_loss


def trainIters(encoder, n_epochs, print_every, eval_every):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=1, factor=0.9, verbose=True)
    criterion = nn.MSELoss()
    itr = 0
    eta = args.sampling_start_value
    for epoch in range(0, n_epochs):
        t0 = time.time()
        loss_epoch = 0
        for data_id in range(22):
            train_Datase = TrainDataset(args.train_data_paths+'/train'+str(data_id).zfill(3)+'.npy')
            train_loader = DataLoader(dataset=train_Datase, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=False, drop_last=True)
            for i, out in enumerate(train_loader, 0):
                itr += 1
               
                # print(itr)
                # input_tensor = (out[:,:5]/255.0).permute(0, 1, 4, 2, 3).contiguous().float().to(args.device)
                # target_tensor =(out[:,5:]/255.0).permute(0, 1, 4, 2, 3).contiguous().float().to(args.device)


                input_tensor = out[:, 0:10,:,120:376,160:416].float().to(args.device)
                target_tensor = out[:, 10:20,:,120:376,160:416].float().to(args.device) 
                input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
                target_tensor = preprocess.reshape_patch(target_tensor, args.patch_size)
          
                eta, real_input_flag = schedule_sampling(eta, itr)
                loss, g_loss, d_loss = train_on_batch(input_tensor, target_tensor, real_input_flag, encoder, encoder_optimizer, criterion)
                loss_epoch += loss

        train_losses.append(loss_epoch)
        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' epoch time ', ' g_loss: ', g_loss, ' d_loss: ', d_loss, time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse = evaluate(encoder, test_loader,epoch)
            scheduler_enc.step(mse)
            stats = {}
            stats['net_param'] = encoder.state_dict()
            save_dir = os.path.join(args.save_dir, 'epoch-' + str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, 'model.ckpt' + '-' + str(epoch))
            torch.save(stats, checkpoint_path)
    return train_losses




def evaluate(encoder, loader,epoch):

    total_mse, total_mae, total_ssim, total_psnr,total_lpips = 0, 0, 0, 0, 0
    encoder.eval()
    mask = np.zeros(
                (args.batch_size,
                 args.total_length - args.input_length - 1,
                 args.img_height ,
                 args.img_width ,
                 args.img_channel))
    mask = torch.FloatTensor(mask).permute(0, 1, 4, 2, 3).contiguous().to(args.device)            
    with torch.no_grad():
        for id, out in enumerate(loader, 0):
            # input_batch = torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[:, 0:10,:,120:376,160:416].float().to(args.device)
            target_tensor = out[:, 10:20,:,120:376,160:416].float().to(args.device) 
            input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
            target_tensor = preprocess.reshape_patch(target_tensor, args.patch_size)
            # print(input_tensor.shape)
            # print(target_tensor.shape)
            # sys.exit()


            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]


            H_queue = [] 
            C_queue = [] 
            channel =16*2
            shape = 64

            for i in range(4):
                in_channel = channel*2**(i)
                in_shape = shape//2**(i)
                H_queue.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
                C_queue.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))


            encoder_H = [] 
            encoder_C = [] 
            channel =16*2
            shape = 64

            for i in range(4):
                in_channel = channel*2**(i)
                in_shape = shape//2**(i)
                encoder_H.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
                encoder_C.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))

            decoder_H = [] 
            decoder_C = [] 
            channel =16*2
            shape = 64

            for i in range(4):
                in_shape = shape//2**(i)
                in_channel = channel*2**(i)
                
                decoder_H.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
                decoder_C.append(torch.randn(args.batch_size, in_channel, in_shape,in_shape).to(args.device))
            

            for ei in range(input_length - 1):
                output_image,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C = encoder(input_tensor[:, ei, :, :, :],H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C)

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                output_image,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C = encoder(decoder_input,H_queue,C_queue,encoder_H,encoder_C,decoder_H,decoder_C)
                decoder_input = output_image
                predictions.append(output_image)

            predictions_tensor = torch.stack(predictions,1)  # for MM: (10, batch_size, 1, 64, 64)
            # predictions_tensor = predictions.swapaxes(0, 1)  # (batch_size,10, 1, 64, 64)


            # output_image , target_image = encoder (input_tensor,target_tensor,mask)
            # input = input_tensor.cpu().numpy()
            # target = target_tensor.cpu().numpy()
            # predictions_tensor = (output_image[:,-args.input_length:]).detach()
            input = preprocess.reshape_patch_back(input_tensor,args.patch_size)
            input_1 = input.permute(0,1,3,4,2).cpu().numpy()
            input = input.cpu().numpy()
            target = preprocess.reshape_patch_back(target_tensor,args.patch_size)
            target_1 = target.permute(0,1,3,4,2).cpu().numpy()
            target = target.cpu().numpy()
            predictions = preprocess.reshape_patch_back(predictions_tensor,args.patch_size)
            predictions_1 = predictions.permute(0,1,3,4,2).cpu().numpy()
            predictions = predictions.cpu().numpy()


            input1 = preprocess.Change_color(input_1* 80)
            target1 = preprocess.Change_color(target_1* 80)
            predictions1 = preprocess.Change_color(predictions_1* 80)


            # predictions = predictions_tensor.cpu().numpy()
    #         start=id*16
    #         for j in range(5):
    #             template[start:start+16,j,0]=target[:,j]*255
    #             template[start:start+16,j,1]=predictions[:,j]*255
    # template = template.reshape(4112*5,2,3,128,160)
    # np.save('train.npy',template[:20000])
    # np.save('test.npy',template[20000:])

            

            
            

            # save prediction examples
            if id < 20:
                path = os.path.join(args.gen_frm_dir, 'epoch'+str(epoch))
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, str(id).zfill(3))
                if not os.path.exists(path):
                    os.makedirs(path)
                for i in range(10):
                    # name = 'gt' + str(i + 1) + '.png'
                    name = 'gt'+str(i + 1).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(input1[0, i, :, :, :] )
                    cv2.imwrite(file_name, img_gt)

                for i in range(10):
                    # name = 'gt' + str(i + 11) + '.png'
                    name = 'gt'+str(i + 11).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(target1[0, i, :, :, :] )
                    cv2.imwrite(file_name, img_gt)

                for i in range(10):
                    # name = 'pd' + str(i + 1 + 10) + '.png'
                    name = 'pd'+str(i + 11).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = np.uint8(predictions1[0, i, :, :, :])
                    cv2.imwrite(file_name, img_pd)





            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b].transpose(1,2,0), predictions[a, b].transpose(1,2,0),channel_axis = -1) / (target.shape[0] * target.shape[1])
                    # total_psnr += psnr(target[a, b].transpose(1,2,0), predictions[a, b].transpose(1,2,0)) / (target.shape[0] * target.shape[1])
                    # total_lpips += loss_fn_alex(target_tensor[a, b], predictions_tensor[a, b]) / (target_tensor.shape[0] * target_tensor.shape[1])


    print('eval mse ', total_mse / len(loader))
    print('eval mae ', total_mae / len(loader))
    print('eval ssim ',total_ssim / len(loader))
    # print('eval psnr ',total_psnr / len(loader))
    # print('eval lpips ',total_lpips / len(loader))
    return total_mse / len(loader)


def evaluate111(encoder, loader,epoch):

    total_mse, total_mae, total_ssim, total_psnr,total_lpips = 0, 0, 0, 0, 0
    encoder.eval()
    mask = np.zeros(
                (args.batch_size,
                 args.total_length - args.input_length - 1,
                 args.img_height ,
                 args.img_width ,
                 args.img_channel))
    mask = torch.FloatTensor(mask).permute(0, 1, 4, 2, 3).contiguous().to(args.device)  
    # template = np.ones([23808,30,1,384,384],dtype='uint8')
    with torch.no_grad():
        for data_id in range(24):
            template = np.ones([960,30,1,384,384],dtype='uint8')
            train_Datase = TrainDataset(args.train_data_paths+'/train'+str(data_id).zfill(3)+'.npy')
            train_loader = DataLoader(dataset=train_Datase, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=False, drop_last=True)
            # print(len(train_loader))
            for id, out in enumerate(train_loader, 0):

                # input_batch = torch.Size([8, 20, 1, 64, 64])
                input_tensor = out[:, 0:10,:,120:376,160:416].float().to(args.device)
                target_tensor = out[:, 10:20,:,120:376,160:416].float().to(args.device) 
                input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
                target_tensor = preprocess.reshape_patch(target_tensor, args.patch_size)
                # print(input_tensor.shape)
                # print(target_tensor.shape)
                # sys.exit()


                input_length = input_tensor.size()[1]
                target_length = target_tensor.size()[1]
                

                output_image , target_image = encoder (input_tensor,target_tensor,mask)
                # input = input_tensor.cpu().numpy()
                # target = target_tensor.cpu().numpy()
                predictions_tensor = (output_image[:,-args.input_length:]).detach()
                input = preprocess.reshape_patch_back(input_tensor,args.patch_size)
                input_1 = input.permute(0,1,3,4,2).cpu().numpy()
                input = input.cpu().numpy()
                target = preprocess.reshape_patch_back(target_tensor,args.patch_size)
                target_1 = target.permute(0,1,3,4,2).cpu().numpy()
                target = target.cpu().numpy()
                predictions = preprocess.reshape_patch_back(predictions_tensor,args.patch_size)
                predictions_1 = predictions.permute(0,1,3,4,2).cpu().numpy()
                predictions = predictions.cpu().numpy()
                template[id*64:(id+1)*64,0:10,:,:,:]=input*255.0
                template[id*64:(id+1)*64,10:20,:,:,:]=target*255.0
                template[id*64:(id+1)*64,20:30,:,:,:]=predictions*255.0
            np.save('/data/shuliang3/MAU_weather/train_dataset/train'+str(data_id).zfill(2)+'.npy',template)

                # print(predictions.shape)
                # sys.exit()


              

                # input1 = preprocess.Change_color(input_1* 80)
                # target1 = preprocess.Change_color(target_1* 80)
                # predictions1 = preprocess.Change_color(predictions_1* 80)


                # predictions = predictions_tensor.cpu().numpy()
        #         start=id*16
        #         for j in range(5):
        #             template[start:start+16,j,0]=target[:,j]*255
        #             template[start:start+16,j,1]=predictions[:,j]*255
        # template = template.reshape(4112*5,2,3,128,160)
        # np.save('train.npy',template[:20000])
        # np.save('test.npy',template[20000:])

                

                
                

                # save prediction examples
                # if id < 10:
                #     path = os.path.join(args.gen_frm_dir, 'epoch'+str(epoch))
                #     if not os.path.exists(path):
                #         os.makedirs(path)
                #     path = os.path.join(path, str(id).zfill(3))
                #     if not os.path.exists(path):
                #         os.makedirs(path)
                #     for i in range(10):
                #         # name = 'gt' + str(i + 1) + '.png'
                #         name = 'gt'+str(i + 1).zfill(2) + '.png'
                #         file_name = os.path.join(path, name)
                #         img_gt = np.uint8(input1[0, i, :, :, :] )
                #         cv2.imwrite(file_name, img_gt)

                #     for i in range(10):
                #         # name = 'gt' + str(i + 11) + '.png'
                #         name = 'gt'+str(i + 11).zfill(2) + '.png'
                #         file_name = os.path.join(path, name)
                #         img_gt = np.uint8(target1[0, i, :, :, :] )
                #         cv2.imwrite(file_name, img_gt)

                #     for i in range(10):
                #         # name = 'pd' + str(i + 1 + 10) + '.png'
                #         name = 'pd'+str(i + 11).zfill(2) + '.png'
                #         file_name = os.path.join(path, name)
                #         img_pd = np.uint8(predictions1[0, i, :, :, :])
                #         cv2.imwrite(file_name, img_pd)

                





        #         mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
        #         mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
        #         total_mse += mse_batch
        #         total_mae += mae_batch

        #         for a in range(0, target.shape[0]):
        #             for b in range(0, target.shape[1]):
        #                 total_ssim += ssim(target[a, b].transpose(1,2,0), predictions[a, b].transpose(1,2,0),channel_axis = -1) / (target.shape[0] * target.shape[1])
        #                 # total_psnr += psnr(target[a, b].transpose(1,2,0), predictions[a, b].transpose(1,2,0)) / (target.shape[0] * target.shape[1])
        #                 # total_lpips += loss_fn_alex(target_tensor[a, b], predictions_tensor[a, b]) / (target_tensor.shape[0] * target_tensor.shape[1])


        # print('eval mse ', total_mse / len(loader))
        # print('eval mae ', total_mae / len(loader))
        # print('eval ssim ',total_ssim / len(loader))
        # # print('eval psnr ',total_psnr / len(loader))
        # # print('eval lpips ',total_lpips / len(loader))
        # return total_mse / len(loader)


print('BEGIN TRAIN')
model = EncoderRNN(args)
encoder = nn.DataParallel(model).cuda()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('encoder ', count_parameters(encoder))

if args.checkpoint_path != '':
    print('load model:', args.checkpoint_path)
    stats = torch.load(args.checkpoint_path)
    encoder.load_state_dict(stats['net_param'])
    plot_losses = trainIters(encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
    mse = evaluate(encoder, test_loader,99999213199999)
else:
    # evaluate(encoder, test_loader,9999999)
    plot_losses = trainIters(encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)

