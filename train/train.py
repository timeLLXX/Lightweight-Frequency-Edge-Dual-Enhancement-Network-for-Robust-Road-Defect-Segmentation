import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset



import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast
import logging
import sys
import time
import warnings
from tqdm import tqdm  # 导入 tqdm 库




warnings.filterwarnings("ignore")
    
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument('--save_path', type=str, default="",
                    help="path to store the checkpoint", )
parser.add_argument("--model_type", type=str, default="edge_sam",
                    help="sam2_model")
parser.add_argument("--checkpoint_path", type=str, default=r"",#root:edge_sam_3x.pth
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, default=r"",
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, default=r"",
                    help="path to the mask file for training")
parser.add_argument("--val_image_path", type=str, default=r"",
                    help="path to the image that used to val the model")
parser.add_argument("--val_mask_path", type=str, default=r"",
                    help="path to the mask file for val")



parser.add_argument("--num_classes", type=int, default=4, help="number classes")
parser.add_argument("--ignore_class_index", type=int, default=255, help="number classes")
parser.add_argument("--image_size", type=int, default=[640,480], help="crop_image size")

parser.add_argument("--epoch", type=int, default=200, help="training epochs")
parser.add_argument("--patience", type=int, default=200, help="early_stoping")#已弃用
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)


def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True




def main(args):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log_file_path = os.path.join(args.save_path, "log.txt")

    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            pass  


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info(str(args))

    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.image_size,'train')
    val_dataset =FullDataset(args.val_image_path, args.val_mask_path, args.image_size, 'private_val')#val

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    model = EdgeSAMUNet(args.num_classes, args.checkpoint_path,args.model_type)
    # 初始化模型权重
    # model.apply(init_weights)


    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    loss_f=structure_loss()

    optim = AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)


    os.makedirs(args.save_path, exist_ok=True)

    train_epoch_losses = []
    val_epoch_losses = []

    scaler = GradScaler()

    # 早停法相关变量
    best_loss = float('inf')

    trigger_times = 0
    best_model_path = os.path.join(args.save_path, 'best_model.pth')

    for epoch in range(args.epoch):
        # 训练模式
        start_train_time = time.time()
        running_train_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epoch}"):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.squeeze(1).to(device)
            optim.zero_grad()
            with autocast():

                pred0, pred1, pred2 = model(x)
                loss0 = loss_f(pred0, target)
                loss1 = loss_f(pred1, target)
                loss2 = loss_f(pred2, target)
                loss = loss0 + loss1 + loss2


            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_epoch_losses.append(avg_train_loss)

        # 验证模式
        running_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{args.epoch}"):
                x = batch['image']
                target = batch['label'].squeeze(1).float().to(device)
                x = x.to(device)

                with autocast():

                    pred0, pred1, pred2 = model(x)
                    loss0 = loss_f(pred0, target)
                    loss1 = loss_f(pred1, target)
                    loss2 = loss_f(pred2, target)
                    loss = loss0 + loss1 + loss2

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_epoch_losses.append(avg_val_loss)
        end_train_time = time.time()
        epoch_time = end_train_time - start_train_time
        logging.info(
            f'Epoch [{epoch + 1}/{args.epoch}], Average Training Loss: {avg_train_loss:.4f}, Average Eval Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds')#

        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch+1
            #trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Best model saved,\t, Eval Loss: {best_loss:.4f}')

        logging.info(f'Best epoic: {best_epoch},\t,Best eval Loss: {best_loss:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))

    # 绘制训练集和测试集的 loss 曲线
    plt.figure()
    plt.plot(range(1, epoch + 2), train_epoch_losses, 'r-', label='Training Loss')
    plt.plot(range(1, epoch + 2), val_epoch_losses, 'b-', label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Eval Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, 'loss_curve.png'))

    logging.info(f'Training completed in {total_time_str}')


if __name__ == "__main__":
    seed_torch(1024)
    from model.edgesamABCMiloracpr  import EdgeSAMUNet, structure_loss
    args.save_path=r""
    main(args)


