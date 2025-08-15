from datetime import datetime
import torch
import os
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import modnet as MODNet
from src import trainer as MODTrainer
from own_dataset import MODNetMetaHumanHairMattingDataset
from torch.utils.tensorboard import SummaryWriter


def main():
    # Parameters form paper https://arxiv.org/pdf/2011.11961
    # Uses pretrained MobileNetV2 weights on the Supervisely Person Segmentation (SPS)
    # MODNet training set has ~3000 portraits. With batch size 16, this results in roughly 180 training steps per epoch
    # If we set batch size to 2, this gives us similar number of updates
    test_set_size = 3
    epochs = 40
    batch_size = 2
    initial_learning_rate = 0.01
    reduce_lr_after_each = 10
    reduce_lr_by_a_factor_of = 0.1

    out_folder = './runs'
    run_folder = os.path.join(out_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
    print('Run folder:', run_folder)
    os.makedirs(run_folder)
    writer = SummaryWriter(log_dir=run_folder)

    # model
    modnet = MODNet.MODNet(backbone_pretrained='./pretrained/mobilenetv2_human_seg.ckpt')
    modnet = modnet.cuda()

    # data
    dataset = MODNetMetaHumanHairMattingDataset(resize=(512, 512))
    train_dataset = Subset(dataset, range(len(dataset) - test_set_size))
    validation_dataset = Subset(dataset, range(len(dataset) - test_set_size, len(dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # optimization
    optimizer = torch.optim.SGD(modnet.parameters(), lr=initial_learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=reduce_lr_after_each, gamma=reduce_lr_by_a_factor_of)

    # Training starts here
    step = 0
    for epoch in range(0, epochs):
        for image, trimap, gt_matte in (pbar := tqdm(train_dataloader)):
            step += 1
            semantic_loss, detail_loss, matte_loss = MODTrainer.supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda())
            pbar.set_postfix_str(f'semantic_loss: {semantic_loss.item():.3f}, detail_loss: {detail_loss:.3f}, matte_loss: {matte_loss:.3f}')
            writer.add_scalar("Loss/semantic", semantic_loss.item(), step)
            writer.add_scalar("Loss/detail", detail_loss.item(), step)
            writer.add_scalar("Loss/matte", matte_loss.item(), step)
            writer.add_scalar("Epoch", epoch, step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        lr_scheduler.step()

        # eval for progress check and save images (here's where u visualize changes over training time)
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(validation_dataset, 'Predicting Validation Set')):
                if epoch == 0:
                    torchvision.utils.save_image(sample[0] * 0.5 + 0.5, os.path.join(run_folder, "eval_%g_%g_im.jpg" % (idx, 0)))
                    torchvision.utils.save_image(sample[2] * 0.5 + 0.5, os.path.join(run_folder, "eval_%g_%g_gt.jpg" % (idx, 0)))
                    torchvision.utils.save_image(sample[1], os.path.join(run_folder, "eval_%g_%g_trimap.jpg" % (idx, 0)))
                _,_,debugImages = modnet(sample[0][None].cuda(),True)
                torchvision.utils.save_image(debugImages[0], os.path.join(run_folder,"eval_%g_%g.jpg"%(idx, epoch+1)))
if __name__ == '__main__':
    main()
