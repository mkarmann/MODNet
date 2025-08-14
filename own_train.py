import torch
import os
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import modnet as MODNet
from src import trainer as MODTrainer
from own_dataset import MetaHumanHairMattingDataset, MODNetMetaHumanHairMattingDataset


def main():
    # Parameters form paper https://arxiv.org/pdf/2011.11961
    # Uses pretrained MobileNetV2 weights on the Supervisely Person Segmentation (SPS)
    epochs = 40
    batch_size = 16
    initial_learning_rate = 0.01
    reduce_lr_after_each = 10
    reduce_lr_by_a_factor_of = 0.1

    # model
    modnet = MODNet.MODNet(backbone_pretrained=False)
    modnet = modnet.cuda()

    # data
    dataset = MODNetMetaHumanHairMattingDataset(resize=(512, 512))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimization
    optimizer = torch.optim.SGD(modnet.parameters(), lr=initial_learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=reduce_lr_after_each, gamma=reduce_lr_by_a_factor_of)

    # Training starts here
    for epoch in range(0, epochs):
        for image, trimap, gt_matte in (pbar := tqdm(train_dataloader)):
            semantic_loss, detail_loss, matte_loss = MODTrainer.supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda())
            pbar.set_postfix_str(f'semantic_loss: {semantic_loss.item():.3f}, detail_loss: {detail_loss:.3f}, matte_loss: {matte_loss:.3f}')
        lr_scheduler.step()

        # eval for progress check and save images (here's where u visualize changes over training time)
        # with torch.no_grad():
        #     _,_,debugImages = modnet(testImages.cuda(),True)
        #     for idx, img in enumerate(debugImages):
        #         saveName = "eval_%g_%g.jpg"%(idx,epoch+1)
        #         torchvision.utils.save_image(img, os.path.join(evalPath,saveName))

        print("Epoch done: " + str(epoch))

if __name__ == '__main__':
    main()
