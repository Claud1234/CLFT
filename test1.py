import configs
import numpy as np
from utils.lidar_process import LidarProcess
from utils.image_augment import ImageProcess
a = configs.LIDAR_MEAN
b = configs.LIDAR_STD

mean_lidar = np.array([-0.17263354, 0.85321806, 24.5527253])
std_lidar = np.array([7.34546552, 1.17227659, 15.83745082])
# print (np.array(a).shape)
# print(np.array(a))
# print(np.array(a) == mean_lidar)

# print (configs.TEST_IMAGE)
# if configs.AUGMENT == 'square_crop':
    # print ('sasa')
# else:
    # print ('ssss')    
cam_path = '/home/claude/1.png'
rgb = ImageProcess(cam_path).square_crop()
print (rgb.size)


'''
#### Co-training
'''
# for epoch in range(args.start_epoch, args.epochs_cotrain):
    # if args.distributed:
        # train_sampler.set_epoch(epoch)
        # semi_sampler.set_epoch(epoch)
        #
    # curr_lr = adjust_learning_rate_semi(optimizer, epoch, 
                                        # args.epochs_cotrain, args)
                                        #
    # # train for one epoch
    # train_semi(train_loader, semi_loader, model, criterion, optimizer, args)   
    # print('Semi -- Epoch: {:.0f}, LR: {:.6f}'.format(epoch, curr_lr))
    # if (epoch+1) % save_epoch == 0 and epoch > 0:
        # if not args.multiprocessing_distributed or \
        # (args.multiprocessing_distributed and 
         # args.rank % ngpus_per_node == 0):
            # save_checkpoint(
                # {'epoch': epoch + 1,
                # 'state_dict': model.state_dict(),
                # 'optimizer' : optimizer.state_dict(),}, 
                # is_best=False, 
                # filename=logdir+'checkpoint_cotrain_{:04d}.pth.tar'.format(
                                                                    # epoch))

def train_semi(train_loader, semi_loader, model, criterion, optimizer, args):
    lambda_cot = 1
    print('semi_train')
    model.train()
    for batch in train_loader:
        # measure data loading time
        if args.gpu is not None:
            batch['rgb'] = batch['rgb'].cuda(args.gpu, non_blocking=True)
            batch['lidar'] = batch['lidar'].cuda(args.gpu, non_blocking=True)
            batch['annotation'] = batch['annotation'].cuda(
                                        args.gpu, non_blocking=True).squeeze(1)

        output = model(batch['rgb'], batch['lidar'], 'all')

        loss_rgb = criterion(output['rgb'], batch['annotation'])
        loss_lidar = criterion(output['lidar'], batch['annotation'])
        loss_fusion = criterion(output['fusion'], batch['annotation'])
        loss = loss_rgb + loss_lidar + loss_fusion

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:', loss)

        try:
            batch = next(unsuper_dataloader)[1]
        except:
            unsuper_dataloader = enumerate(semi_loader)
            batch = next(unsuper_dataloader)[1]

        if args.gpu is not None:
            batch['rgb'] = batch['rgb'].cuda(args.gpu, non_blocking=True)
            batch['lidar'] = batch['lidar'].cuda(args.gpu, non_blocking=True)
            batch['annotation'] = batch['annotation'].cuda(
                                args.gpu, non_blocking=True).squeeze(1)

        with torch.no_grad():
            model.eval()
            output = model(batch['rgb'], batch['lidar'], 'fusion')
            annotation_teacher = F.softmax(output['fusion'], 1)
            _, annotation_teacher = torch.max(annotation_teacher, 1)
            mask_not_valid = batch['annotation'] == 3
            annotation_teacher[mask_not_valid] = 3

        model.train()
        output = model(batch['rgb'], batch['lidar'], 'ind')
        loss_rgb = lambda_cot*criterion(output['rgb'],
                                        annotation_teacher.detach().clone())
        loss_lidar = lambda_cot*criterion(output['lidar'],
                                          annotation_teacher.detach().clone())
        loss_unsuper = loss_rgb + loss_lidar

        # output = model(batch['rgb'], batch['lidar'],'all')

        # loss_rgb = criterion(output['rgb'], batch['annotation'])
        # loss_lidar = criterion(output['lidar'], batch['annotation'])
        # loss_fusion = criterion(output['fusion'], batch['annotation'])
        # loss_unsuper = loss_rgb+loss_lidar+loss_fusion

        optimizer.zero_grad()
        loss_unsuper.backward()
        optimizer.step()


def evaluation(train_dataset, model, criterion, optimizer, args):
    model.eval()
    print('evaluation')
    with torch.no_grad():
        for batch in train_dataset:
            output = model(batch['rgb'], batch['lidar'], 'ind')
        # annotation_teacher = F.softmax(output['fusion'], 1)
        # _, annotation_teacher = torch.max(annotation_teacher, 1)
        # mask_not_valid = batch['annotation'] == 3
        # annotation_teacher[mask_not_valid] = 3
        # print('output:', output)
        # print(output['rgb'].shape)
        # create a color pallette, selecting a color for each class
        # from PIL import Image
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")

        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(output['rgb'].byte().cpu().numpy()).resize(rgb.size)
        # r.putpalette(colors)
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(r)
        # plt.show()
