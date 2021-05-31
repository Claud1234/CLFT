import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
#import config

from tensorboardX import SummaryWriter

label_colors_list = [
        (255, 0, 0), # animal
        (0, 255, 0), # archway
        (0, 0, 255),] # bicyclist]

# all the classes that are present in the dataset
ALL_CLASSES = ['sign', 'bicyclist', 'background']

"""
This (`class_values`) assigns a specific class label to the each of the classes.
For example, `animal=0`, `archway=1`, and so on.
"""
class_values = [ALL_CLASSES.index(cls.lower()) for cls in ALL_CLASSES]

class TensorboardWriter():
    def __init__(self):
        super(TensorboardWriter, self).__init__()
    # initilaize `SummaryWriter()`
        self.writer = SummaryWriter()
    def tensorboard_writer(self, loss, mIoU, pix_acc, iterations, phase=None):
        if phase == 'train':
            self.writer.add_scalar('Train Loss', loss, iterations)
            self.writer.add_scalar('Train mIoU', mIoU, iterations)
            self.writer.add_scalar('Train Pixel Acc', pix_acc, iterations)
        if phase == 'valid':
            self.writer.add_scalar('Valid Loss', loss, iterations)
            self.writer.add_scalar('Valid mIoU', mIoU, iterations)
            self.writer.add_scalar('Valid Pixel Acc', pix_acc, iterations)

def get_label_mask(mask, class_values): 
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def draw_seg_maps(data, output, epoch, i):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 0.6 # how much transparency
    beta = 1 - alpha # alpht + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = data[0]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255 # else OpenCV will save black image


    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)
    
    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = seg_map == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            blue_map[idx] = np.array(label_colors_list)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(rgb, alpha, image, beta, gamma, image)
    cv2.imwrite(f"train_seg_maps/e{epoch}_b{i}.jpg", image)

def draw_test_segmentation_map(outputs):
    """
    This function will apply color mask as per the output that we
    get when executing `test.py` or `test_vid.py` on a single image 
    or a video. NOT TO BE USED WHILE TRAINING OR VALIDATING.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = labels == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            blue_map[idx] = np.array(label_colors_list)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    """
    This function will apply an overlay of the output segmentation
    map on top of the orifinal input image. MAINLY TO BE USED WHEN
    EXECUTING `test.py` or `test_vid.py`.
    """
    alpha = 0.6 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image

def visualize_from_dataloader(data_loader): 
    """
    Helper function to visualzie the data from 
    dataloaders. Only executes if `DEBUG` is `True` in
    `config.py`
    """
    data = iter(data_loader)   
    images, labels = data.next()
    image = images[1]
    # image = np.array(image, dtype='uint8') # use this if not normalizing, which is probably never
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    mean = np.array([0.45734706, 0.43338275, 0.40058118])
    std = np.array([0.23965294, 0.23532275, 0.2398498])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    label = labels[1]
    images = [image, label.squeeze()]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image)
    plt.show()

def visualize_from_path(image_path, seg_path):
    """
    Helper function to visualize image and segmentation maps after
    reading from the images from path.
    Only executes if `DEBUG` is `True` in
    `config.py`
    """
    train_sample_img = cv2.imread(image_path[0])
    train_sample_img = cv2.cvtColor(train_sample_img, cv2.COLOR_BGR2RGB)
    train_sample_seg = cv2.imread(seg_path[0])
    train_sample_seg = cv2.cvtColor(train_sample_seg, cv2.COLOR_BGR2RGB)
    images = [train_sample_img, train_sample_seg]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image)
    plt.show()

def save_model_dict(model, epoch, optimizer, 
                    criterion, valid_iters, train_iters):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'valid_iters': valid_iters, 
            'train_iters': train_iters, 
            }, f"checkpoints/model_{epoch}.pth")
