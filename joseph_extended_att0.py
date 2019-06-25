experiments_path = "experiments/path"
experiment_name = "joseph_extended_att0"

# Model parameters
core_model = "joseph_extended"  # [vgg16, resnet50, joseph_basic, joseph_extended]
imagenet_weights = True
early_fusion = 0  # [0: None, n: number of FC dimensions]
multi_loss = False
branches = [0]  # [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]]
loss_weights = [1, 0.8]  # weight for each branch, only for multi_loss training

# Data parameters
data_path = "/data/path"
nb_classes = 5
input_shape = (320, 224)
color_mode = 'rgb'
horizontal_flip = True  # only for training
shuffle = True  # only for training
data_rate = 1.  # only for training

# Training parameters
epochs = 100
batch_size = 64

lr = 0.00005
beta = 0.9
plateau_factor = 0.1
plateau_patience = 2
plateau_min_lr = 0
