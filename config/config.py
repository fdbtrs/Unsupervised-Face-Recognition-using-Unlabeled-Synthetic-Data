architecture = "resnet50"

batch_size = 128
workers = 8

auto_schedule = False
learning_rate = 0.1
epochs = 200 if auto_schedule else 40
schedule = [8, 16, 24, 32]

output_dir = "output/disco100K_test"
resume_epoch = 0

momentum = 0.9
weight_decay = 5e-4

dataset = "synthetic"  # synthetic, real
number_of_images = 100000

# h_flip, num_mag_exp, randaug_4_16, aug_operation_exp, moco, disco, disco_HF
augmentation = "disco"

"""
MoCo specific configs:
"""
# feature dimension
moco_dim = 512
# queue size; number of negative keys
moco_k = 32768
# should queue samples be replaced by the most similar / dissimilar keys?
queue_type = "normal"  # normal, simQ, dissimQ
# reset class queue every x epochs
reset_queue_every = 1
# moco momentum of updating key encoder (default: 0.999)
moco_m = 0.999
# softmax temperature (default: 0.07)
moco_t = 0.07
drop_ratio = 0.0
# margin value for loss function
loss_margin = 0.0

"""
Options for moco v2
"""
# use mlp head
mlp = False
# use cosine lr schedule
cos = False

eval_datasets = "/data/fboutros/faces_emore"
val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]
datapath = ""
lmark_dir = ""
if dataset == "synthetic":
    datapath = "/data/maklemt/synthetic_imgs/DiscoFaceGAN_aligned"
elif dataset == "real":
    datapath = "/data/maklemt/CASIA_WebFace/preprocessed/imgs_face"
