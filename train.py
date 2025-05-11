import os
import random
from tqdm import tqdm

import torch

from datasets.imagenet import ImageNet
from datasets.dtd import DescribableTextures
from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.stanford_cars import StanfordCars
from datasets.eurosat import EuroSAT
from datasets.sun397 import SUN397
from datasets.ucf101 import UCF101
from datasets.food101 import Food101
from datasets.caltech101 import Caltech101
from datasets.fgvc import FGVCAircraft
from datasets import dataset_list
import clip
from utils import *
from clip.moco import load_moco
from clip.amu import *
from parse_args import parse_args
from datasets import build_dataset
from datasets.utils import build_data_loader, DatasetWrapper
import torchvision.transforms as transforms


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_one_epoch(model, data_loader, optimizer, scheduler, logger):
    # Train
    model.train()
    model.apply(freeze_bn)  # freeze BN-layer
    correct_samples, all_samples = 0, 0
    loss_list = []
    loss_aux_list = []
    loss_merge_list = []

    # origin image
    for i, (images, target) in enumerate(tqdm(data_loader)):
        images, target = images.cuda(), target.cuda()
        return_dict = model(images, labels=target)

        acc = cls_acc(return_dict['logits'], target)
        correct_samples += acc / 100 * len(return_dict['logits'])
        all_samples += len(return_dict['logits'])

        loss_list.append(return_dict['loss'].item())
        loss_aux_list.append(return_dict['loss_aux'].item())
        loss_merge_list.append(return_dict['loss_merge'].item())

        optimizer.zero_grad()
        return_dict['loss'].backward()
        optimizer.step()
        scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    logger.info('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                         correct_samples, all_samples,
                                                                         sum(loss_list) / len(loss_list)))
    logger.info("""Loss_aux: {:.4f}, Loss_merge: {:.4f}""".format(sum(loss_aux_list) / len(loss_aux_list),
                                                                  sum(loss_merge_list) / len(loss_merge_list)))


def train_and_eval(f, args, logger, model, clip_test_features,
                   aux_test_features, test_labels, train_loader_F):
    model.cuda()
    model.requires_grad_(False)
    model.aux_adapter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=0.01,
        lr=args.lr,
        eps=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0

    for train_idx in range(1, args.train_epoch + 1):
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
        train_one_epoch(model, train_loader_F, optimizer, scheduler, logger)
        # Eval
        model.eval()
        with torch.no_grad():
            return_dict = model(
                clip_features=clip_test_features,
                aux_features=aux_test_features,
                # aux_features=clip_test_features,
                labels=test_labels
            )
            acc = cls_acc(return_dict['logits'], test_labels)
            acc_aux = cls_acc(return_dict['aux_logits'], test_labels)
        logger.info("----- Aux branch's Test Acc: {:.2f} ----".format(acc_aux))
        logger.info("----- AMU's Test Acc: {:.2f} -----\n".format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(model.aux_adapter.state_dict(),
                       args.cache_dir + f"/best_adapter_" + str(args.shots) + "shots.pt")
    logger.info(f"----- Best Test Acc: {best_acc:.2f}, at epoch: {best_epoch}.-----\n")
    f.write(str(args.shots) + " shots\n")
    f.write('Best Test Acc: {:.4f}, at epoch: {}.\n'.format(best_acc, best_epoch))


if __name__ == '__main__':
    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    # args.dataset = 'ucf101'
    # args.root_path = '../Tip-Adapter/data'
    # args.shots = 2
    # args.clip_backbone = 'RN50'
    # args.rand_seed = 1
    # args.train_epoch = 20
    # args.moco_ep = 100

    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    logger = config_logging(args)
    logger.info("\nRunning configs.")
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    logger.info(message)

    # CLIP
    clip_model, preprocess = clip.load(args.clip_backbone)
    clip_model.eval()
    # AUX MODEL 
    aux_model, args.feat_dim = load_moco(
        "moco_checkpoints/r-50-{}ep.pth.tar".format(str(args.moco_ep)))  # Aux model path

    aux_model.cuda()
    aux_model.eval()

    # ImageNet dataset
    random.seed(args.rand_seed)
    torch.manual_seed(args.torch_rand_seed)

    # save results
    os.makedirs(args.exp_name, exist_ok=True)
    f = open(os.path.join(args.exp_name, '{}.txt'.format(args.dataset)), 'a')
    f.write(str(args) + '\n')

    if args.dataset == 'imagenet':
        logger.info("Loading ImageNet dataset....")
        imagenet = ImageNet(args.root_path, args.shots)
        test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=128, num_workers=8, shuffle=False)

        train_loader = torch.utils.data.DataLoader(imagenet.train, batch_size=args.batch_size, num_workers=8,
                                                   shuffle=True)
        train_loader_feature = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    else:
        dataset_wrapper = DatasetWrapper
        dataset = dataset_list[args.dataset](args.root_path, args.shots)
        test_loader = torch.utils.data.DataLoader(
            dataset_wrapper(dataset.test, input_size=224, transform=tfm_test_base, is_train=False), batch_size=128,
            num_workers=0, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            dataset_wrapper(dataset.train_x, input_size=224, transform=tfm_train_base, is_train=True),
            batch_size=args.batch_size,
            num_workers=0, shuffle=True)
        train_loader_feature = torch.utils.data.DataLoader(
            dataset_wrapper(dataset.train_x, input_size=224, transform=tfm_train_base, is_train=True), batch_size=256,
            num_workers=0, shuffle=False)
    # else:
    #     print("Preparing dataset.")
    #     dataset = build_dataset(args.dataset, args.root_path, args.shots)
    #     test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
    #                                     shuffle=False)
    #     train_tranform = transforms.Compose([
    #         transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    #     ])
    #     train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True,
    #                                      shuffle=True)
    #     train_loader_feature = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform,
    #                                              is_train=True, shuffle=False)

    # Textual features
    logger.info("Getting textual features as CLIP's classifier...")
    if args.dataset == 'imagenet':
        clip_weights = gpt_clip_classifier(imagenet.classnames, clip_model, imagenet.template)
    else:
        clip_weights = gpt_clip_classifier(dataset.classnames, clip_model, dataset.template)

    # Load visual features of few-shot training set
    logger.info("Load visual features of few-shot training set...")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader_feature, tfm_norm=tfm_aux)

    # Pre-load test features
    logger.info("Loading visual features and labels from test set.")

    logger.info("Loading CLIP test feature.")
    test_clip_features, test_labels = load_test_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip,
                                                         model_name='clip')

    logger.info(f"Loading AUX test feature.")
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux,
                                                        model_name='aux')

    test_clip_features = test_clip_features.cuda()
    test_aux_features = test_aux_features.cuda()

    # zero shot
    tmp = test_clip_features / test_clip_features.norm(dim=-1, keepdim=True)
    l = 100. * tmp @ clip_weights
    print(
        f"{l.argmax(dim=-1).eq(test_labels.cuda()).sum().item()}/ {len(test_labels)} = {l.argmax(dim=-1).eq(test_labels.cuda()).sum().item() / len(test_labels) * 100:.2f}%")

    # build amu-model
    model = AMU_Model(
        clip_model=clip_model,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=clip_weights,
        feat_dim=args.feat_dim,
        class_num=clip_weights.shape[1],
        lambda_merge=args.lambda_merge,
        alpha=args.alpha,
        uncent_type=args.uncent_type,
        uncent_power=args.uncent_power
    )

    train_and_eval(f, args, logger, model, test_clip_features, test_aux_features, test_labels, train_loader)

    f.close()
