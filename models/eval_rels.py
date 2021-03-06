
from dataloaders.georgia_tech import GTDataLoader, GT
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval_rt import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE_GT, IM_SCALE_GT
import dill as pkl
import os

conf = ModelConfig()
if conf.model == 'rtnet':
    from lib.rt_rel_model import RTRelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RTRelModel
else:
    raise ValueError()

train, val, test = GT.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test
train_loader, val_loader = GTDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RTRelModel(classes=train.ind_to_classes, aff_classes=train.ind_to_aff_classes,
                      att_classes=train.ind_to_att_classes, rel_classes=train.ind_to_predicates,
                      num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                      use_resnet=conf.use_resnet, order=conf.order,
                      nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                      use_proposals=conf.use_proposals,
                      pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                      pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                      pooling_dim=conf.pooling_dim,
                      rec_dropout=conf.rec_dropout,
                      use_bias=conf.use_bias,
                      use_tanh=conf.use_tanh,
                      limit_vision=conf.limit_vision
                      )

detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])

all_pred_entries = []
def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, obj_aff_i, obj_aff_scores_i, obj_att_i, obj_att_scores_i,
            rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_aff_classes': val.gt_aff_classes[batch_num + i][:, 1:].copy(),
            'gt_att_classes': val.gt_att_classes[batch_num + i][:, 1:].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE_GT/IM_SCALE_GT,
            'pred_classes': objs_i,
            'pred_aff_classes': obj_aff_i,
            'pred_att_classes': obj_att_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'obj_aff_scores': obj_aff_scores_i,
            'obj_att_scores': obj_att_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }
        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    evaluator[conf.mode].print_stats()
else:
    detector.eval()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus*val_b, batch, evaluator)

    evaluator[conf.mode].print_stats()

    print("Total relations in the testing dataset: {}".format(evaluator[conf.mode].total_rel))

    for gt_key in evaluator[conf.mode].gt_rel:
        print("GT: {} has {} samples in the testing set".format(gt_key, evaluator[conf.mode].gt_rel[gt_key]))

    print("Relation prediction failure case statistic:\n")
    for gt_key in evaluator[conf.mode].rel_failure_case:
        print('GT: {} has {} failure cases'.format(gt_key, evaluator[conf.mode].rel_failure_case[gt_key]))

    print("Analysis of object category failure case\n")
    for gt_key in list(sorted(evaluator[conf.mode].gt_cat)):
        print("Object index: {} has {} samples in the testing set".format(gt_key, evaluator[conf.mode].gt_cat[gt_key]))
        print("Object index: {} has {} failure cases".format(gt_key, evaluator[conf.mode].category_failure_case[gt_key]))
        print("Object index: {} has {} failure rate".format(gt_key,
                                                            evaluator[conf.mode].category_failure_case[gt_key]/evaluator[conf.mode].gt_cat[gt_key]*100.))

    if conf.cache is not None:
        with open(conf.cache,'wb') as f:
            pkl.dump(all_pred_entries, f)
