from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from models import FDDTN
from skorch.callbacks import EpochTimer, PrintLog, EpochScoring, BatchScoring, EarlyStopping
from skorch.utils import train_loss_score, valid_loss_score
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from collections import deque
from  torch.optim import Adam
import torch.nn as nn

import matplotlib.pyplot as plt
import arff
import numpy as np
import argparse
import os
import graphviz
import torch

def load_cancer():
    dset = load_breast_cancer(False)
    return dset['data'].astype(np.float32), dset['target'], \
           dset['feature_names'].tolist(), dset['target_names'].tolist()


def load_ttt():
    dset = fetch_openml('tic-tac-toe')
    return dset['data'].astype(np.int64), dset['target'], \
           dset['feature_names'], np.unique(dset['target']).tolist()


def load_cesarean():
    data = arff.load(open("./data/cesarean/caesarian.csv.arff"))
    xy = np.array(data['data']).astype(np.int64)
    tags = [a[0] for a in data['attributes']]
    x = xy[:, :-1]
    return x, xy[:, -1], tags[:-1], [tags[-1], 'Not '+tags[-1]]


def train_step_single_monkey_patch(self, Xi, yi, **fit_params):
    from skorch.utils import TeeGenerator
    self.module_.train()
    self.optimizer_.zero_grad()
    y_pred = self.infer(Xi, **fit_params)

    loss = self.get_loss(
        y_pred,
        torch.max(torch.from_numpy(self.encoder.transform(np.array(yi).reshape(-1, 1))), 1)[1],
        X=Xi,
        training=True)
    if len(self.history) > 100:
        last_ten_best = [ep['valid_loss_best'] for ep in self.history[-11:-1]]
        if not any(last_ten_best) and not self.regularization_triggered:
            self.regularization_triggered = True
            print('[INFO] Regularization Triggered.')
        if self.regularization_triggered:
            loss = loss + self.module_.regregularization() * self.reg_base
            self.reg_base = self.reg_base * self.reg_mult
    loss.backward()

    self.notify(
        'on_grad_computed',
        named_parameters=TeeGenerator(self.module_.named_parameters()),
        X=Xi,
        y=yi
    )

    return {
        'loss': loss,
        'y_pred': y_pred,
    }


def validation_step_monkey_patch(self, Xi, yi, **fit_params):
    self.module_.eval()
    with torch.no_grad():
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(
            y_pred,
            torch.max(torch.from_numpy(
                self.encoder.transform(np.array(yi).reshape(-1, 1))), 1)[1],
            X=Xi,
            training=False)
    return {
        'loss': loss,
        'y_pred': y_pred,
    }


def callbacks_monkey_patch(self):
    return [
        ('epoch_timer', EpochTimer()),
        ('train_loss', BatchScoring(
            train_loss_score,
            name='train_loss',
            on_train=True,
            target_extractor= \
                lambda x: self.encoder.transform(np.array(x).reshape(-1, 1)),
        )),
        ('valid_loss', BatchScoring(
            valid_loss_score,
            name='valid_loss',
            target_extractor= \
                lambda x: self.encoder.transform(np.array(x).reshape(-1, 1)),
        )),
        ('print_log', PrintLog()),
    ]

DATASETS = {'ttt': load_ttt, 'cesarean': load_cesarean, 'cancer': load_cancer}
MODELS = ['tree', 'ddt']


class SLPipeline:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.save_dir = os.path.join(self.save_dir, self.dataset)
        self.save_dir = os.path.join(self.save_dir, self.model)
        os.makedirs(self.save_dir, exist_ok=True)
        self.x, self.labels, self.feat_lbl, self.target_lbl = DATASETS[self.dataset]()
        self.enc = OneHotEncoder(categories='auto', sparse=False)
        self.enc = self.enc.fit(self.labels.reshape(-1, 1))
        plt.figure(1)
        if self.model == 'ddt':
            self.classifier = self._DDTClassifier()
        elif self.model == 'tree':
            self.classifier = DecisionTreeClassifier()

    def _DDTClassifier(self):
        if self.dataset == 'ttt':
            depth = 13
            enc = OneHotEncoder(categories='auto', sparse=False)
            enc = enc.fit(self.x)
            self.feat_lbl = enc.get_feature_names(self.feat_lbl).tolist()
            self.x = enc.transform(self.x).astype(np.float32)
        elif self.dataset == 'cancer':
            depth = 8
        elif self.dataset == 'cesarean':
            depth = 11
            enc = OneHotEncoder(categories='auto', sparse=False, )
            enc = enc.fit(self.x)
            self.feat_lbl = enc.get_feature_names(self.feat_lbl).tolist()
            self.x = enc.transform(self.x).astype(np.float32)
        else:
            raise ValueError('[ERROR] Invalid Dataset.')
        n_output = len(self.target_lbl)
        n_input = self.x.shape[1]
        module = lambda: FDDTN(depth=depth,
                       n_input=n_input,
                       n_output=n_output,
                       continuous=False,
                       labels=self.feat_lbl,
                       param_initer=lambda *x: 0.5 * torch.ones(*x),
                       action_labels=self.target_lbl)
        NeuralNetClassifier.train_step_single = train_step_single_monkey_patch
        NeuralNetClassifier.validation_step = validation_step_monkey_patch
        NeuralNetClassifier._default_callbacks = property(callbacks_monkey_patch)
        self.classifier = NeuralNetClassifier(
            module=module,
            criterion=nn.CrossEntropyLoss,
            optimizer=Adam,
            train_split=CVSplit(cv=0.3),
            # callbacks=[('EarlyStopping', EarlyStopping(patience=20,
            #                                            threshold=1e-6,
            #                                            threshold_mode='abs'))],
            lr=1e-2,
            max_epochs=600,
            batch_size=256,
            device='cuda')
        self.classifier.encoder = self.enc
        self._reset_classifier()
        return self.classifier

    def _reset_classifier(self):
        if self.model == 'ddt':
            self.classifier.regularization_triggered = False
            self.classifier.reg_base = 5e-2
            self.classifier.reg_mult = 1.005

    def _save_decision_tree(self):
        if self.model == 'tree':
            dot_data = export_graphviz(self.classifier, out_file=None,
                                       feature_names=self.feat_lbl,
                                       class_names=self.target_lbl,
                                       special_characters=True)
            graph = graphviz.Source(dot_data, format='svg')
            graph.render(os.path.join(self.save_dir, 'tree'))
        elif self.model == 'ddt':
            self.classifier.module_.tree_to_png(
                os.path.join(self.save_dir, 'tree.svg'))
            self.classifier.module_.hard_tree_to_png(
                os.path.join(self.save_dir, 'hard_tree.svg'))
        else:
            raise ValueError('[ERROR] Invalid Model Type.')

    def run(self):
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        # enc = OneHotEncoder(categories='auto', sparse=False)
        # enc = enc.fit(self.labels.reshape(-1, 1))
        y_real = []
        y_proba = []
        y_probas_hard = []
        precisions = []
        recalls = []
        precisions_hard = []
        recalls_hard = []
        positive_label = 'positive' if self.dataset == 'ttt' else None
        for i, (train, test) in enumerate(cv.split(self.x, self.labels)):
            Xtrain, Ytrain = self.x[train], self.labels[train]
            XTest, YTest = self.x[test], self.labels[test]
            if self.dataset == 'cancer' and self.model == 'ddt':
                enc = StandardScaler()
                enc = enc.fit(Xtrain)
                Xtrain = enc.transform(Xtrain).astype(np.float32)
                XTest = enc.transform(XTest).astype(np.float32)
            trained_classifier = self.classifier.fit(Xtrain, Ytrain)

            probas_ = trained_classifier.predict_proba(XTest)

            self._reset_classifier()
            precision, recall, _ = precision_recall_curve(YTest, probas_[:, 1],
                                                          positive_label)
            precisions.append(precision)
            recalls.append(recall)
            fig_lbl = 'Fold %d AUC=%.4f' % (i+1,
                                            roc_auc_score(YTest, probas_[:, 1]))
            plt.step(recall, precision, label=fig_lbl)
            y_real.append(YTest)
            y_proba.append(probas_)
            if self.model == 'ddt':
                trained_classifier.module_.make_hard()
                probas_hard = trained_classifier.predict_proba(XTest)
                trained_classifier.module_.make_soft()
                prec_hard, recall_hard, _ =\
                    precision_recall_curve(YTest, probas_hard[:, 1], positive_label)
                hard_fig_lbl = 'Fold %d HAUC=%.4f' %\
                (i+1, roc_auc_score(YTest, probas_hard[:, 1]))
                plt.step(recall_hard, prec_hard, label=hard_fig_lbl)
                precisions_hard.append(prec_hard)
                recalls_hard.append(recall_hard)
                y_probas_hard.append(probas_hard)
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba[:, 1],
                                                      positive_label)
        fig_lbl = 'Overall AUC=%.4f' % (
            roc_auc_score(y_real, y_proba[:, 1]))
        plt.step(recall, precision, label=fig_lbl, lw=2, color='black')
        if self.model == 'ddt':
            y_proba_hard = np.concatenate(y_probas_hard)
            precisions_hard, recall_hard, _ =\
                precision_recall_curve(y_real, y_proba_hard[:, 1], positive_label)
            hard_fig_lbl = 'Overall HAUC=%.4f' % (
                roc_auc_score(y_real, y_proba_hard[:, 1]))
            plt.step(recall_hard, precisions_hard, label=hard_fig_lbl,
                     lw=2, color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left', fontsize='small')
        plt.title('Precision Recall Curve')
        plt.savefig(os.path.join(self.save_dir, 'prc.svg'))
        self._save_decision_tree()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=list(DATASETS.keys()),
                        default='ttt')
    parser.add_argument('--save_dir', type=str, default='./plots')
    parser.add_argument('--model', choices=MODELS, default='ddt')
    args = parser.parse_args()
    pipeline = SLPipeline(**vars(args))
    pipeline.run()
