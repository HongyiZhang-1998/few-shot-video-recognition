# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, help='GPU device', default=1)
    # parser.add_argument('-ds', '--device_ids', type=int, help='GPU device', default=0)
    parser.add_argument('-set', '--dataset', type=str, help='dataset', default='ucf101')
    parser.add_argument('--arch', type=str, default="3dresnet18")
    parser.add_argument('--consensus_type', type=str, default='avg')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
    parser.add_argument('--dropout', '--do', default=0, type=float, metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('-num_segments', '--num_segments', type=int, help='length of num_segments', default=16)
    parser.add_argument('-modality', '--modality', type=str,  help='is or not leave all frame', default="RGB")
    parser.add_argument('-mode', '--mode', type=str, help='mode', default='train')
    parser.add_argument('-model', '--model',  type=int,help='use or not best model', default=0)
    parser.add_argument('-dense_sample', '--dense_sample', type=bool, default=False)
    parser.add_argument('-workers', '--workers', type=int, default=4)
    parser.add_argument('-output_features', '--output_features', type=int, default=256)

    parser.add_argument('-start_epoch', '--start_epoch',
                        type=int,
                        help='use or not best model',
                        default=0)
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=500)
    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='test')


    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)
    parser.add_argument('-lrf', '--lr_flag',
                        type=str,
                        help='lr_scheduler type',
                        default='reduceLR')
    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)
    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-dtw', '--dtw',
                        type=str,
                        help='if using dtw',
                        default='sdtw')
    parser.add_argument('-dtw_threshold', '--dtw_threshold',
                        type=int,
                        help='if margin dtw',
                        default=0)
    parser.add_argument('-gama', '--gamma',
                        type=float,
                        help='reg',
                        default=0.01)

    parser.add_argument('-use_bias', '--use_bias',
                        type=int,
                        help='reg',
                        default=0)
    parser.add_argument('-vat', '--vat',
                        type=float,
                        help='vat',
                        default=1)
    parser.add_argument('-alpha', '--alpha',
                        type=float,
                        help='reg',
                        default=10)
    parser.add_argument('-eps', '--eps',
                        type=float,
                        help='reg',
                        default=1)
    parser.add_argument('-xi', '--xi',
                        type=float,
                        help='reg',
                        default=20)
    parser.add_argument('-ip', '--ip',
                        type=int,
                        help='times',
                        default=1)

    parser.add_argument('-its', '--train_iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=500)
    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=5)
    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=1)
    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)
    parser.add_argument('-test_its', '--test_iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=200)
    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)
    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=1)
    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=5)

    parser.add_argument('-dbg', '--debug',
                        type=int,
                        help='debug to save x and sim_tenor',
                        default=0)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
