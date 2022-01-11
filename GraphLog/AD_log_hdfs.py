

import BuildRulesFastParameterFree
import BuildRulesFastParameterFreeFreq
import BuildNetwork
import numpy as np
import os
import shutil
import time
import random
import argparse
from tqdm import tqdm
import math
import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


###########################################
# Functions
###########################################

def ReadSequentialData(InputFileName, read_ratio):
    print('Reading raw sequential data from {}'.format(InputFileName.split('/')[-1]))
    RawTrajectories = []
    if '.csv' in InputFileName:
        df_data = pd.read_csv(InputFileName, index_col=False)
        list_seq = df_data['EventSequence'].tolist()
        random.shuffle(list_seq)
        LoopCounter = 0
        for seq in list_seq:
            if LoopCounter < round(len(list_seq) * read_ratio):
                movements = seq.split(' ')
                movements = [key for key, grp in itertools.groupby(movements)]  # remove adjacent duplications
                movements.insert(0, "Start")  # start
                movements.append('End')
                LoopCounter += 1
                RawTrajectories.append([str(LoopCounter), movements])
            else:
                break
    elif '.txt' in InputFileName:
        with open(InputFileName) as f:
            LoopCounter = 0
            for line in f:
                if LoopCounter < round(len(f.readlines()) * read_ratio):
                    fields = line.strip().split(InputFileDeliminator)
                    ## In the context of global shipping, a ship sails among many ports
                    ## and generate trajectories.
                    ## Every line of record in the input file is in the format of:
                    ## [Ship1] [Port1] [Port2] [Port3]...
                    # ship = fields[0]
                    movements = fields
                    movements.insert(0, "Start") # start
                    movements.append('End')
                    LoopCounter += 1
                    if LoopCounter % 10000 == 0:
                        VPrint(LoopCounter)
                    ## Other preprocessing or metadata processing can be added here
                    ## Test for movement length
                    MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
                    if len(movements) < MinMovementLength:
                        continue
                    RawTrajectories.append([str(LoopCounter), movements])
                else:
                    break
    return RawTrajectories

def ReadTestData(name, training_ratio=0.0):
    hdfs = {}
    length = 0
    if '.csv' in name:
        df_data = pd.read_csv(name, index_col=False)
        list_sequence = df_data['EventSequence'].tolist()
        for ln in list_sequence[round(len(list_sequence) * training_ratio):]:
            if '[' in ln:
                ln = ln[ln.find("[") + 1:ln.find("]")].split(',')
                ln = [i.strip() for i in ln]
            else:
                ln = ln.split(' ')
            ln = [key for key, grp in itertools.groupby(ln)]
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
        print('Number of sequences ({}): {}'.format(name.split('/')[-1], len(hdfs)))
        return hdfs, length
    elif '.txt' in name:
        with open(name, 'r') as f:
            for ln in f.readlines()[round(len(f.readlines()) * training_ratio):]:
                ln = list(map(lambda n: n, map(int, ln.strip().split())))
                hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
                length += 1
        print('Number of sequences ({}): {}'.format(name.split('/')[-1], len(hdfs)))
        return hdfs, length


def DumpRules(Rules, OutputRulesFile):
    VPrint('Dumping rules to file')
    with open(OutputRulesFile, 'w') as f:
        for Source in Rules:
            for Target in Rules[Source]:
                f.write(' '.join([' '.join([str(x) for x in Source]), '=>', Target, str(Rules[Source][Target])]) + '\n')

def DumpNetwork(Network, OutputNetworkFile):
    VPrint('Dumping network to file')
    LineCount = 0
    with open(OutputNetworkFile, 'w') as f:
        for source in Network:
            for target in Network[source]:
                f.write(','.join([SequenceToNode(source), SequenceToNode(target), str(Network[source][target])]) + '\n')
                LineCount += 1
    VPrint(str(LineCount) + ' lines written.')

def SequenceToNode(seq):
    curr = seq[-1]
    node = curr + '|'
    seq = seq[:-1]
    while len(seq) > 0:
        curr = seq[-1]
        node = node + curr + '.'
        seq = seq[:-1]
    if node[-1] == '.':
        return node[:-1]
    else:
        return node

def VPrint(string):
    if Verbose:
        print(string)

def EventsProbability(Trajectory):
    count = {} # key: event, value: count
    for Tindex in range(len(Trajectory)):
        trajectory = Trajectory[Tindex][1]
        for event in trajectory:
            if event not in count:
                count[event] = 0
            count[event] += 1
    df_events = pd.DataFrame(count.items(), columns=['event', 'count'])
    total_counts = sum(count.values())
    df_events['probability'] = df_events.apply(lambda x: float(x['count'])/total_counts, axis=1)
    df_events.sort_values(by=['probability'], inplace=True, ascending=False)
    df_events.reset_index(drop=True, inplace=True)
    df_events.to_csv(path_network + 'events.csv', index=False)

def GenerateWholeGraph():
    print('generating network')
    OutputNetworkFile = path_network + 'network.csv'
    # print(OutputRulesFile, OutputNetworkFile)
    EventsProbability(RawTrajectories)
    Rules = BuildRulesFastParameterFree.ExtractRules(RawTrajectories, MaxOrder, MinSupport)
    # DumpRules(Rules, OutputRulesFile)
    print(len(Rules))
    Network = BuildNetwork.BuildNetwork(Rules)
    print(len(Network))
    DumpNetwork(Network, OutputNetworkFile)
    # ###
    # # generate network with freq as edge weight
    # ###
    # OutputNetworkFileFreq = path_network + 'network-freq.csv'
    # # print(OutputRulesFile, OutputNetworkFile)
    # Rules_Freq = BuildRulesFastParameterFreeFreq.ExtractRules(Training_data, MaxOrder, MinSupport)
    # # DumpRules(Rules, OutputRulesFile)
    # print(len(Rules_Freq))
    # Network_Freq = BuildNetwork.BuildNetwork(Rules_Freq)
    # print(len(Network_Freq))
    # DumpNetwork(Network_Freq, OutputNetworkFileFreq)

def FindLowNode(pre_s):
    if '.' in pre_s:
        pre_s = '.'.join(pre_s.split('.')[1:])
    else:
        pre_s = ''
    return pre_s


def SIM_N_v0(trajectory,NEdges, dict_events):
    Pt = 0.0
    for i in range(len(trajectory)-1):
        pre_s = '.'.join(trajectory[:i])
        s = trajectory[i]
        t = trajectory[i+1]
        s_hon = s + '|' + pre_s
        path_exist = False
        while not path_exist:
            if s_hon in NEdges.keys():
                targets = NEdges[s_hon]
                if t in targets:
                    pt = targets[t]
                    path_exist = True
                else:
                    if pre_s:
                        pre_s = FindLowNode(pre_s)
                        s_hon = s + '|' + pre_s
                    else:
                        pt = p0
                        path_exist = True
            else:
                if pre_s:
                    pre_s = FindLowNode(pre_s)
                    s_hon = s + '|' + pre_s
                else:
                    pt = p0
                    path_exist = True
        Pt += math.log(pt)
    sim = Pt / len(trajectory)
    return sim

def SIM_N_v1(trajectory,NEdges, dict_events):
    Pt = 0.0
    for i in range(len(trajectory)-1):
        pre_s = '.'.join(trajectory[:i])
        s = trajectory[i]
        t = trajectory[i+1]
        s_hon = s + '|' + pre_s
        path_exist = False
        while not path_exist:
            if s_hon in NEdges.keys():
                targets = NEdges[s_hon]
                if t in targets:
                    pt = targets[t]
                    path_exist = True
                else:
                    pt = p1
                    path_exist = True
            else:
                if pre_s:
                    pre_s = FindLowNode(pre_s)
                    s_hon = s + '|' + pre_s
                else:
                    if s in dict_events:
                        # pt = dict_events[s]
                        pt = p0
                    else:
                        pt = p0
                    path_exist = True
        Pt += math.log(pt)
    sim = Pt / len(trajectory)
    return sim

def Reverse(lst):
    return [ele for ele in reversed(lst)]

def SIM_N(trajectory,NEdges, dict_events, node_order):
    Pt = 0.0
    for i in range(len(trajectory)-1):
        if i < node_order:
            pre_s = '.'.join(Reverse(trajectory[:i]))
        else:
            pre_s = '.'.join(Reverse(trajectory[i-node_order+1:i]))
        s = trajectory[i]
        t = trajectory[i+1]
        s_hon = s + '|' + pre_s
        path_exist = False
        while not path_exist:
            if s_hon in NEdges.keys():
                targets = NEdges[s_hon]
                if t in targets:
                    pt = targets[t]
                    path_exist = True
                else:
                    pt = p1
                    path_exist = True
            else:
                if pre_s:
                    pre_s = FindLowNode(pre_s)
                    s_hon = s + '|' + pre_s
                else:
                    if s in dict_events:
                        pt = dict_events[s]
                        # pt = p0
                    else:
                        pt = p0
                    path_exist = True
        Pt += math.log(pt)
    sim = Pt / len(trajectory)
    return sim

def Cal_SIM(Test_data, NEdges, dict_events, node_order):
    list_sim = []
    Trajectory = list(Test_data.keys())
    counts = list(Test_data.values())
    for i in tqdm(range(len(Trajectory))):
        t = Trajectory[i]
        t = [str(j) for j in t]
        t.insert(0, "Start")  # start
        t.append('End')
        if len(t) > MinimumLengthForTesting:
            sim = SIM_N(t, NEdges, dict_events, node_order)
        else:
            sim = -50
        list_sim.append(sim)
    return list_sim, counts

def VisualSim(list_SIM_normal, list_SIM_abnormal, counts_normal, counts_abnormal, mean, threshold):
    sim_normal = []
    sim_abnormal = []
    for i, sim in enumerate(list_SIM_normal):
        sim_normal += counts_normal[i] * [sim]
    for i, sim in enumerate(list_SIM_abnormal):
        sim_abnormal += counts_abnormal[i] * [sim]
    list_SIM = sim_normal + sim_abnormal
    x1 = list(range(len(sim_normal)))
    x2 = list(range(len(sim_normal), len(list_SIM)))
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x1, sim_normal, color = 'blue', alpha=0.5, label='Normal data')
    plt.scatter(x2, sim_abnormal, color='red', alpha=0.5, label='Abnormal data')
    l2 = plt.plot(x1+x2, [mean]*len(list_SIM), 'y--', label='Mean')
    l1 = plt.plot(x1 + x2, [threshold]*len(list_SIM), 'r--', label='Threshold')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Similarity')
    # plt.show()
    plt.savefig(path_results + 'similarity_p0'+str(p0)+'_p1'+str(p1)+'.png', dpi=300)
    plt.show()
    plt.close()

def SaveSimlarity(list_SIM_normal, list_SIM_abnormal, counts_normal, counts_abnormal):
    df_normal_sim = pd.DataFrame({'similarity': list_SIM_normal,
                                  'counts': counts_normal,
                                  'sequence': [' '.join(key) for key in Test_normal.keys()]})
    df_abnormal_sim = pd.DataFrame({'similarity': list_SIM_abnormal,
                                  'counts': counts_abnormal,
                                  'sequence': [' '.join(key) for key in Test_abnormal.keys()]})
    df_normal_sim.to_csv(path_data+'normal_similarity.csv', index=False)
    df_abnormal_sim.to_csv(path_data + 'abnormal_similarity.csv', index=False)

def Predict():
    network_file = path_network + 'network.csv'
    NEdges = {}
    node_order = 0
    with open(network_file) as FG:
        for line in FG:
            fields = line.split(',')
            FromNode = fields[0]
            ToNode = fields[1].split('|')[0]
            weight = float(fields[2].strip())
            if FromNode not in NEdges:
                NEdges[FromNode] = {}
            NEdges[FromNode][ToNode] = weight
            from_order = len(FromNode.split('.')) + 1
            if from_order > node_order:
                node_order = from_order
    df_events = pd.read_csv(path_network+'events.csv', index_col=False)
    dict_events = dict(zip(df_events.event,df_events.probability))
    print('calculate SIM for normal test')
    list_SIM_normal, counts_normal = Cal_SIM(Test_normal, NEdges, dict_events, node_order)
    print('calculate SIM for abnormal test')
    time.sleep(0.02)
    list_SIM_abnormal, counts_abnormal = Cal_SIM(Test_abnormal, NEdges, dict_events, node_order)
    list_SIM = []
    for i, sim in enumerate(list_SIM_normal):
        list_SIM += counts_normal[i] * [sim]
    for i, sim in enumerate(list_SIM_abnormal):
        list_SIM += counts_abnormal[i] * [sim]

    arr_sim = np.array(list_SIM)
    mean = np.mean(arr_sim)
    std = np.std(arr_sim)
    threshold = mean - co_std * std

    VisualSim(list_SIM_normal, list_SIM_abnormal, counts_normal, counts_abnormal, mean, threshold)
    SaveSimlarity(list_SIM_normal, list_SIM_abnormal, counts_normal, counts_abnormal)

    pred_normal = [sim<threshold for sim in list_SIM_normal]
    pred_abnormal = [sim < threshold for sim in list_SIM_abnormal]

    TP = sum([counts_abnormal[i] for i,p in enumerate(pred_abnormal) if p==True])
    FN = sum([counts_abnormal[i] for i, p in enumerate(pred_abnormal) if p == False])
    FP = sum([counts_normal[i] for i, p in enumerate(pred_normal) if p == True])

    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print(
        'true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(TP, FP, FN, P, R, F1))

    performance = {'Values': [TP, FN, FP, P, R, F1]}
    df_p = pd.DataFrame(performance, index=['TP',
                                   'FN',
                                   'FP',
                                   'P',
                                   'R',
                                   'F1'])
    df_p.to_csv(path_results + 'performance.csv')

    parameters = {'Values': [p0, p1, co_std, training_ratio, MaxOrder, mean, std, threshold]}
    df_par = pd.DataFrame(parameters, index=['p0',
                                             'p1',
                                             'co_std',
                                             'training_ratio',
                                             'MaxOrder',
                                             'mean',
                                             'std',
                                             'threshold'])
    df_par.to_csv(path_results + 'parameters.csv')
    print('predict finished!!!')

def GenerateFPFN():
    df_abnormal = pd.read_csv(path_data + 'abnormal_similarity.csv', index_col=False)
    df_normal = pd.read_csv(path_data + 'normal_similarity.csv', index_col=False)
    df_parameters = pd.read_csv(path_results + 'parameters.csv', index_col=0)
    threshold = df_parameters.loc['threshold']['Values']
    df_tp = df_abnormal.loc[df_abnormal['similarity'] < threshold]
    df_fn = df_abnormal.loc[df_abnormal['similarity'] >= threshold]
    df_fp = df_normal.loc[df_normal['similarity'] < threshold]
    df_tn = df_normal.loc[df_normal['similarity'] >= threshold]
    df_tp.sort_values(by='counts', ascending=False, inplace=True)
    df_fp.sort_values(by='counts', ascending=False, inplace=True)
    df_fn.sort_values(by='counts', ascending=False, inplace=True)
    df_tn.sort_values(by='counts', ascending=False, inplace=True)
    df_tp.to_csv(path_data+'tp.csv', index=False)
    df_fp.to_csv(path_data+'fp.csv', index=False)
    df_fn.to_csv(path_data+'fn.csv', index=False)
    df_tn.to_csv(path_data+'tn.csv', index=False)
    print('generate tp fp fn finished!!!')

###########################################
# Main function
###########################################
## Initialize algorithm parameters
p1 = 1e-10
p0 = 1e-10
co_std = 1.0
training_ratio = 0.01
MaxOrder = 1
MinSupport = 50
LastStepsHoldOutForTesting = 0
MinimumLengthForTraining = 1
MinimumLengthForTesting = 5
InputFileDeliminator = ' '
Verbose = False

InputFolder = '../../data_preprocessed/Drain_HDFS_1/'
abnormal_file = InputFolder + 'data_anomaly.csv'
normal_file = InputFolder + 'data_normal.csv'
path_ADHD = '../../results/AD_log/hdfs/'
path_results = path_ADHD + 'hdfs_' + str(training_ratio) + '/'
path_data = path_results + 'data/'
path_network = path_results + 'network/'
os.makedirs(path_ADHD, exist_ok=True)
os.makedirs(path_results, exist_ok=True)
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_network, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict', 'analysis'])
    args = parser.parse_args()
    if args.mode == 'train':
        RawTrajectories = ReadSequentialData(normal_file, training_ratio)
        start_time = time.time()
        GenerateWholeGraph()
        print("--- Training process: %s seconds ---" % (time.time() - start_time))
    elif args.mode == 'predict':
        Test_normal, test_normal_length = ReadTestData(normal_file)
        Test_abnormal, test_abnormal_length = ReadTestData(abnormal_file)
        Predict()
        GenerateFPFN()
    else:
        GenerateFPFN()

    print('finished!!!')
