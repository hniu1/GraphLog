# GraphLog: Execution Anomaly Detection for System Logs
A graph-based execution anoamly detection method based on variable-order network representation. The description instructs GraphLog and baseline methods and helps to reproduce the evaluation results. 3 parts were included:
* [Log sequence datasets](https://github.com/hniu1/GraphLog/tree/main/data_preprocessed)
* [GraphLog](https://github.com/hniu1/GraphLog/tree/main/GraphLog)
* [Baseline methods](https://github.com/hniu1/GraphLog/tree/main/baselines)

## Dataset
The preprocessed datasets are provided for evaluation. We used two dataset here:
* [OpenStackLog](https://github.com/hniu1/GraphLog/tree/main/data_preprocessed/OpenStackLog): We collected it from OpenStack that was deployed on CouldLab, which is a testbed for research and education in cloud computing. There are 174,725 logs collected. After preprocessing, it contains 6,000 sessions as normal, 500 abnormal sessions and 36 event templates. The detail of the dataset can be found in the paper.
* [HDFS](https://github.com/hniu1/GraphLog/tree/main/data_preprocessed/HDFS): The HDFS dataset was collected running Hadoop-based jobs from more than 200 Amazon’s EC2 nodes, and labeled by Hadoop domain experts. There are 11,175,629 logs in the dataset and it parsed 558,223 normal sequences and 16,838 abnormal sequences (2.9%). The detail of the dataset can be found [here](https://github.com/logpai/loghub/tree/master/HDFS).

## GraphLog
This section shows the steps how to run GraphLog.
```
cd GraphLog/
# Training
python AD_log_openstacklog.py train
# Testing
python AD_log_openstacklog.py predict
```
'training_ratio' can be set to different value in the code for different percentage of normal data as training data.

<!-- For HDFS dataset:
```
cd GraphLog/
# Training
python AD_log_hdfs.py train
# Testing
python AD_log_hdfs.py predict
```
When training ratio is 1%, the parameters we used are listed [here](https://github.com/hniu1/GraphLog/blob/main/results/AD_log/hdfs/hdfs_0.01/parameters.csv);
when training ratio is 5%, the parameters we used are listed [here](https://github.com/hniu1/GraphLog/blob/main/results/AD_log/hdfs/hdfs_0.05/parameters.csv). -->

## Baseline methods
4 baseline methods are used:
* PCA: [Large-Scale System Problems Detection by Mining Console Logs](http://iiis.tsinghua.edu.cn/~weixu/files/sosp09.pdf)
* InvariantsMiner: [Mining Invariants from Console Logs for System Problem Detection](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Lou.pdf)
* LogCluster: [Log Clustering based Problem Identification for Online Service Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/ICSE-2016-2-Log-Clustering-based-Problem-Identification-for-Online-Service-Systems.pdf)
* DeepLog: [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)

we use open-source machine learning-based log analysis toolkit for baseline methods, [Loglizer](https://github.com/logpai/loglizer) and [logdeep](https://github.com/donglee-afar/logdeep).

### PCA
In [PCA_OpenStackLog.py](https://github.com/hniu1/GraphLog/blob/main/baselines/PCA_OpenStackLog.py), first set training_ratio. Then,
```
cd baselines/
python PCA_OpenStackLog.py
```

### InvariantsMiner
In [InvariantsMiner_OpenStackLog.py](https://github.com/hniu1/GraphLog/blob/main/baselines/InvariantsMiner_OpenStackLog.py), first set training_ratio. Then,
```
cd baselines/
python InvariantsMiner_OpenStackLog.py
```

### LogCluster
In [LogClustering_OpenStackLog.py](https://github.com/hniu1/GraphLog/blob/main/baselines/LogClustering_OpenStackLog.py), first set training_ratio. Then,
```
cd baselines/
python LogClustering_OpenStackLog.py
```

### DeepLog
In [LogClustering_OpenStackLog.py](https://github.com/hniu1/GraphLog/blob/main/baselines/LogClustering_OpenStackLog.py), first set training_ratio. Then,
```
cd baselines/deeplog/
# training
python deeplog_OpenStackLog.py train
# testing
python deeplog_OpenStackLog.py predict
```
