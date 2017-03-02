Notes from Gully about setting up and running TensorFLow on HPC nodes for a beginner.

```
pip install --user https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl
git clone https://github.com/ybisk/DLHackathon
cd DLHackathon
```

How does job.pbs actually start up a node on HPC   
```
qsub -l walltime=2:00.00 -l gpus=2:shared -I
```

This gives us the following output:

```
----------------------------------------
Begin PBS Prologue Wed Mar  1 16:27:59 PST 2017
Job ID:            22931185.hpc-pbs1.hpcc.usc.edu
Username:          gully
Group:             isi-ar
Project:           default
Name:              STDIN
Queue:             quick
Shared Access:     no
All Cores:         no
Has MIC:           no
Nodes:             hpc3066 
TMPDIR:            /tmp/22931185.hpc-pbs1.hpcc.usc.edu
End PBS Prologue Wed Mar  1 16:27:59 PST 2017
----------------------------------------
```
To log into this node: types

```
ssh hpc3066
```

Set up the basic configuration

```
source /usr/usc/python/2.7.8/setup.sh
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/7.5-v5.1/setup.sh
``` 
Other scripts provides data and basic templates. 

python data.py
python train.py 
python Template.py

'''
