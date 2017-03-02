Notes from Gully about setting up and running this on HPC nodes for a beginner.

pip install --user https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl
git clone https://github.com/ybisk/DLHackathon
cd DLHackathon

#How does job.pbs actually start up a node on HPC   
qsub -l walltime=2:00.00 -l gpus=2:shared -I

# Set up the basic configuration
source /usr/usc/python/2.7.8/setup.sh
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/7.5-v5.1/setup.sh
 
python data.py
python train.py 

