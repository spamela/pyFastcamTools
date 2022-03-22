module purge
module load profile/base      
module load profile/deeplrn
module load cuda/10.1
module load gnu/8.4.0
module load python/3.8.2
module load cudnn/7.6.5--cuda--10.1
module load nccl/2.7.8--cuda--10.1
module load openblas/0.3.9--gnu--8.4.0
module load szip/2.1.1--gnu--8.4.0
module load zlib/1.2.11--gnu--8.4.0
module load hdf5/1.10.6--gnu--8.4.0       
module load tensorflow/2.3.0--cuda--10.1  

# or...
module load profile/base profile/deeplrn
module load cuda/10.1 gnu/8.4.0 python/3.8.2 cudnn/7.6.5--cuda--10.1 nccl/2.7.8--cuda--10.1 openblas/0.3.9--gnu--8.4.0 szip/2.1.1--gnu--8.4.0 zlib/1.2.11--gnu--8.4.0 hdf5/1.10.6--gnu--8.4.0 tensorflow/2.3.0--cuda--10.1


python3 -m venv venv
. ./venv/bin/activate

pip3 install --upgrade pip
#pip3 install cython
pip3 install matplotlib
#pip3 install tensorflow
#pip3 install --upgrade tensorflow
pip3 install opencv-python
pip3 install Pillow
pip3 install tqdm

# in python execution of tensorflow, need to add 
>> tf.enable_eager_execution()
# after importing tensorflow

