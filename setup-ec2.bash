#!/usr/bin/env bash
# ubuntu 16.04
# when spinning up the server, you have to expand the storage from the default 8gb to 16gb (for CUDA)
# this'll result in $0.10 * 8 = $0.80 / month extra for the extra storage
# note that the EBS charge is incurred for stopped instances
sudo apt update
sudo apt upgrade

# prep nvidia env
sudo apt-get install gcc linux-headers-$(uname -r)

# intalling CUDA with package manager
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda

# update path with CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# # installing CUDA from .run file. THIS SHIT DOESN'T WORK
# # grab the correct URL from: https://developer.nvidia.com/cuda-downloads
# wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
# sudo sh cuda_8.0.61_375.26_linux-run  # accept EULA by pressing (q), then type "accept". 
#                                       # don't install samples, no need for display driver

# install cuDNN
echo "Download and copy cuDNN from your local machine to the root directory."  
    # be patient when scp'ing... incomplete transfer will fail when untarring
read -p "Press any key to continue..."

sudo tar -xvf cudnn-8.0-linux-x64-v6.0.tgz -C /usr/local.  # make sure this makes it into: /usr/local/cuda-8.0/!
export CUDA_HOME=/usr/local/cuda

sudo apt install libcupti-dev

# setup python environment
sudo apt install python3 python3-pip python-dev speedtest-cli python3-tk
pip3 install --upgrade numpy keras tensorflow-gpu matplotlib scikit-learn
export TF_CPP_MIN_LOG_LEVEL=2  # reduce warnings

# run keras once and change backend to tensorflow
python -c "import keras"
