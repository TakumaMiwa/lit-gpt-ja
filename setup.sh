wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tar.xz
tar xJf Python-3.10.11.tar.xz
./Python-3.10.11/configure
make
sudo make install
sudo apt-get update
sudo apt-get install git-all
sudo apt install python3-pip
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
git clone https://github.com/TakumaMiwa/lit-gpt-ja.git
pip install -r lit-gpt-ja/requirements-all.txt