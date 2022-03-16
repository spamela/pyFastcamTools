module purge
module load python/3.9.4

mkdir venv
python3 -m venv venv
. ./venv/bin/activate

pip3 install opencv-python
pip3 install Pillow
pip3 install pandas
pip3 install matplotlib
pip3 install xarray
pip3 install h5py

cd ../pyIpx
pip3 install -e .

