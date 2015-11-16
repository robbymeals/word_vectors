sudo apt-get -y install python3.4-dev libicu-dev mysql-client libmysqlclient-dev
sudo apt-get -y install build-essential python3-dev libatlas-dev libatlas3gf-base
sudo apt-get -y install gfortran libopenblas-dev liblapack-dev libhdf5-dev

python3 -m venv venv/ --without-pip;
source venv/bin/activate;
curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python;
deactivate;
source venv/bin/activate;
pip install --no-cache-dir -r requirements
