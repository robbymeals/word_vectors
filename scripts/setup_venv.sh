sudo apt-get -y install python3.4-dev libicu-dev build-essential libatlas-dev libatlas3gf-base gfortran libopenblas-dev liblapack-dev

python3 -m venv venv/ --without-pip;
source venv/bin/activate;
curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python;
deactivate;
source venv/bin/activate;
pip install --no-cache-dir -r requirements
