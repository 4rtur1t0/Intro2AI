#
# INSTALL Intro2AI and the required packages
#

# Change INSTALLDIR to your needs. It currently points to your desktop
# the library's directory Intro2AI should be placed at the same directory.
INSTALLDIR=~/Escritorio
mkdir $INSTALLDIR

echo 'INSTALLING Intro2AI in'
echo $INSTALLDIR
echo 'INSTALLING virtualenv, python3-dev and git'
read -p "Do you want to proceed? (y/n) " choice
case $choice in
[yY]* ) echo "Ok, installing" ;;
[nN]* ) exit ;;
*) exit ;;
esac

# this installs the virtualenv package and python and git
sudo apt install virtualenv
sudo apt install python3-dev
sudo apt install git

cd ~
cd $INSTALLDIR

echo 'CLONING repository Intro2AI from github. Say no if you already cloned it'
read -p "Do you want to proceed? (y/n) " choice
case $choice in
[yY]* ) echo "OK, installing" ;;
[nN]* ) exit ;;
*) exit ;;
esac

git clone https://github.com/4rtur1t0/Intro2AI

echo 'NOW creating a virtual environment and installing requirements with pip in it.'
echo $INSTALLDIR
read -p "Do you want to proceed? (y/n) " choice
case $choice in
[yY]* ) echo "Ok! Creating virtual environment" ;;
[nN]* ) exit ;;
*) exit ;;
esac

# move to the home directory
cd ~
cd $INSTALLDIR
echo 'CREATING VIRTUAL ENVIRONMENT IN'
echo $INSTALLDIR
virtualenv venv
cd venv/bin
pwd
echo 'INSTALLING DEPENDENCIES WITH PIP'
./pip install -r ../../Intro2AI/requirements.txt


