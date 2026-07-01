Install

Vamos a instalar un entorno virtual en python. Trabajaremos en un directorio propio e instalaremos 
ahí todos los paquetes necesarios
sudo apt-get update

sudo apt install python3-virtualenv

Otros paquetes necesarios
sudo apt install swig build-essential python3-dev
sudo apt install python3-tk
sudo apt install swig


Creamos un entorno virtual de python
cd /home/usuario/Applications/
virtualenv venvAI

Entramos en nuestra carpeta
cd venvAI/bin
Activamos el entorno virtual, al llamar a python o pip estaremos ejecutando específicamente
este entorno.
~/Applications/venvAI/bin$ source ./activate
# ejecutamos pip usando el fichero de requirements
./pip install -r /home/usuario/Desktop/IntroAI/requirements.txt
./pip install numpy
./pip install gymnasium[all]
./pip install stable-baselines3 swig torch


Alternativamente, (o si falla lo anterior)
instala los paquetes


