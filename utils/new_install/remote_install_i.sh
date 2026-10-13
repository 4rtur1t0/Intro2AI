#!/bin/bash

address=$1

# --- Configuración ---
SERVIDOR="172.16.28.$address"
USUARIO="usuario"
CONTRASENA=usuario

echo "INTENTANDO CONECTAR con IP" $SERVIDOR
# POR SIMPLICIDAD, SALIMOS SI EL SERVIDOR NO RESPONDE
nc -z -w 5 "$SERVIDOR" 22 || exit 1

echo "=== Conectando a $SERVIDOR mediante contraseña e instalando librerías ==="

# Usamos sshpass para enviar la contraseña de forma no interactiva
sshpass -p "$CONTRASENA" ssh -o StrictHostKeyChecking=no "$USUARIO@$SERVIDOR" "
    echo "Actualizando lista de paquetes..."
    echo "$CONTRASENA" | sudo -S apt update -y
    echo "$CONTRASENA" | sudo -S apt install -y git python3 python3-virtualenv
    echo "$CONTRASENA" | sudo -S apt install -y libavcodec-dev libavformat-dev libswscale-dev
    echo "$CONTRASENA" | sudo -S apt install -y net-tools
    echo "$CONTRASENA" | sudo -S apt install -y swig

    echo "CLONANDO INTRO2AI"
    cd ~/Escritorio/
    git clone https://github.com/4rtur1t0/Intro2AI

    echo "INSTALANDO VIRTUALENV Y LIBRERÍAS PARA INTRO2AI"
    # install virtualenv and python libraries
    cd ~/Escritorio/
    virtualenv venvAI
    cd ~/Escritorio/venvAI/bin
    ./pip install -r ../../Intro2AI/requirements.txt

    # TODO: INSTALAR CON CONDA... MINICONDA, ETC.
    echo "INSTALANDO CONDA Y LIBRERÍAS PARA INTRO2AI"
    # install virtualenv and python libraries
    cd ~/Escritorio/

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    cd ~/Escritorio/
    conda create -n iia python=3.11
    conda activate iia
    pip install numpy matplotlib
    pip install scikit-learn networkx osmnx contextily
    pip install "gymnasium[toy-text]"

    echo "Instalacion completada con exito"
"

# TRANSFERENCIA DE FICHEROS
# CUIDADO, el directorio debe existir
#echo "Transferring Coppelia Sim"
#sshpass -p usuario scp -r -O "$COPPELIA_PATH" usuario@172.16.28.$address:"$COPPELIA_PATH"

echo "=== Proceso finalizado ==="

