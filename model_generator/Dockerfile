FROM ubuntu:24.04
RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt upgrade -y 
RUN apt install -y python3 python3-pip wget git gfortran libopenblas-dev  pkg-config libopenblas64-dev cmake libabsl-dev 

RUN mkdir -p /usr/simrna
WORKDIR /usr/simrna
RUN wget https://genesilico.pl/software/simrna/version_3.20/SimRNA_64bitIntel_Linux.tgz
RUN tar xf SimRNA_64bitIntel_Linux.tgz
RUN export PATH=$PATH:/usr/simrna/SimRNA_64bitIntel_Linux/SimRNA

RUN pip install boltz -U --break-system-packages
