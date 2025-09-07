FROM ubuntu:24.04
LABEL maintainer="Bartosz Adamczyk <contact@bartekadamczyk.eu>"
LABEL description="Webserver for processing structures for RNAQuANet"

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    git \
    golang \
    openjdk-8-jre \
    jq \
    wget

RUN mkdir -p /rnaquanet
RUN mkdir -p /web
COPY ./rnaquanet/ /rnaquanet
COPY ./webserver/ /web
COPY ./requirements.txt /opt
WORKDIR /web
RUN CGO_ENABLED=0 GOOS=linux go build -o /server
RUN pip3 install -r /opt/requirements.txt --break-system-packages


CMD ["/server","-port=8080","-python=/rnaquanet/run.py","-maxiturl=172.17.0.1:4545"]