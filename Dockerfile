
FROM ubuntu
RUN apt-get update && apt upgrade
RUN apt-get -y install make
RUN apt-get update  && apt-get -y install python3 python3-pip

COPY dist/twconvrecsys-0.1-py3-none-any.whl /home
WORKDIR /home
RUN pip3 install twconvrecsys-0.1-py3-none-any.whl -t .
