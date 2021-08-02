FROM ubuntu:20.04

MAINTAINER @ebadi

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y python3-pip
RUN pip3 install torch>=1.1.0
RUN pip3 install transformers>=2.1.1
RUN pip3 install PyFunctional==1.2.0
RUN python3 -c "from transformers import BertTokenizer, BertForMaskedLM; BertTokenizer.from_pretrained('bert-base-uncased'); BertForMaskedLM.from_pretrained('bert-base-uncased');"

ADD fitbert /fitbert
RUN cd /fitbert;  python3 setup.py install
# To download models:
RUN python3 -c "from fitbert import FitBert; fb = FitBert();"