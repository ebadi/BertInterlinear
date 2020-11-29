FROM ubuntu:20.04

MAINTAINER @ebadi

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y python3-pip 
RUN pip3 install fitbert
RUN python3 -c "from fitbert import FitBert; fb = FitBert();"
RUN python3 -c "from transformers import BertTokenizer, BertForMaskedLM; BertTokenizer.from_pretrained('bert-base-uncased'); BertForMaskedLM.from_pretrained('bert-base-uncased');"
