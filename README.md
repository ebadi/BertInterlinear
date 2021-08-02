# BertInterlinear
Using Bert Interlinear

git clone https://github.com/ebadi/fitbert.git

```
docker build -f Dockerfile -t bertinterlinear:latest . #This installs the models as well.
docker run -v=$PWD:/data/ bertinterlinear:latest /bin/bash -c "python3 /data/activebert.py"
docker run -v=$PWD:/data/ bertinterlinear:latest /bin/bash -c "python3 /data/fitbertx.py"
docker run -v=$PWD:/data/ bertinterlinear:latest /bin/bash -c "python3 /data/betterfitbert.py"
```


https://github.com/renatoviolin/next_word_prediction/blob/master/app.py

https://github.com/gdario/learning_transformers/blob/master/src/bert_example.py




