# BertInterlinear
Using Bert Interlinear


```
sudo docker build -f Dockerfile -t bertinterlinear:latest . #This installs the models as well.

sudo docker run -v=$PWD:/data/ bertinterlinear:latest /bin/bash -c "python3 /data/fitbert.py"
sudo docker run -v=$PWD:/data/ bertinterlinear:latest /bin/bash -c "python3 /data/bert.py"

```


