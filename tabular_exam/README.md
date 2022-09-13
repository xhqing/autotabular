# AutoDL Source Code

## tabular example
```bash
python example.py
```

## development
```sh
conda create --name python36 python=3.6.13
conda activate python36
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
python example.py
```

## deployment
```sh
pwd # xxx/xxx/xxx/autodl_tabular_example
docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 --name=tabular evariste/autodl:cpu-latest
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
python example.py
```

