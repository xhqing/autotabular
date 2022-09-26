# autotabular
Automated Machine Learning Framework in tabular domain.

## dev environment
using microsoft devcontainer.
vscode -> `cmd` + `shift` + `P` -> Remote-Container

```sh
docker pull xhq123/code-server:latest
docker commit code-server xhq123/code-server
dokcer push xhq123/code-server
docker run -d -u root -p 8088:8080 --name code-server -v $(pwd):/home/code xhq123/code-server
```