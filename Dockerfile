FROM python:3

LABEL maintainer="imrilu"

RUN pip install --upgrade pip && pip install torch torchvision pytorch-lightning argparse

COPY NetPL.py MNISTDataModule.py ./

WORKDIR /workdir

ENTRYPOINT ["python", "/NetPL.py"]
