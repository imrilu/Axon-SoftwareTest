FROM python:3

RUN pip install --upgrade pip && pip install torch torchvision pytorch-lightning argparse

COPY NetPL.py MNISTDataModule.py ./

WORKDIR /workdir

ENTRYPOINT ["python", "/NetPL.py"]
