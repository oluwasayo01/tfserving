FROM gcr.io/kubeflow-images-public/tensorflow-2.1.0-notebook-cpu:1.0.0
USER root
RUN pip install --upgrade --user tensorflow
WORKDIR /train
COPY . .
CMD ["python", "train.py"]