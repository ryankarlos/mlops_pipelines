FROM python:3.9-slim-buster

# Install Python dependencies
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt


COPY flyte_examples /root/flyte_examples

# when registering tasks, workflows, and launch plans
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag