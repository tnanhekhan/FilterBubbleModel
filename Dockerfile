FROM python:3.9

ENV PYTHONUNBUFFERED=1

WORKDIR  /app

COPY ./ /app

EXPOSE 8521/tcp

RUN pip3 install -r requirements.txt

CMD ["python" , "fb_run.py"]
