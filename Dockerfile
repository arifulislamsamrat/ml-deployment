FROM python:3.9-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app/ .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:create_app()"]