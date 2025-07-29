FROM python:3.14.0rc1-alpine

WORKDIR /app
COPY . /app

# Install AWS CLI and other dependencies
RUN apt-get update && apt-get install -y build-essential gcc

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python3", "app.py"]
