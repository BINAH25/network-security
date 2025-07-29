FROM python:alpine

WORKDIR /home/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /home/app
# Copy entrypoint script

EXPOSE 8000
CMD ["python3", "app.py"]