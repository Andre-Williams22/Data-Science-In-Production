FROM python:3.7-slim-buster

# Create app directory
RUN mkdir app
WORKDIR app

# Copy current directory into /app directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Define command
CMD ["python", "/app/app.py"]
