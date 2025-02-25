# Use Python slim version for a smaller image size
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app_folder

# Copy dependencies first and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py ./

# Set environment variable to prevent buffering
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
