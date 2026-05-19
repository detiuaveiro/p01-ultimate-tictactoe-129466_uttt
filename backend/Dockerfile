FROM python:3.14-alpine

WORKDIR /app

# Install the required websocket library
RUN pip install websockets

# Copy the server script and engine package into the container
COPY backend/server.py .
COPY engine/ ./engine/

# Expose the websocket port
EXPOSE 8765

# Run the server
CMD ["python", "server.py"]