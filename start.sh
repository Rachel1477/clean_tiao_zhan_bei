#!/bin/bash
echo "Starting Flask API server..."
python app.py &

echo "Starting Gradio Training UI..."
python gradio_src/gradio_interface.py &

echo "Both processes started in the background."