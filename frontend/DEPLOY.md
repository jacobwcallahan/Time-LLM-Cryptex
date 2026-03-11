# Deploying the Gradio Frontend

## Quick options

### 1. Local network (already works)
With `server_name="0.0.0.0"`, anyone on your LAN can access:
```
http://<your-machine-ip>:9001
```

### 2. Temporary public URL (Gradio tunnel)
```bash
GRADIO_SHARE=true python app.py
```
Gradio will print a public URL (e.g. `https://xxx.gradio.live`) that works for ~72 hours.

### 3. Production behind reverse proxy (nginx, etc.)

Set environment variables before running:

```bash
# Required: MLflow must be reachable from the host running the app
export MLFLOW_TRACKING_URI="http://your-mlflow-host:5000"
export MLFLOW_S3_ENDPOINT_URL="http://your-minio-host:9000"

# When served at https://yoursite.com/app (subpath)
export GRADIO_ROOT_PATH="/app"

# Optional
export GRADIO_SERVER_PORT=9001
export GRADIO_DEBUG=false

cd frontend && python app.py
```

**Nginx example** (serving at `/app`):
```nginx
location /app {
    proxy_pass http://127.0.0.1:9001;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### 4. Docker (optional)
If containerizing, ensure MLflow/MinIO hostnames are reachable and set the env vars. The built-in Gradio server (uvicorn) is suitable for production when behind a reverse proxy.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://192.168.1.103:5000` | MLflow server URL |
| `MLFLOW_S3_ENDPOINT_URL` | `http://192.168.1.103:9000` | MinIO/S3 endpoint for artifacts |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Bind address (0.0.0.0 = all interfaces) |
| `GRADIO_SERVER_PORT` | `9001` | Port to listen on |
| `GRADIO_SHARE` | `false` | Set `true` for temporary public URL |
| `GRADIO_ROOT_PATH` | (empty) | Subpath when behind reverse proxy (e.g. `/app`) |
| `GRADIO_DEBUG` | `true` | Enable debug mode |

## Checklist for hosted deployment

1. **MLflow reachable** – The host running the Gradio app must reach MLflow (and MinIO if using S3 artifacts). Use `MLFLOW_TRACKING_URI` and `MLFLOW_S3_ENDPOINT_URL` with the correct hostnames (e.g. `http://mlflow:5000` in Docker, or your server's public IP).

2. **Data paths** – The app uses `Path(__file__).parent.parent` for the project root. Ensure the project directory structure is preserved when deploying.

3. **GPU** – If running inference, the host needs GPU access. The model loads via `run_inference.py`.

4. **Port conflict** – Docker Compose uses MinIO on 9001. Use a different port for Gradio if both run on the same host:
   ```bash
   GRADIO_SERVER_PORT=9002 python app.py
   ```
