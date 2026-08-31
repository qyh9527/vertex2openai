FROM python:3.11-slim

# 日志时区：装 tzdata 并设 TZ=Asia/Shanghai（Python time.tzset 依赖系统时区数据库，
# slim 镜像默认不含）；logger.py 的 time.tzset() 由此生效——文件日志/SSE 时间戳/
# 按天轮转与 stats.json 日期 key 全部走北京时间。LOG_TZ 环境变量可覆盖。
ENV TZ=Asia/Shanghai
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip cache purge && pip install --no-cache-dir -r requirements.txt

# Copy application code and local model fallback
COPY app/ .
COPY vertexModels.json .

# Expose the port
EXPOSE 7860

# Command to run the application
# Run the FastAPI service on the container port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
