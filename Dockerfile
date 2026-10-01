FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRGYLINK_ENV=production \
    BRGYLINK_ENABLE_ADMIN=false \
    PORT=7860

WORKDIR /app

RUN groupadd --system appuser && useradd --system --gid appuser --home-dir /app appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 7860

# One worker deliberately keeps a short-lived opaque session on the same
# process. Mission17 must still send a stable, non-personal session ID.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "30", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
