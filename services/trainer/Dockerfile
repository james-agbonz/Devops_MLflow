FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install gunicorn

RUN pip install scikit-learn shap matplotlib pandas numpy flask seaborn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]