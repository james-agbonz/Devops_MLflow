FROM python:3.8-slim

RUN pip install scikit-learn shap matplotlib pandas numpy flask seaborn
RUN pip install mlflow

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]