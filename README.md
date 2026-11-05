# Olist ML-пайплайн

Airflow читает заказы из PostgreSQL, проверяет данные, делит их по дате и обучает модель опоздания доставки. MLflow хранит модель и метрики.

## Запуск

```bash
read -r -p 'Адрес репозитория: ' REPO_URL
git clone "$REPO_URL" olist-homework
cd olist-homework
cp .env.example .env
docker compose up -d --build --wait
```

Airflow доступен на http://127.0.0.1:8080, MLflow на http://127.0.0.1:5050.

В Airflow снимите DAG olist_training с паузы, нажмите Trigger и дождитесь успеха всех четырёх задач. Возьмите run_id из эксперимента olist_training в MLflow. Команда ниже загружает модель и выдаёт пять прогнозов. В новом окружении подставьте run_id своего запуска.

```bash
docker compose exec -T airflow python -m check_model --run-id a4ce5b35acaf4e52b6955f2c6d5bb1ac
```

## Что проверено

Два ручных запуска создали разные run_id с одинаковым SHA256 снимка и размерами выборок. Дубликат order_id остановил DAG на validate. После намеренной ошибки и Clear повторилась только train, а остальные задачи сохранили первую попытку. После пересоздания MLflow все пять прогнозов совпали.

[Скрин Airflow](screenshots/olist-airflow-success.jpg) · [Скрин MLflow](screenshots/olist-mlflow-run.jpg)
