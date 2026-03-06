# MLOps Boilerplate : A streamlined environment for MLOps projects.

This README provides a concise guide to setting up and managing your MLOps environment, including service URLs and credentials in an easy-to-read table format.

## Architecture

```mermaid
graph TB
    subgraph "External Access"
        USER[User Browser]
    end

    subgraph "API"
        MongoDB[MongoDB:27025]
        FASTAPI[FASTAPI:8000]
        
    end

    subgraph "Airflow Orchestration"
        WEBSERVER[Airflow Webserver<br/>:8082]
        SCHEDULER[Airflow Scheduler]
        WORKER[Airflow Worker]
        TRIGGERER[Airflow Triggerer]
        FLOWER[Flower Monitor<br/>:5555]
    end

    subgraph "ML Development"
        JUPYTER[JupyterLab<br/>:8888]
        MLFLOW[MLflow Server<br/>:5000]
    end
    
    subgraph "Storage Services"
        MINIO[MinIO S3<br/>:9000/:9001]
        POSTGRES_AF[(PostgreSQL<br/>Airflow DB)]
        POSTGRES_ML[(PostgreSQL<br/>MLflow DB)]
        REDIS[(Redis<br/>Message Broker)]
    end
    
    subgraph "Shared Volumes"
        DAGS[/dags/]
        LOGS[/logs/]
        PLUGINS[/plugins/]
        DATA[/data/]
    end

    
    USER -->|:8082| WEBSERVER
    USER -->|:8888| JUPYTER
    USER -->|:5000| MLFLOW
    USER -->|:9001| MINIO
    USER -->|:5555| FLOWER
    USER -->|:8000| FASTAPI
    
    WEBSERVER --> POSTGRES_AF
    SCHEDULER --> POSTGRES_AF
    SCHEDULER --> REDIS
    WORKER --> REDIS
    WORKER --> POSTGRES_AF
    TRIGGERER --> POSTGRES_AF
    FLOWER --> REDIS
    MongoDB --> FASTAPI 
    
    MLFLOW --> POSTGRES_ML
    MLFLOW --> MINIO
    JUPYTER --> MLFLOW
    
    SCHEDULER -.-> DAGS
    WORKER -.-> DAGS
    WORKER -.-> PLUGINS
    WEBSERVER -.-> LOGS
    SCHEDULER -.-> LOGS
    JUPYTER -.-> DATA
    JUPYTER -.-> PLUGINS
    
    style USER fill:#e1f5ff
    style WEBSERVER fill:#ffecb3
    style SCHEDULER fill:#ffecb3
    style WORKER fill:#ffecb3
    style TRIGGERER fill:#ffecb3
    style FLOWER fill:#ffecb3
    style JUPYTER fill:#c8e6c9
    style MLFLOW fill:#c8e6c9
    style POSTGRES_AF fill:#f8bbd0
    style POSTGRES_ML fill:#f8bbd0
    style REDIS fill:#f8bbd0
    style MINIO fill:#d1c4e9
```

## Initialization
Run the following commands to set up everything:

```sh
make init-airflow
make start
```

## Running after a first run
Run

```sh
make start
```


## Services
Here is a list of the services provided, including their URLs and credentials:

| Services          | URL                       | Credentials           |
|------------------|---------------------------|-----------------------|
| Airflow          | http://localhost:8082     | airflow/airflow       |
| JupyterLab       | http://localhost:8888     | Token: cd4ml          |
| MLflow           | http://localhost:5000     | -                     |
| MinIO S3 server  | http://localhost:9001     | mlflow_access/mlflow_secret |
| Flower (Celery)  | http://localhost:5555     | -                     |
| FastAPI          | http://localhost:8000     | -                     |

## Cleanup
To stop all running Docker containers, press `Ctrl+C` and run:

```sh
make stop
```

To delete all running Docker containers and images:

```sh
make del-containers-and-images
```








