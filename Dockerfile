FROM quay.io/jupyter/pyspark-notebook:aarch64-spark-3.5.0

USER root

RUN apt-get update && apt-get install -y wget \
    && wget -P /usr/local/spark/jars https://jdbc.postgresql.org/download/postgresql-42.2.20.jar

USER $NB_UID

RUN mkdir -p /tmp/spark-events && chmod -R go+rwX /tmp/spark-events

CMD ["start-notebook.sh"]
