from dagster import job, op, execute_job
from app.scripts.flows import collect_data, transfrom_data

@op
def collect_data_op():
    collect_data()

@op
def transform_data_op():
    transfrom_data()

@job
def chembl_job():
    collect_data_op()
    transform_data_op()

if __name__ == "__main__":
    execute_job(chembl_job)