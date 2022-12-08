from flytekit import task, workflow, kwtypes
from flytekitplugins.bigquery import BigQueryConfig, BigQueryTask

from flytekit.types.structured.structured_dataset import StructuredDataset


bigquery_task = BigQueryTask(
    name="sql.bigquery.pip_downloads",
    inputs=kwtypes(package=str),
    output_structured_dataset_type=StructuredDataset,
    query_template="""
    SELECT COUNT(*) AS num_downloads
    FROM `bigquery-public-data.pypi.file_downloads`
    WHERE file.project = '@package'
    -- Only query the last 30 days of history
    AND DATE(timestamp)
        BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
        AND CURRENT_DATE()
    """,
    task_config=BigQueryConfig(
        ProjectID="bigquery-public-data",
        Location="US",
    ),
)

@task
def analyze_bigquery(dataset: StructuredDataset) -> int:
    return int(dataset.shape[0])


@workflow
def wf(package: str) -> int:
    return analyze_bigquery(dataset=bigquery_task(package=package))


if __name__ == "__main__":
    wf(package="flytekit")
