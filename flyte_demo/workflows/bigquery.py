import os
import pandas as pd

from flytekit import task, workflow, kwtypes
from flytekit.testing import task_mock
from flytekitplugins.bigquery import BigQueryConfig, BigQueryTask


# specify these env vars with your own project id
PROJECT_ID = os.environ.get("BIGQUERY_PROJECT_ID", "flytebigquery")
LOCATION = os.environ.get("BIGQUERY_LOCATION", "US")


# get download data for the past day
QUERY = """
SELECT *
FROM `bigquery-public-data.pypi.file_downloads`
WHERE file.project = '@package'
AND DATE(timestamp)
    BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    AND CURRENT_DATE()
"""


bigquery_task = BigQueryTask(
    name="sql.bigquery.pip_downloads",
    inputs=kwtypes(package=str),
    output_structured_dataset_type=pd.DataFrame,
    query_template=QUERY,
    task_config=BigQueryConfig(
        ProjectID=PROJECT_ID,
        Location=LOCATION,
    ),
)


@task
def analyze_bigquery(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute downloads by country."""
    return (
        dataset.groupby("country_code")
        ["project"].count().rename("num_downloads").to_frame()
    )


@workflow
def wf(package: str) -> pd.DataFrame:
    return analyze_bigquery(dataset=bigquery_task(package=package))


if __name__ == "__main__":
    with task_mock(bigquery_task) as mock:
        mock.return_value = pd.read_gbq(
            QUERY.replace("@package", "flytekit"),
            project_id=PROJECT_ID,
            location=LOCATION,
        )
        output = wf(package="flytekit")
        print(output)
        assert (output["num_downloads"] > 0).all()
