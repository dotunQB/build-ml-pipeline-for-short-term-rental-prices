#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_columns = df.select_dtypes(include=['number']).columns
    # # fill 0 to all NaN
    df[numeric_columns] = df[numeric_columns].fillna(0)

    min_price = float(args.min_price)
    max_price = float(args.max_price)
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    filename = "cleaned_data.csv"
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # This waits for the artifact to be uploaded to W&B. If you
    # do not add this, the temp directory might be removed before
    # W&B had a chance to upload the datasets, and the upload
    # might fail
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name of the artifact to be used as the input",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="artifact output name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="artifact output type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="minimum price for dataframe",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="maximum price for dataframe",
        required=True
    )


    args = parser.parse_args()

    go(args)
