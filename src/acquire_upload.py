import logging.config
import os
from typing import Tuple
from zipfile import ZipFile
from io import BytesIO
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
import boto3
import botocore.exceptions

from s3path import S3Path


logger = logging.getLogger(__name__)


def s3_parse(s3_path: str) -> Tuple[str, str]:
    """
    Parse the given s3 path to pull the bucket name and the key

    Args:
        s3_path(`str`): full s3 path with both the key and the bucket name

    Return:
        s3_bucket(`str`): S3 bucket
        s3_path(`str`): s3_key for the bucket
    """
    path = S3Path.from_uri(s3_path)
    s3_bucket = path.bucket
    s3_path = f"{path.key}/"

    return s3_bucket, s3_path


def upload_files_to_s3(local_dir: str, s3_dir: str) -> None:
    """Uploads files from s3 directory to local directory

    Args:
        local_dir(`str`): local directory path
        s3_dir(`str`): s3 directory path like s3://bucket_name/data/

    Return:
        None
    """

    # pulling s3 key and s3 bucket from given s3 direction and inputting it to boto3
    s3_bucket, s3_key = s3_parse(s3_dir)
    bucket = boto3.resource("s3").Bucket(s3_bucket)

    # changing s3_key to have filename in the s3 directory
    filename = os.path.basename(local_dir)
    final_s3_key = f"{s3_key}{filename}"

    # looping through directory and uploading all the files to S3
    try:
        logger.info("Upload to AWS s3 is starting")
        for f_name in os.listdir(local_dir):
            if not f_name.startswith('.'):
                new_s3_key = f"{final_s3_key}{f_name}"
                local_file = f"{local_dir}{f_name}"
                bucket.upload_file(local_file, new_s3_key)
                logger.info("Upload of %s to AWS s3 is successful", f_name)

    except FileNotFoundError:
        logger.error("Please check whether the file path is correct and the data exists")

    except botocore.exceptions.NoCredentialsError:
        logger.warning("Please provide AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

    else:
        logger.info("Uploaded all the files in %s to s3 path", local_dir)


def download_from_s3_dir(local_dir: str, s3_dir: str) -> None:
    """ Downloads data from s3 to local directory
    Args:
        local_dir(`str'): local directory to download the file to
        s3_dir(`str`): directory for AWS s3 where the file lives

    Returns:
        None
    """

    # parsing s3 directory argument to get bucket name and s3 key
    s3_bucket, s3_key = s3_parse(s3_dir)
    bucket = boto3.resource("s3").Bucket(s3_bucket)

    # filtering directory names that include s3_key
    objs = list(bucket.objects.filter(Prefix=s3_key))

    try:
        for obj in objs:
            # extracting file name from the key of s3 path
            f_name = obj.key.split("/")[-1]

            if f_name != "":
                bucket.download_file(obj.key, f"{local_dir}{f_name}")
                logger.info("Downloaded %s from s3 directory", f_name)

    except botocore.exceptions.NoCredentialsError:
        logger.error("Data download failed. "
                     "Please provide AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

    else:
        logger.info("Finished downloading all the files from s3")


def download_from_web(local_dir: str, web_url: str) -> None:
    """ Downloads data from the given url
    Args:
        local_dir(`str`): local directory to download the file to
        web_url(`str`): url for the file

    Returns:
        None
    """
    try:
        with urlopen(web_url) as zip_resp:
            with ZipFile(BytesIO(zip_resp.read())) as z_file:
                z_file.extractall(local_dir)
    except HTTPError as error:
        logger.error("Error code: %s. Please check your url", error.code)

    except URLError as error:
        logger.error("Reason for the failed attempt to download: %s", error.reason)

    else:
        logger.info("Downloaded all the required files from %s to directory: %s",
                    web_url, local_dir)
