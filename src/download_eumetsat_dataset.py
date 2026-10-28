
import os
import sys
from pathlib import Path

import eumdac

PRODUCT_ID = (
    "MSG3-SEVI-MSG15-0100-NA-"
    "20261024141243.420000000Z-NA"
)

COLLECTION_ID = "EO:EUM:DAT:MSG:MSG15-RSS"
FILENAME = PRODUCT_ID + ".nat"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = DATA_DIR / FILENAME


def main():
    # Skip the download if the dataset already exists.
    if OUTPUT_FILE.is_file() and OUTPUT_FILE.stat().st_size > 0:
        print(f"Dataset already exists: {OUTPUT_FILE}")
        return

    # Read personal EUMETSAT API credentials.
    consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
    consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        sys.exit(
            "Missing EUMETSAT credentials.\n"
            "Set EUMETSAT_CONSUMER_KEY and "
            "EUMETSAT_CONSUMER_SECRET first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_name(
        OUTPUT_FILE.name + ".part"
    )

    try:
        print("Authenticating with EUMETSAT...")
        token = eumdac.AccessToken(
            (consumer_key, consumer_secret)
        )
        datastore = eumdac.DataStore(token)

        print(f"Looking for product: {PRODUCT_ID}")
        product = datastore.get_product(
            COLLECTION_ID, PRODUCT_ID
        )

        print("Downloading dataset...")
        with product.open() as source:
            with temporary_file.open("wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        temporary_file.replace(OUTPUT_FILE)
        print(f"Download completed: {OUTPUT_FILE}")

    except Exception as error:
        temporary_file.unlink(missing_ok=True)
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
