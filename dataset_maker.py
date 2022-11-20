from google_images_download import google_images_download

queries = [
    # "plate on table",
    # "empty breakfast bowl",
    # "dirty drinking glass",
    # "clean drinking glass on table",
    # "empty plate after dinner",
    # "empty dirty bowl",
    # "empty plate on table"
    "mug on table",
    "cup on table"
]
def download_images(query):
    if __name__ == "__main__":
        arguments = {
            "keywords": query,
            "limit": 100,
            "output_directory": "datasets",
            "chromedriver": "C:\Program Files (x86)\chromedriver\chromedriver.exe"

        }
        response = google_images_download.googleimagesdownload()
        response.download(arguments)

for query in queries:
    download_images(query)