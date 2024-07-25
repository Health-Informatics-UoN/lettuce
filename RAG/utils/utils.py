import glob
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader


def load_folder_contents(path: str, glob: str, doc_type: str, *args, **kwargs):
    if doc_type == "csv":
        loader = DirectoryLoader(
            path,
            glob,
            loader_cls=CSVLoader,
            loader_kwargs=kwargs["loader_kwargs"],
            show_progress=True,
            use_multithreading=kwargs["use_multithreading"],
        )
    else:
        try:
            loader = DirectoryLoader(path, glob)
        except Exception as e:
            print(e)
            print("Error loading documents")
            return None
    documents = loader.load()

    return documents


def load_csv_file_names(path: str):
    files = glob.glob(path + "/**/*.csv", recursive=True)
    return files
