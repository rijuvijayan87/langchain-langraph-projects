import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()


def load_text_document():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        print(f"file path -> {temp_file.name}")
        temp_file.write(
            b"this is a temporary file \n This file is used to handle and test text document using langchain"
        )
        temp_file_path = temp_file.name

    try:
        doc_loader = TextLoader(temp_file_path)
        documents = doc_loader.load()
        for doc in documents:
            print(doc.page_content)
    except:
        os.remove(temp_file_path)


def lazy_loader():
    """Lazy loader to load document from directory"""
    with tempfile.TemporaryDirectory() as tmpDir:
        print(f"temporary directory created : {tmpDir}")
        for i in range(10):
            path = Path(tmpDir) / f"doc{i}.txt"
            path.write_text(
                f"This is a content of document index# {i}. Eventually be used for text loader"
            )

        loader = DirectoryLoader(path=tmpDir, glob="*.txt", loader_cls=TextLoader)
        print(f"Initialized lazy loader for directory {tmpDir}")

        for doc in loader.lazy_load():
            print(doc.metadata["source"])


def generator_example(num: int):
    i = 0
    while i < num:
        yield i
        i += 1


if __name__ == "__main__":
    # load_text_document()
    # lazy_loader()

    for x in generator_example(3):
        print(x)

    print(sum(generator_example(5)))  # 10

    print(list(generator_example(5)))
