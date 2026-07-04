from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from regex import splititer

SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()


SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''.strip()


def split_text():
    splitter = RecursiveCharacterTextSplitter(
        ["\n\n", "\n", "", " "],
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"sample text size : {len(SAMPLE_TEXT)}")
    print(f"number of chunks : {len(chunks)}")
    print(f"size of each chunks : {[len(chunk) for chunk in chunks]}")
    print(f"\nsample from first chunk : \n{chunks[0][:200]}...")


def chunk_size_comparison():
    chunk_sizes = [200, 500, 1000]

    def get_splitter_configuration(chunk_size: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_size // 5
        )

    splitter_configurations = [get_splitter_configuration(c_s) for c_s in chunk_sizes]

    chunks_total = [
        splitter.split_text(SAMPLE_TEXT) for splitter in splitter_configurations
    ]

    print("=== Chunk size comparison ===")

    print(f"chunk size {len(SAMPLE_TEXT)}")

    for size, chunks in zip(chunk_sizes, chunks_total):
        print(f"number of chunks for size {size}: {len(chunks)}")
        print(
            f"size of each chunk with chunk size {size}: {[len(chunk) for chunk in chunks]}"
        )


def importance_of_overlap():
    text = "This is a comparison of Text splitter. " * 10

    splitter_without_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=0
    )
    splitter_with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=20
    )

    chunk_without_overlap = splitter_without_overlap.split_text(text)
    chunk_with_overlap = splitter_with_overlap.split_text(text)

    print("without overlap")

    print(f" Chunk 1 end : ...{chunk_without_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunk_without_overlap[1][:20]}...")

    print("\nwith overlap:")
    print(f" Chunk 1 end: ...{chunk_with_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunk_with_overlap[0][:-20]}...")


def code_splitter():
    splliter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=500, chunk_overlap=50
    )

    chunks = splliter.split_text(SAMPLE_CODE)

    print(f"total code char count: {len(SAMPLE_CODE)}")
    print(f"total chunks : {len(chunks)}")
    print(f"chars in each chunks : {[len(c) for c in chunks]}")

    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i} ({len(chunk)} chars).")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


def pdf_splitter():
    SAMPLE_FILE = "docs/test.pdf"

    pdf = PyPDFLoader(SAMPLE_FILE).load()

    print(f"Loaded {len(pdf)} from pdf.")

    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_pdf = pdf_splitter.split_documents(pdf)

    print(f"split into {len(split_pdf)} chunks")
    print(f"first chunk metadata {split_pdf[0].metadata}")
    print(f"first chunk content: {split_pdf[0].page_content[:200]}...")
    print(f"last chunk content: ...{split_pdf[-1].metadata}")


if __name__ == "__main__":
    # split_text()
    # chunk_size_comparison()
    # importance_of_overlap()
    # code_splitter()
    pdf_splitter()
