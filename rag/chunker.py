from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Splits raw documents into overlapping semantic chunks for efficient retrieval.
    proper chunking is required for efficient retrieval and processing.
    """
    def __init__(
            self,
            chunk_size: int =500,
            chunk_overlap: int =100,
    ):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            seperators=["\n\n","\n","."," ",""]
        )
    def chunk(self,documents:List[str])->List[str]:
        """
        Splits a list of raw documents into overlapping chunks.

        Args:
            documents (List[str]): List of raw document strings.

        Returns:
            List[str]: List of chunked document strings.
        """
        chunks=[]
        for doc in documents:
            chunks.extend(self.splitter.split_text(doc))
            return chunks
        