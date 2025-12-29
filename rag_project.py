# ===============================
# IMPORTS
# ===============================

import re
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===============================
# SESSION STORE
# ===============================

SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


# ===============================
# CONFIG
# ===============================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GGVSEE_PDF_PATH = "/Users/burakmemis/Downloads/ggvsee-englisch.pdf"
GGVSEE_CHROMA_DIR = "./chroma_db_ggvsee"
GGVSEE_RETRIEVER_K = 4

ICS2_PDF_PATH = "/Users/burakmemis/Downloads/ICS2-BCP-EO-(2025-12-18)-v1.70.pdf"
ICS2_CHROMA_DIR = "./chroma_db"
ICS2_RETRIEVER_K = 3

OLLAMA_MODEL = "mistral"
OLLAMA_TEMPERATURE = 0.3
SESSION_ID = "ics2_chat"

SYSTEM_PROMPT = """
You are a legal and compliance assistant.
- The user may ask questions in Turkish.
- Source documents are in English.
- ALWAYS answer in Turkish.
- Use ONLY the provided context.
- Cite section numbers exactly.
- If the answer is not explicitly stated in the context, say:
"Bu soru verilen dokümanda açıkça tanımlanmamıştır."
""".strip()


# ===============================
# SPLITTERS
# ===============================

@dataclass
class RegulationSplitterConfig:
    # "Section 6" gibi başlıklar
    section_regex: str = r"(?m)^\s*Section\s+(?P<section>\d+[a-z]?)\s*$"

    # "(1)" "(2)" gibi paragraf girişleri
    paragraph_regex: str = r"(?m)^\s*\((?P<para>\d+)\)\s+"

    # Chunk parametreleri
    chunk_size: int = 900
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None

    # Çok uzun blokları “soft split”e sokma eşiği
    min_soft_split_length: int = 1200

    # Metadata
    source_name: str = "GGVSee"

    # PDF’den gelen header/footer gürültüsünü azaltmak için basit temizlik
    cleanup: bool = True

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


class GGVSeeRegulationSplitter:
    """
    GGVSee / benzeri mevzuatlar için:
    Section -> Paragraph -> Soft Split (uzunsa)
    """

    def __init__(self, config: RegulationSplitterConfig):
        self.cfg = config
        self.section_pat = re.compile(self.cfg.section_regex)
        self.par_pat = re.compile(self.cfg.paragraph_regex)

        self.soft_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=self.cfg.separators,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []

        for doc in docs:
            text = doc.page_content or ""

            if self.cfg.cleanup:
                text = self._clean_common_noise(text)

            # 1) Section'lara böl
            sec_matches = list(self.section_pat.finditer(text))

            # Section yoksa: soft split fallback
            if not sec_matches:
                out.extend(
                    self._soft_split_with_metadata(
                        doc, section=None, paragraph=None
                    )
                )
                continue

            for i, match in enumerate(sec_matches):
                sec_start = match.start()
                sec_end = (
                    sec_matches[i + 1].start()
                    if i + 1 < len(sec_matches)
                    else len(text)
                )

                section_id = match.group("section")
                section_block = text[sec_start:sec_end].strip()

                # 2) Section içinde (1)(2) paragraf bloklarına böl
                para_docs = self._split_section_into_paragraphs(
                    section_block=section_block,
                    base_meta=doc.metadata,
                    section_id=section_id,
                )

                out.extend(para_docs)

        return out

    def _split_section_into_paragraphs(
        self,
        section_block: str,
        base_meta: Dict,
        section_id: str,
    ) -> List[Document]:
        out: List[Document] = []
        para_matches = list(self.par_pat.finditer(section_block))

        # Paragraf (1)(2) bulunamazsa: Section bazında davran
        if not para_matches:
            base_doc = Document(
                page_content=section_block,
                metadata={
                    **base_meta,
                    "source": self.cfg.source_name,
                    "section": section_id,
                },
            )
            if len(section_block) >= self.cfg.min_soft_split_length:
                out.extend(
                    self._soft_split_with_metadata(
                        base_doc, section=section_id, paragraph=None
                    )
                )
            else:
                out.append(base_doc)
            return out

        for i, para_match in enumerate(para_matches):
            para_start = para_match.start()
            para_end = (
                para_matches[i + 1].start()
                if i + 1 < len(para_matches)
                else len(section_block)
            )

            para_id = para_match.group("para")
            para_text = section_block[para_start:para_end].strip()

            base_doc = Document(
                page_content=para_text,
                metadata={
                    **base_meta,
                    "source": self.cfg.source_name,
                    "section": section_id,
                    "paragraph": para_id,
                },
            )

            if len(para_text) >= self.cfg.min_soft_split_length:
                out.extend(
                    self._soft_split_with_metadata(
                        base_doc, section=section_id, paragraph=para_id
                    )
                )
            else:
                out.append(base_doc)

        return out

    def _soft_split_with_metadata(
        self,
        doc: Document,
        section: Optional[str],
        paragraph: Optional[str],
    ) -> List[Document]:
        sub_docs = self.soft_splitter.split_documents([doc])

        for sub_doc in sub_docs:
            sub_doc.metadata["source"] = doc.metadata.get(
                "source", self.cfg.source_name
            )
            if section is not None:
                sub_doc.metadata["section"] = section
            if paragraph is not None:
                sub_doc.metadata["paragraph"] = paragraph

        return sub_docs

    def _clean_common_noise(self, text: str) -> str:
        # Örnek: tek başına sayfa numarası satırları ("2", "15" gibi)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

        # Çoklu boşlukları sadeleştir
        text = re.sub(r"[ \t]+", " ", text)

        # Fazla boş satırları azalt
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


@dataclass
class ICS2SplitterConfig:
    section_regex: str = r"(?m)^(?P<section>\d+(?:\.\d+)+)\s+.+$"
    min_soft_split_length: int = 1200
    chunk_size: int = 900
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None
    source_name: str = "ICS2-BCP-EO-v1.70"

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


class ICS2LegalSplitter:
    """
    ICS2 Business Continuity Plan için
    section-aware, hukuki RAG uyumlu splitter.
    """

    def __init__(self, config: ICS2SplitterConfig):
        self.config = config
        self.section_pattern = re.compile(config.section_regex)

        self.soft_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        structured_docs: List[Document] = []

        for doc in docs:
            text = doc.page_content
            matches = list(self.section_pattern.finditer(text))

            # Eğer section yoksa -> soft split
            if not matches:
                structured_docs.extend(
                    self.soft_splitter.split_documents([doc])
                )
                continue

            for idx, match in enumerate(matches):
                start = match.start()
                end = (
                    matches[idx + 1].start()
                    if idx + 1 < len(matches)
                    else len(text)
                )

                section_id = match.group("section")
                section_text = text[start:end].strip()

                base_doc = Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "section": section_id,
                        "source": self.config.source_name,
                    },
                )

                if len(section_text) >= self.config.min_soft_split_length:
                    structured_docs.extend(
                        self._soft_split_with_metadata(base_doc)
                    )
                else:
                    structured_docs.append(base_doc)

        return structured_docs

    def _soft_split_with_metadata(
        self, doc: Document
    ) -> List[Document]:
        sub_docs = self.soft_splitter.split_documents([doc])

        for sub_doc in sub_docs:
            sub_doc.metadata["section"] = doc.metadata["section"]
            sub_doc.metadata["source"] = doc.metadata["source"]

        return sub_docs


# ===============================
# PIPELINE HELPERS
# ===============================

def load_pdf(path: str) -> List[Document]:
    return PyMuPDFLoader(path).load()


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str,
) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def build_ggvsee_retriever(
    embeddings: HuggingFaceEmbeddings,
):
    docs = load_pdf(GGVSEE_PDF_PATH)
    config = RegulationSplitterConfig(
        source_name="GGVSee (2019 recast)",
        chunk_size=900,
        chunk_overlap=200,
        min_soft_split_length=1200,
    )
    chunks = GGVSeeRegulationSplitter(config).split(docs)

    print("Toplam chunk:", len(chunks))
    print(chunks[0].metadata)
    print(chunks[0].page_content[:400])

    vectorstore = build_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        persist_dir=GGVSEE_CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": GGVSEE_RETRIEVER_K}
    )


def build_ics2_retriever(
    embeddings: HuggingFaceEmbeddings,
):
    docs = load_pdf(ICS2_PDF_PATH)
    config = ICS2SplitterConfig()
    chunks = ICS2LegalSplitter(config).split(docs)
    vectorstore = build_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        persist_dir=ICS2_CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": ICS2_RETRIEVER_K}
    )


def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    return (
        {
            "context": itemgetter("input") | retriever,
            "input": itemgetter("input"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def run_chat_loop(chat_chain, session_id: str) -> None:
    print("📘 ICS2 Chat başlatıldı. Çıkmak için 'exit' yaz.\n")

    while True:
        user_input = input("🧑 Sen: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Görüşürüz!")
            break

        response = chat_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        print("\n🤖 Asistan:")
        print(response)
        print("-" * 50)


def run_single_query(rag_chain) -> None:
    question = input("Question: ")
    response = rag_chain.invoke({"input": question})

    print("\n📄 ANSWER:")
    print(response)


def main() -> None:
    embeddings = build_embeddings(EMBEDDING_MODEL)

    build_ggvsee_retriever(embeddings)
    retriever = build_ics2_retriever(embeddings)

    llm = Ollama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
    rag_chain = build_rag_chain(retriever, llm)

    chat_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    run_chat_loop(chat_chain, SESSION_ID)
    run_single_query(rag_chain)


if __name__ == "__main__":
    main()
