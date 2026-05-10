from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from loguru import logger
from src.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, EMBED_DIM


SCHEMA_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="slide_index", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
]

INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}


class MilvusStore:
    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        collection_name: str = MILVUS_COLLECTION,
    ):
        self.collection_name = collection_name
        connections.connect(host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            logger.info(f"Loading existing collection '{self.collection_name}'")
            col = Collection(self.collection_name)
            col.load()
            return col

        logger.info(f"Creating collection '{self.collection_name}'")
        schema = CollectionSchema(fields=SCHEMA_FIELDS, description="AutoReport knowledge base")
        col = Collection(name=self.collection_name, schema=schema)
        col.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        col.load()
        return col

    def insert(self, texts: list[str], embeddings: list[list[float]], sources: list[str],
               doc_types: list[str], slide_indices: list[int]) -> int:
        data = [texts, sources, doc_types, slide_indices, embeddings]
        result = self.collection.insert(data)
        self.collection.flush()
        count = len(result.primary_keys)
        logger.info(f"Inserted {count} vectors into '{self.collection_name}'")
        return count

    def search(self, query_vector: list[float], top_k: int = 5,
               doc_type_filter: str | None = None) -> list[dict]:
        expr = f'doc_type == "{doc_type_filter}"' if doc_type_filter else None
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
            expr=expr,
            output_fields=["text", "source", "doc_type", "slide_index"],
        )
        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "doc_type": hit.entity.get("doc_type"),
                "slide_index": hit.entity.get("slide_index"),
            })
        return hits

    def count(self) -> int:
        return self.collection.num_entities

    def drop(self):
        utility.drop_collection(self.collection_name)
        logger.warning(f"Dropped collection '{self.collection_name}'")

    def health_check(self) -> bool:
        try:
            _ = self.collection.num_entities
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False
