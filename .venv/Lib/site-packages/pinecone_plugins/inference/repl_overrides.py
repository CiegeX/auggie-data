from .core.client.models import (
    EmbeddingsList as OpenAPIEmbeddingsList,
    Embedding as OpenAPIEmbedding,
    Document as OpenAPIDocument,
    RerankResult as OpenAPIRerankResult,
    RankedDocument as OpenAPIRankedDocument,
)


def present_list(mylist):
    if len(mylist) <= 5:
        # Show list as usual when fewer than 5 items.
        # This number is arbitrary and can be adjusted
        # but it seems silly to show the abbreviated
        # message with (2  more) or whatever when the
        # number of items is so small and it's no problem
        # to display the real values.
        return f"[{', '.join(repr(x) for x in mylist)}]"
    first_part = ", ".join(repr(x) for x in mylist[:2])
    last_part = ", ".join(repr(x) for x in mylist[-2:])
    formatted_values = f"[{first_part}, ..., {last_part}]"
    return formatted_values


def embedding_to_str(self: OpenAPIEmbedding):
    formatted_values = present_list(self.values)
    return """{{'values': {formatted_values}}}""".format(
        formatted_values=formatted_values
    )


def embedding_list_to_string(self: OpenAPIRerankResult):
    if len(self.data) == 0:
        formatted_embeddings = "[]"
    elif len(self.data) <= 5:
        formatted_embeddings = (
            "[\n    "
            + ",\n    ".join([embedding.to_str() for embedding in self.data])
            + "\n  ]"
        )
    else:
        omitted_msg = f"... ({len(self.data) - 4} more embeddings) ..."
        items_to_show = (
            [e.to_str() for e in self.data[:2]]
            + [omitted_msg]
            + [e.to_str() for e in self.data[-2:]]
        )
        formatted_embeddings = "[\n    " + ",\n    ".join(items_to_show) + "\n  ]"

    return """EmbeddingsList(
  model='{self.model}',
  data={formatted_embeddings},
  usage={self.usage}
)""".format(
        self=self, formatted_embeddings=formatted_embeddings
    )


def rank_result_document_to_str(self: OpenAPIDocument):
    return f",\n        ".join([f"{k}={repr(v)}" for k, v in self.to_dict().items()])


def rank_result_item_to_str(self: OpenAPIRankedDocument):
    if self.document is None:
        document_str = ""
    else:
        document_str = f"document={{\n        {self.document}\n    }}"

    return f"{{\n    index={self.index},\n    score={self.score},\n    {document_str}\n  }}"


def rerank_result_to_str(self: OpenAPIRerankResult):
    if len(self.data) == 0:
        formatted_rerank = "[]"
    else:
        formatted_rerank = (
            "["
            + ",".join([rank_result_item.to_str() for rank_result_item in self.data])
            + "]"
        )

    return f"""RerankResult(
  model='{self.model}',
  data={formatted_rerank},
  usage={self.usage}
)"""


def install_repl_overrides():
    OpenAPIEmbedding.to_str = embedding_to_str
    OpenAPIEmbeddingsList.to_str = embedding_list_to_string
    OpenAPIDocument.to_str = rank_result_document_to_str
    OpenAPIRankedDocument.to_str = rank_result_item_to_str
    OpenAPIRerankResult.to_str = rerank_result_to_str
