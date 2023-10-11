import os

# set environment variables before the openai SDK gets imported
from dotenv import load_dotenv
load_dotenv()

from azure.ai.generative import AIClient
from azure.identity import DefaultAzureCredential

from azure.ai.generative.operations._index_data_source import LocalSource, ACSOutputConfig
from azure.ai.generative.functions.build_mlindex import build_mlindex

# build the index using the product catalog docs from data/3-product-info
def build_cogsearch_index(index_name, path_to_data):
    # Set up environment variables for cog search SDK
    os.environ["AZURE_COGNITIVE_SEARCH_TARGET"] = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
    os.environ["AZURE_COGNITIVE_SEARCH_KEY"] = os.environ["AZURE_AI_SEARCH_KEY"]
    
    client = AIClient.from_config(DefaultAzureCredential())
    
    # Use the same index name when registering the index in AI Studio
    index = build_mlindex(
        output_index_name=index_name,
        vector_store="azure_cognitive_search",
        embeddings_model = f"azure_open_ai://deployment/{os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT']}/model/{os.environ['AZURE_OPENAI_EMBEDDING_MODEL']}",
        data_source_url="https://product_info.com",
        index_input_config=LocalSource(input_data=path_to_data),
        acs_config=ACSOutputConfig(
            acs_index_name=index_name,
        ),
    )

    # register the index so that it shows up in the project
    cloud_index = client.mlindexes.create_or_update(index)
    
    print(f"Created index '{cloud_index.name}'")
    print(f"Local Path: {index.path}")
    print(f"Cloud Path: {cloud_index.path}")
    
if __name__ == "__main__":
    build_cogsearch_index("contoso_product_index", "data/3-product_info")



