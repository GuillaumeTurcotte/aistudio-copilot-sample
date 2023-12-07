from semantic_kernel.skill_definition import (
    sk_function,
)
import openai
import json
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding, AzureChatCompletion
from semantic_kernel.connectors.ai.complete_request_settings import CompleteRequestSettings
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential
import re

import os

class BR:
    def __init__(self, number_of_docs, embedding_model_deployment, chat_model_deployment, temperature=0.5):
        self.number_of_docs = number_of_docs
        self.embedding_model_deployment = embedding_model_deployment
        self.chat_model_deployment = chat_model_deployment
        self.temperature = temperature
        self.context = ""

    @sk_function(
        description="Give all the information related to a Business Request (BR) given the br_number",
        name="GetBRInformation",
        input_description="The br_number of the BR to retreive information about"
    )
    async def GetBRInformation(self, input: str) -> str:
        #  retrieve documents relevant to the user's question from Cognitive Search
        search_client = SearchClient(
            endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]),
            index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"])

        # generate a vector embedding of the user's question
        numbers = re.findall(r'\b\d{5}\b', input)
        numbers_str = ""
        if numbers:
            numbers_str = ", ".join(numbers)

        chunks = ""
        async with search_client:
            results = await search_client.search(
                search_text=numbers_str,
                search_fields=["title"],
                select=["title", "chunk"])
            
            async for result in results:
                #print(result['title'])
                chunks += f"\n>>> From: {result['title']}\n{result['chunk']}"

        self.context += "## GetBRInformation data\n" + str(chunks) + "\n\n"

        # Initialize a kernel so we can get the answer from the context
        kernel = sk.Kernel()
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(
                self.chat_model_deployment,
                os.getenv("OPENAI_API_BASE"),
                os.getenv("OPENAI_API_KEY"),
            )
        )

        # Import the chat plugin from the plugins directory.
        plugins_directory = os.path.dirname(os.path.realpath(__file__)) + "/../../plugins"

        chat_plugin = kernel.import_semantic_skill_from_directory(
            plugins_directory, "chat_plugin"
        )

        # Set context variables
        variables = sk.ContextVariables()
        variables["question"] = input
        variables["context"] = chunks

        # Change temperature of qna semantic function for evaluations
        chat_plugin["qna"].set_ai_configuration(settings = CompleteRequestSettings(
            temperature=self.temperature,
            max_tokens=800
        ))

        # Run the qna function with the right temperature and context.
        result = await (
            kernel.run_async(chat_plugin["qna"], input_vars=variables)
        )

        return result.result

    @sk_function(
        description="Use this function to get the amount of BRs that are forecasted to be delivered in month and year X and Y.",
        name="BrForecast",
        input_description="The question about the BRs; include as many details as possible in the question YEAR AND MONTH",
    )
    async def BrForecast(self, input: str) -> str:
        #  retrieve documents relevant to the user's question from Cognitive Search
        search_client = SearchClient(
            endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]),
            index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"])

        embedding = await openai.Embedding.acreate(input=input,
            model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
            deployment_id=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
        query_vector = embedding["data"][0]["embedding"] # type: ignore

        chunks = ""
        async with search_client:
            # use the vector embedding to do a vector search on the index
            vector_query = RawVectorQuery(vector=query_vector, k=self.number_of_docs, fields="vector")
            results = await search_client.search(
                search_text="",
                vector_queries=[vector_query],
                select=["title", "chunk"])
            
            async for result in results:
                #print(result['title'])
                chunks += f"\n>>> From: {result['title']}\n{result['chunk']}"

        self.context += "## BrForecast data\n" + str(chunks) + "\n\n"

        # Initialize a kernel so we can get the answer from the context
        kernel = sk.Kernel()
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(
                self.chat_model_deployment,
                os.getenv("OPENAI_API_BASE"),
                os.getenv("OPENAI_API_KEY"),
            )
        )

        # Import the chat plugin from the plugins directory.
        plugins_directory = os.path.dirname(os.path.realpath(__file__)) + "/../../plugins"

        chat_plugin = kernel.import_semantic_skill_from_directory(
            plugins_directory, "chat_plugin"
        )

        # limit size of returned context
        #if len(chunks) > 32000:
        #    chunks = chunks[:32000] # type: ignore

        # Set context variables
        variables = sk.ContextVariables()
        variables["question"] = input
        variables["context"] = chunks

        # Change temperature of qna semantic function for evaluations
        chat_plugin["qna"].set_ai_configuration(settings = CompleteRequestSettings(
            temperature=self.temperature,
            max_tokens=800
        ))

        # Run the qna function with the right temperature and context.
        result = await (
            kernel.run_async(chat_plugin["qna"], input_vars=variables)
        )

        return result.result