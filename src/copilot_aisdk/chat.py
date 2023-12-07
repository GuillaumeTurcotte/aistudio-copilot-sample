# enable type annotation syntax on Python versions earlier than 3.9
from __future__ import annotations

import os
import openai
import jinja2
import pathlib

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from copilot_aisdk.plugins.br_plugin.br import BR

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import RawVectorQuery
from semantic_kernel.planning import StepwisePlanner
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.planning.basic_planner import BasicPlanner

from streaming_utils import add_context_to_streamed_response
from semantic_kernel.core_skills import TimeSkill

import logging

logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)

templateLoader = jinja2.FileSystemLoader(pathlib.Path(__file__).parent.resolve())
templateEnv = jinja2.Environment(loader=templateLoader)
system_message_template = templateEnv.get_template("system-message.jinja2")


async def get_documents(query, num_docs=5):
    #  retrieve documents relevant to the user's question from Cognitive Search
    search_client = SearchClient(
        endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]),
        index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"])

    # generate a vector embedding of the user's question
    embedding = await openai.Embedding.acreate(input=query,
                                               model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
                                               deployment_id=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
    embedding_to_query = embedding["data"][0]["embedding"] # type: ignore

    context = ""
    async with search_client:
        # use the vector embedding to do a vector search on the index
        vector_query = RawVectorQuery(vector=embedding_to_query, k=num_docs, fields="vector")
        results = await search_client.search(
            search_text="",
            search_fields=["title", "chunk"],
            vector_queries=[vector_query],
            select=["title", "chunk"])

        async for result in results:
            context += f"\n>>> From: {result['title']}\n{result['chunk']}"

    return context


async def chat_completion(messages: list[dict], stream: bool = False,
                          session_state: any = None, context: dict[str, any] = {}): # type: ignore

    br_plugin = BR(
        number_of_docs = context.get("num_retrieved_docs", 5),
        embedding_model_deployment = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
        chat_model_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=context.get("temperature", 0.7)
    )

     # Add the customer support plugin to the kernel
    kernel = sk.Kernel()
    kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(
            os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"), # type: ignore
            os.getenv("OPENAI_API_BASE"),
            os.getenv("OPENAI_API_KEY"),
        )
    )
    function_base = kernel.import_skill(br_plugin, skill_name="BR") # type: ignore

    # get search documents for the last user message in the conversation
    user_message = messages[-1]["content"]

    # Add hints to the customer ask
    # TODO: Make the ID configurable
    ask = user_message + "\nThe BR number is 61749; only use this if you need information about a specific BR."

    # Create and run plan based on the customer ask
    planner = BasicPlanner()
    plan = await planner.create_plan_async(user_message, kernel) # type: ignore
    #result = await plan.invoke_async()
    # Execute the plan
    result = await planner.execute_plan_async(plan, kernel)
    #result = await kernel.run_async(plan)
    #result = await kernel.run_async(function_base["GetBRInformation"], input_str=user_message)

     # limit size of returned context
    context = br_plugin.context # type: ignore
    if len(context) > 32000:
        context = context[:32000] # type: ignore

    return result
    # return {
    #     "choices": [{
    #         "index": 0,
    #         "message": {
    #             "role": "assistant",
    #             "content": result.result
    #         },
    #         "context": context,
    #     }]
    # }
