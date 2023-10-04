import json
from typing import Any, AsyncGenerator

import openai
import pandas as pd
import json
import math

import pytz
from datetime import datetime

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import Approach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


from externaldata.chat_code_utils import extract_code_from_message , appendimportsandprints, execute_extracted_code


import pytz
from datetime import datetime
def get_current_time(location):
    try:
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except:
        return "Sorry, I couldn't find the timezone for that location."

import pandas as pd
import json
def get_stock_market_data(index):
    available_indices = ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", "Financial Times Stock Exchange 100 Index"]

    if index not in available_indices:
        return "Invalid index. Please choose from 'S&P 500', 'NASDAQ Composite', 'Dow Jones Industrial Average', 'Financial Times Stock Exchange 100 Index'."

    # Read the CSV file
    data = pd.read_csv('stock_data.csv')

    # Filter data for the given index
    data_filtered = data[data['Index'] == index]

    # Remove 'Index' column
    data_filtered = data_filtered.drop(columns=['Index'])

    # Convert the DataFrame into a dictionary
    hist_dict = data_filtered.to_dict()

    for key, value_dict in hist_dict.items():
        hist_dict[key] = {k: v for k, v in value_dict.items()}

    return json.dumps(hist_dict)

import math
def calculator(num1, num2, operator):
    if operator == '+':
        return str(num1 + num2)
    elif operator == '-':
        return str(num1 - num2)
    elif operator == '*':
        return str(num1 * num2)
    elif operator == '/':
        return str(num1 / num2)
    elif operator == '**':
        return str(num1 ** num2)
    elif operator == 'sqrt':
        return str(math.sqrt(num1))
    else:
        return "Invalid operator"

import inspect

# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True

class ChatReadRetrieveReadApproach(Approach):

    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    NO_RESPONSE = "0"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """Assistant helps the users with their business problem related questions, and questions about using the quantum computing for different Business problems. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook.
Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about employee healthcare plans and the employee handbook.
You have access to Azure Cognitive Search index with 100's of documents.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'What is a MIS problem ?' },
        {'role' : ASSISTANT, 'content' : 'Explain MIS problem' },
        {'role' : USER, 'content' : 'What is QUBO Formulation?' },
        {'role' : ASSISTANT, 'content' : 'Explain QUBO and its formulation' }
    ]

    def __init__(
        self,
        search_client: SearchClient,
        openai_host: str,
        chatgpt_deployment: str,
        chatgpt_model: str,
        embedding_deployment: str,
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_host = openai_host
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    def get_search_query(self, chat_completion: dict[str, any], user_query: str):
        response_message = chat_completion["choices"][0]["message"]
        if function_call := response_message.get("function_call"):
            if function_call["name"] == "search_sources":
                arg = json.loads(function_call["arguments"])
                search_query = arg.get("search_query", self.NO_RESPONSE)
                if search_query != self.NO_RESPONSE:
                    return search_query
        elif query_text := response_message.get("content"):
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        #exclude_category = overrides.get("exclude_category") or None
        #expect_code_output = overrides.get("expect_code_output") or False
        #filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
        filter = self.build_filter(overrides, auth_claims)

        user_query_request = "Generate search query for: " + history[-1]["user"]

        functions = [
            {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London",
                        }
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_stock_market_data",
                "description": "Get the stock market data for a given index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "enum": ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", "Financial Times Stock Exchange 100 Index"]},
                    },
                    "required": ["index"],
                },    
            },
            {
                "name": "calculator",
                "description": "A simple calculator used to perform basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num1": {"type": "number"},
                        "num2": {"type": "number"},
                        "operator": {"type": "string", "enum": ["+", "-", "*", "/", "**", "sqrt"]},
                    },
                    "required": ["num1", "num2", "operator"],
                },
            }
        ]

        available_functions = {
            "get_current_time": get_current_time,
            "get_stock_market_data": get_stock_market_data,
            "calculator": calculator,
        } 

        #investment_keywords = [
        #'Invest money', 'maximise return', 'minimise risk', 'create portfolio',
        #'make me richer', 'Portfolio optimization', 'Portfolio optimisation',
        #'investment advice', 'investment advise']

        # Extract the latest user message


        last_user_message = history[-1]["user"].lower()

        '''if any(keyword.lower() in last_user_message for keyword in investment_keywords):
            ##return {"answer": "Sure I can help, please specify your budget and list of assets you want to invest in."}
            results = ["Source: This is a guided approach","Approach: Powered by Qatalive"]
            query_text = ["First Question"]
            msg_to_display = "The Message is for our guided approach"
            return {"data_points": results, "answer": "Sure I can help, please specify your budget and list of assets you want to invest in the following Investment form", "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}
        '''
        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_query_request,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_query_request),
        )

        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        #assistant_response = run_conversation(messages, functions, available_functions, deployment_name)

        #chat_completion = self.run_conversation(messages, functions, available_functions, self.chatgpt_deployment)
        
        #return chat_completion
        chat_completion = await openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            n=1,
            functions=functions,
            function_call="auto",
        )

        
        response_message = chat_completion["choices"][0]["message"]
        print (chat_completion)

        query_text = None
        if response_message.get("function_call"):
            print("Recommended Function call:")
            print(response_message.get("function_call"))
            
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            
            function_name = response_message["function_call"]["name"]
            
            # verify function exists
            if function_name not in available_functions:
                query_text = "Function " + function_name + " does not exist"
            function_to_call = available_functions[function_name]  
            
            # verify function has correct number of arguments
            function_args = json.loads(response_message["function_call"]["arguments"])
            if check_args(function_to_call, function_args) is False:
                query_text = "Invalid number of arguments for function: " + function_name
            function_response = function_to_call(**function_args)
            
            print("Output of function call:")
            print(function_response)
            query_text = function_response
          
            messages.append(
                {
                    "role": response_message["role"],
                    "function_call": {
                        "name": response_message["function_call"]["name"],
                        "arguments": response_message["function_call"]["arguments"],
                    },
                    "content": None
                }
            )

            # adding function response to messages
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            
            print("Messages in second request:")
            for message in messages:
                print(message)
            

            msg_to_display = "\n\n".join([str(message) for message in messages])
            #print(msg_to_display)
            extra_info = {
            "data_points": function_name,
            "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>"
            + msg_to_display.replace("\n", "<br>"),
            }

            second_response = openai.ChatCompletion.create(
               messages=messages,
                deployment_id=self.chatgpt_deployment
            )  # get a new response from GPT where it can see the function response

            #return second_response
            
            print ("second_response : ")
            print (second_response["choices"][0]["message"])

            chat_coroutine = openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1,
            stream=should_stream,
            )
            print (chat_completion)
            return (extra_info, chat_coroutine)

        else:
            query_text = self.get_search_query(chat_completion, history[-1]["user"])    
            print ("this is chat completion")
            print (chat_completion)
            print ("the is after printing completion")
            # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

            # If retrieval mode includes vectors, compute an embedding for the query
            if has_vector:
                embedding_args = {"deployment_id": self.embedding_deployment} if self.openai_host == "azure" else {}
                embedding = await openai.Embedding.acreate(**embedding_args, model=self.embedding_model, input=query_text)
                query_vector = embedding["data"][0]["embedding"]
            else:
                query_vector = None

            # Only keep the text query if the retrieval mode uses text, otherwise drop it
            print (query_text)
            if not has_text:
                query_text = None

            # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
            if overrides.get("semantic_ranker") and has_text:
                r = await self.search_client.search(
                    query_text,
                    filter=filter,
                    query_type=QueryType.SEMANTIC,
                    query_language="en-us",
                    query_speller="lexicon",
                    semantic_configuration_name="default",
                    top=top,
                    query_caption="extractive|highlight-false" if use_semantic_captions else None,
                    vector=query_vector,
                    top_k=50 if query_vector else None,
                    vector_fields="embedding" if query_vector else None,
                )
            else:
                r = await self.search_client.search(
                    query_text,
                    filter=filter,
                    top=top,
                    vector=query_vector,
                    top_k=50 if query_vector else None,
                    vector_fields="embedding" if query_vector else None,
                )
            if use_semantic_captions:
                results = [
                    doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc["@search.captions"]]))
                    async for doc in r
                ]
            else:
                results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
            content = "\n".join(results)

            follow_up_questions_prompt = (
                self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
            )

            # STEP 3: Generate a contextual and content specific answer using the search results and chat history

            # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
            prompt_override = overrides.get("prompt_template")
            if prompt_override is None:
                system_message = self.system_message_chat_conversation.format(
                    injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
                )
            elif prompt_override.startswith(">>>"):
                system_message = self.system_message_chat_conversation.format(
                    injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
                )
            else:
                system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

            messages = self.get_messages_from_history(
                system_message,
                self.chatgpt_model,
                history,
                history[-1]["user"] + "\n\nSources:\n" + content,
                max_tokens=self.chatgpt_token_limit,  # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            )
            msg_to_display = "\n\n".join([str(message) for message in messages])

            extra_info = {
                "data_points": results,
                "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>"
                + msg_to_display.replace("\n", "<br>"),
            }

            print (messages)
            chat_coroutine = openai.ChatCompletion.acreate(
                **chatgpt_args,
                model=self.chatgpt_model,
                messages=messages,
                temperature=overrides.get("temperature") or 0.7,
                max_tokens=1024,
                n=1,
                stream=should_stream,
            )
            return (extra_info, chat_coroutine)

    async def run_without_streaming(
        self, history: list[dict[str, str]], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=False
        )
        chat_resp = await chat_coroutine
        chat_content = chat_resp.choices[0].message.content
        extra_info["answer"] = chat_content
        return extra_info

    async def run_with_streaming(
        self, history: list[dict[str, str]], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=True
        )
        yield extra_info
        async for event in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            if event["choices"]:
                yield event

    def get_messages_from_history(
            self,
            system_prompt: str,
            model_id: str,
            history: list[dict[str, str]],
            user_conv: str,
            few_shots=[],
            max_tokens: int = 4096,
    ) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get("role"), shot.get("content"))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
                message_builder.append_message(self.ASSISTANT, "Perform Function requests for the user", index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break
        
        print ("I am just outside the for loop")
        messages = message_builder.messages
        return messages

    def run_conversation(messages, functions, available_functions, deployment_id):
    # Step 1: send the conversation and available functions to GPT
        response = openai.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=messages,
            functions=functions,
            function_call="auto", 
        )
        response_message = response["choices"][0]["message"]


        # Step 2: check if GPT wanted to call a function
        if response_message.get("function_call"):
            print("Recommended Function call:")
            print(response_message.get("function_call"))
            print()
            
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            
            function_name = response_message["function_call"]["name"]
            
            # verify function exists
            if function_name not in available_functions:
                return "Function " + function_name + " does not exist"
            function_to_call = available_functions[function_name]  
            
            # verify function has correct number of arguments
            function_args = json.loads(response_message["function_call"]["arguments"])
            if check_args(function_to_call, function_args) is False:
                return "Invalid number of arguments for function: " + function_name
            function_response = function_to_call(**function_args)
            
            print("Output of function call:")
            print(function_response)
            print()
            
            # Step 4: send the info on the function call and function response to GPT
            
            # adding assistant response to messages
            messages.append(
                {
                    "role": response_message["role"],
                    "function_call": {
                        "name": response_message["function_call"]["name"],
                        "arguments": response_message["function_call"]["arguments"],
                    },
                    "content": function_response
                }
            )

            # adding function response to messages
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response

            print("Messages in second request:")
            for message in messages:
                print(message)
            print()

            second_response = openai.ChatCompletion.create(
                messages=messages,
                deployment_id=deployment_id
            )  # get a new response from GPT where it can see the function response

            return second_response
