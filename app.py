import os
import boto3
from langchain.llms.bedrock import Bedrock
import chainlit as cl
from chainlit.input_widget import Select, Slider
from prompt_template import get_template
from typing import Optional
import json
import traceback
import logging

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == (AUTH_ADMIN_USR, AUTH_ADMIN_PWD):
    return cl.User(identifier=AUTH_ADMIN_USR, metadata={"role": "admin", "provider": "credentials"})
  else:
    return None
  
#@cl.author_rename
#def rename(orig_author: str):
#    mapping = {
#        "ConversationChain": bedrock_model_id
#    }
#    return mapping.get(orig_author, orig_author)

@cl.on_chat_start
async def main():
    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    
    response = bedrock.list_foundation_models(
        byOutputModality="TEXT"
    )
    
    model_ids = []
    for item in response["modelSummaries"]:
        model_ids.append(item['modelId'])
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Amazon Bedrock - Model",
                values=model_ids,
                initial_index=model_ids.index("anthropic.claude-v2"),
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.3,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id = "TopP",
                label = "Top P",
                initial = 1,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopK",
                label = "Top K",
                initial = 250,
                min = 0,
                max = 500,
                step = 5,
            ),
            Slider(
                id="MaxTokenCount",
                label="Max Token Size",
                initial=1024,
                min=256,
                max=4096,
                step=256,
            ),
        ]
    ).send()
    await setup_agent(settings)

@cl.on_settings_update
async def setup_agent(settings):

    bedrock_model_id = settings["Model"]
    TEMPERATURE = settings["Temperature"]

    inference_parameters = dict (
        temperature = settings["Temperature"]
    )
    request_key_mapping = {}
    
    llm = Bedrock(
        region_name = AWS_REGION,
        model_id = bedrock_model_id,
        model_kwargs = {
            "temperature": TEMPERATURE,
        }
    )

    provider = bedrock_model_id.split(".")[0]

    TOP_P = float(settings["TopP"])
    TOP_K = int(settings["TopK"])
    MAX_TOKEN_SIZE = int(settings["MaxTokenCount"])
    
    # Model specific adjustments
    if provider == "anthropic": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
        llm.model_kwargs["top_p"] = TOP_P
        llm.model_kwargs["top_k"] = TOP_K
        llm.model_kwargs["max_tokens_to_sample"] = MAX_TOKEN_SIZE
    elif provider == "ai21": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jurassic2.html
        llm.model_kwargs["topP"] = TOP_P
        llm.model_kwargs["maxTokens"] = MAX_TOKEN_SIZE
        llm.streaming = False
    elif provider == "cohere": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
        llm.model_kwargs["p"] = TOP_P
        llm.model_kwargs["k"] = TOP_K
        llm.model_kwargs["max_tokens"] = MAX_TOKEN_SIZE    
        llm.model_kwargs["stream"] = True
        request_key_mapping = {'top_p': 'p', 'top_k': 'k', 'max_tokens_to_sample': 'max_tokens'}
    elif provider == "amazon": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
        llm.model_kwargs["topP"] = TOP_P
        llm.model_kwargs["maxTokenCount"] = MAX_TOKEN_SIZE
    elif provider == "meta": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        llm.model_kwargs["top_p"] = TOP_P
        llm.model_kwargs["max_gen_len"] = MAX_TOKEN_SIZE
    else:
        print(f"Unsupported Provider: {provider}")
        raise ValueError(f"Error, Unsupported Provider: {provider}")

    prompt_template = get_template(provider)

    cl.user_session.set("prompt_template", prompt_template)
    
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    cl.user_session.set("bedrock_runtime", bedrock_runtime)
    cl.user_session.set("bedrock_model_id", bedrock_model_id)
    cl.user_session.set("inference_parameters", inference_parameters)
    cl.user_session.set("request_key_mapping", request_key_mapping)
    
    

@cl.on_message
async def main(message: cl.Message):

    prompt_template = cl.user_session.get("prompt_template") 
    bedrock_runtime = cl.user_session.get("bedrock_runtime")
    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    request_key_mapping = cl.user_session.get("request_key_mapping")

    prompt = prompt_template.replace("{input}", message.content)

    print(request_key_mapping)

    request = {
        "prompt": prompt,
        "temperature": inference_parameters.get("temperature"),
        "top_p": 0.5,
        "top_k": 300,
        "max_tokens_to_sample": 2048,
        "stop_sequences": []
    }
 
    request = {request_key_mapping.get(key, key): value for key, value in request.items()}
    print(request)


    msg = cl.Message(content="")

    try:

        response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))

        stream = response["body"]
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    #print(object)
                    if "completion" in object:
                        completion = object["completion"]
                        #print(completion)
                        await msg.stream_token(completion)
                    stop_reason = None
                    if "stop_reason" in object:
                        stop_reason = object["stop_reason"]
                    
                    if stop_reason == 'stop_sequence':
                        invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                        if invocation_metrics:
                            input_token_count = invocation_metrics["inputTokenCount"]
                            output_token_count = invocation_metrics["outputTokenCount"]
                            latency = invocation_metrics["invocationLatency"]
                            lag = invocation_metrics["firstByteLatency"]
                            stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            await msg.stream_token(f"\n\n{stats}")

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()

    print("End")