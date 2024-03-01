import os
import boto3
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

    inference_parameters = dict (
        temperature = settings["Temperature"],
        top_p = float(settings["TopP"]),
        top_k = int(settings["TopK"]),
        max_tokens_to_sample = int(settings["MaxTokenCount"]),
        stop_sequences =  []
    )

    request_key_mapping = {}
    model_strategy = BedrockModelStrategy()

    provider = bedrock_model_id.split(".")[0]

    if provider == "anthropic": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
        request_key_mapping = {'top_p': 'top_p', 'top_k': 'top_k', 'max_tokens_to_sample': 'max_tokens_to_sample'}
        model_strategy = AnthropicBedrockModelStrategy()
    elif provider == "ai21": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jurassic2.html
        # An error occurred (ValidationException) when calling the InvokeModelWithResponseStream operation: The model is unsupported for streaming
        request_key_mapping = {'top_p': 'topP', 'max_tokens_to_sample': 'maxTokens'}
    elif provider == "cohere": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
        request_key_mapping = {'top_p': 'p', 'top_k': 'k', 'max_tokens_to_sample': 'max_tokens'}
        model_strategy = CohereBedrockModelStrategy()
    elif provider == "amazon": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
        request_key_mapping = {'top_p': 'topP', 'max_tokens_to_sample': 'maxTokenCount'}
    elif provider == "meta": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        request_key_mapping = {'top_p': 'top_p', 'max_tokens_to_sample': 'max_gen_len'}
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
    cl.user_session.set("bedrock_model_strategy", model_strategy)
    
    
    

@cl.on_message
async def main(message: cl.Message):

    prompt_template = cl.user_session.get("prompt_template") 
    bedrock_runtime = cl.user_session.get("bedrock_runtime")
    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    request_key_mapping = cl.user_session.get("request_key_mapping")
    bedrock_model_strategy : BedrockModelStrategy = cl.user_session.get("bedrock_model_strategy")

    prompt = prompt_template.replace("{input}", message.content)

    print(request_key_mapping)

    request = {
        "prompt": prompt,
        "temperature": inference_parameters.get("temperature"),
        "top_p": inference_parameters.get("top_p"), #0.5,
        "top_k": inference_parameters.get("top_k"), #300,
        "max_tokens_to_sample": inference_parameters.get("max_tokens_to_sample"), #2048,
        "stop_sequences": []
    }
 
    request = {request_key_mapping.get(key, key): value for key, value in request.items()}
    print(f"{type(request)} {request}")

    msg = cl.Message(content="")

    await msg.send()

    try:

        response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))

        stream = response["body"]
        await bedrock_model_strategy.process_response_stream(stream, msg)

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()

    print("End")


class BedrockModelStrategy():

    def process_request(self, data: dict):
        pass

    async def process_response_stream(self, stream, msg : cl.Message):
        print("unknown")
        await msg.stream_token("unknown")

class AnthropicBedrockModelStrategy(BedrockModelStrategy):

    def process_request(self, data: dict):
        pass

    async def process_response_stream(self, stream, msg : cl.Message):
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

class CohereBedrockModelStrategy(BedrockModelStrategy):

    def process_request(self, data: dict):
        pass

    async def process_response_stream(self, stream, msg : cl.Message):
        #print("cohere")
        #await msg.stream_token("Cohere")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    if "generations" in object:
                        generations = object["generations"]
                        for generation in generations:
                            print(generation)
                            await msg.stream_token(generation["text"])
                            if "finish_reason" in generation:
                                finish_reason = generation["finish_reason"]
                                await msg.stream_token(f"\nfinish_reason={finish_reason}")