import chainlit as cl
import json

class BedrockModelStrategy():

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        pass

    def send_request(self, request:dict, bedrock_runtime, bedrock_model_id:str):
        response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))
        return response

    async def process_response(self, response, msg : cl.Message):
        stream = response["body"]
        await self.process_response_stream(stream, msg)

    async def process_response_stream(self, stream, msg : cl.Message):
        print("unknown")
        await msg.stream_token("unknown")

class BedrockModelStrategyFactory():

    @staticmethod
    def create(bedrock_model_id : str) -> BedrockModelStrategy:

        model_strategy = None

        provider = bedrock_model_id.split(".")[0]

        if provider == "anthropic": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
            model_strategy = AnthropicBedrockModelStrategy()
        elif provider == "cohere": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
            model_strategy = CohereBedrockModelStrategy()
        elif provider == "amazon": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
            model_strategy = TitanBedrockModelStrategy()
        else:
            #raise ValueError(f"Not Supported Model. Model={bedrock_model_id} Provide={provider}")
            model_strategy = BedrockModelStrategy()
        
        return model_strategy
    

class AnthropicBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            "top_k": inference_parameters.get("top_k"), #300,
            "max_tokens_to_sample": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"stop_sequences": []
        }
        return request

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

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,
            "temperature": inference_parameters.get("temperature"),
            "p": inference_parameters.get("top_p"), #0.5,
            "k": inference_parameters.get("top_k"), #300,
            "max_tokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            "stream": True,
            #"stop_sequences": []
        }
        return request

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


class TitanBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": inference_parameters.get("temperature"),
                "topP": inference_parameters.get("top_p"), #0.5,
                #"top_k": inference_parameters.get("top_k"), #300,
                "maxTokenCount": inference_parameters.get("max_tokens_to_sample"), #2048,
                #"stop_sequences": []
            }
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        #print("titan")
        #await msg.stream_token("Titan")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    #print(object)
                    if "outputText" in object:
                        completion = object["outputText"]
                        await msg.stream_token(completion)
                    if "completionReason" in object:
                        finish_reason = object["completionReason"]
                        if finish_reason:
                            if "amazon-bedrock-invocationMetrics" in object:
                                invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                                if invocation_metrics:
                                    input_token_count = invocation_metrics["inputTokenCount"]
                                    output_token_count = invocation_metrics["outputTokenCount"]
                                    latency = invocation_metrics["invocationLatency"]
                                    lag = invocation_metrics["firstByteLatency"]
                                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag} finish_reason={finish_reason}"
                                    await msg.stream_token(f"\n\n{stats}")

