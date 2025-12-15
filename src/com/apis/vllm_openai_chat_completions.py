from typing import Annotated, Any, Iterable, Literal, Optional, Required, TypedDict, Union

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, Field

__all__ = [
    'vLLMOpenAIChatCompletionsConfig',
]


# === Extra Body ===


class vLLMExtraBody(TypedDict, total=False):
    """
    Additional body parameters for vLLM.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2.
    See https://github.com/vllm-project/vllm/blob/d7fbc6ddaccbcbd514cc6e5a48a04666d9930329/vllm/entrypoints/openai/protocol.py#L462.

    Args:
        echo: If true, the new message will be prepended with the last message if they belong to
            the same role.

        add_generation_prompt: If true, the generation prompt will be added to the chat template.
            This is a parameter used by chat template in tokenizer config of the model.
            If this is set, the chat will be formatted so that the final message in the chat is open-ended,
            without any EOS tokens. The model will continue this message rather than starting a new one.
            This allows you to \"prefill\" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

        continue_final_message: If true, the chat will be formatted so that the final message in the chat is open-ended,
            without any EOS tokens. The model will continue this message rather than starting a new one.
            This allows you to \"prefill\" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

        add_special_tokens: If true, special tokens (e.g. BOS) will be added to the prompt on top of what is added by the chat template.
            For most models, the chat template takes care of adding the special tokens so this should be set to false (as is the default).

        documents: A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation).
            If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing \"title\" and \"text\" keys.

        chat_template: A Jinja template to use for this conversion. As of transformers v4.44, default chat template is no longer allowed,
            so you must provide a chat template if the tokenizer does not define one.

        chat_template_kwargs: Additional keyword args to pass to the template renderer. Will be accessible by the chat template.

        mm_processor_kwargs: Additional kwargs to pass to the HF processor.

        guided_json: If specified, the output will follow the JSON schema.

        guided_regex: If specified, the output will follow the regex pattern.

        guided_choice: If specified, the output will be exactly one of the choices.

        guided_grammar: If specified, the output will follow the context free grammar.

        structural_tag: If specified, the output will follow the structural tag schema.

        guided_decoding_backend: If specified, will override the default guided decoding backend of the server for this specific request.
            If set, must be either 'outlines' / 'lm-format-enforcer'.

        guided_whitespace_pattern: If specified, will override the default whitespace pattern for guided json decoding.

        priority: The priority of the request (lower means earlier handling; default: 0). Any priority other than 0 will raise an error if
            the served model does not use priority scheduling.

        return_tokens_as_token_ids: If specified with 'logprobs', tokens are represented as strings of the form 'token_id:{token_id}'
            so that tokens that are not JSON-encodable can be identified.

        return_token_ids: If specified, the result will include token IDs alongside the generated text.
            In streaming mode, prompt_token_ids is included only in the first chunk, and token_ids contains the delta tokens for each chunk.
            This is useful for debugging or when you need to map generated text back to input tokens.

        cache_salt: If specified, the prefix cache will be salted with the provided string to prevent an attacker to
            guess prompts in multi-user environments. The salt should be random, protected from access by 3rd parties,
            and long enough to be unpredictable (e.g., 43 characters base64-encoded, corresponding to 256 bit). Not
            supported by vLLM engine V0.

        kv_transfer_params: KVTransfer parameters used for disaggregated serving.

        vllm_xargs: Additional request parameters with string or numeric values, used by custom extensions.
    """

    echo: bool  # = False
    add_generation_prompt: bool  # = True
    continue_final_message: bool  # = False
    add_special_tokens: bool  # = False
    documents: Optional[list[dict[str, str]]]  # = None
    chat_template: Optional[str]  # = None
    chat_template_kwargs: Optional[dict[str, Any]]  # = None
    mm_processor_kwargs: Optional[dict[str, Any]]  # = None
    guided_json: Optional[Union[str, dict, BaseModel]]  # = None
    guided_regex: Optional[str]  # = None
    guided_choice: Optional[list[str]]  # = None
    guided_grammar: Optional[str]  # = None
    structural_tag: Optional[str]  # = None
    guided_decoding_backend: Optional[str]  # = None
    guided_whitespace_pattern: Optional[str]  # = None
    priority: int  # = 0
    return_tokens_as_token_ids: Optional[bool]  # = None
    return_token_ids: Optional[bool]  # = None
    cache_salt: Optional[str]  # = None
    kv_transfer_params: Optional[dict[str, Any]]  # = None
    vllm_xargs: Optional[dict[str, Union[str, int, float]]]  # = None


class vLLMSamplingParams(TypedDict, total=False):
    """
    Additional sampling parameters for vLLM.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api_1.
    See https://github.com/vllm-project/vllm/blob/d7fbc6ddaccbcbd514cc6e5a48a04666d9930329/vllm/entrypoints/openai/protocol.py#L443.

    Args:
        best_of: Number of output sequences that are generated from the prompt. From
            these `best_of` sequences, the top `n` sequences are returned. `best_of`
            must be greater than or equal to `n`. By default, `best_of` is set to `n`.
            Warning, this is only supported in V0.
        use_beam_search: Whether to use beam search.
        top_k: Controls the number of top tokens to consider. Set to 0 (or -1) to
            consider all tokens.
        min_p: Represents the minimum probability for a token to be considered,
            relative to the probability of the most likely token. Must be in [0, 1].
            Set to 0 to disable this.
        repetition_penalty: Penalizes new tokens based on whether they appear in the prompt and the
            generated text so far. Values > 1 encourage the model to use new tokens,
            while values < 1 encourage the model to repeat tokens.
        length_penalty: Beam search length penalty.
        stop_token_ids: Token IDs that stop the generation when they are generated. The returned
            output will contain the stop tokens unless the stop tokens are special
            tokens.
        include_stop_str_in_output: Whether to include the stop strings in output text.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        min_tokens: Minimum number of tokens to generate per output sequence before EOS or
            `stop_token_ids` can be generated
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special tokens in the output.
        truncate_prompt_tokens: If set to -1, will use the truncation size supported by the model. If
            set to an integer k, will use only the last k tokens from the prompt
            (i.e., left truncation). If set to `None`, truncation is disabled.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        allowed_token_ids: If provided, the engine will construct a logits processor which only
            retains scores for the given token ids.
        bad_words: Words that are not allowed to be generated. More precisely, only the
            last token of a corresponding token sequence is not allowed when the next
            generated token can complete the sequence.
    """

    best_of: Optional[int]  # = None
    use_beam_search: bool  # = False
    top_k: Optional[int]  # = None
    min_p: Optional[float]  # = None
    repetition_penalty: Optional[float]  # = None
    length_penalty: float  # = 1.0
    stop_token_ids: Optional[list[int]]  # = []
    include_stop_str_in_output: bool  # = False
    ignore_eos: bool  # = False
    min_tokens: int  # = 0
    skip_special_tokens: bool  # = True
    spaces_between_special_tokens: bool  # = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]]  # = None
    prompt_logprobs: Optional[int]  # = None
    allowed_token_ids: Optional[list[int]]  # = None
    bad_words: list[str]  # = []


# === Config ===


class vLLMOpenAIChatCompletionsConfig(vLLMSamplingParams):
    """
    Input for vLLM-compatible OpenAI chat.completions.create().

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api_1.
    See https://github.com/vllm-project/vllm/blob/d7fbc6ddaccbcbd514cc6e5a48a04666d9930329/vllm/entrypoints/openai/protocol.py#L406.

    POST: (http://<host>:<port>/v1/<model_name>)

    Args:
        messages: A list of messages comprising the conversation so far. Depending on the
            [model](https://platform.openai.com/docs/models) you use, different message
            types (modalities) are supported, like
            [text](https://platform.openai.com/docs/guides/text-generation),
            [images](https://platform.openai.com/docs/guides/vision), and
            [audio](https://platform.openai.com/docs/guides/audio).

        model: Model ID used to generate the response, like `gpt-4o` or `o3`. OpenAI offers a
            wide range of models with different capabilities, performance characteristics,
            and price points. Refer to the
            [model guide](https://platform.openai.com/docs/models) to browse and compare
            available models.

        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.

        include_reasoning: Whether to include reasoning in the response.

        logit_bias: Modify the likelihood of specified tokens appearing in the completion.

            Accepts a JSON object that maps tokens (specified by their token ID in the
            tokenizer) to an associated bias value from -100 to 100. Mathematically, the
            bias is added to the logits generated by the model prior to sampling. The exact
            effect will vary per model, but values between -1 and 1 should decrease or
            increase likelihood of selection; values like -100 or 100 should result in a ban
            or exclusive selection of the relevant token.

        logprobs: Whether to return log probabilities of the output tokens or not. If true,
            returns the log probabilities of each output token returned in the `content` of
            `message`.

        max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
            including visible output tokens and
            [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

        max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
            completion. This value can be used to control
            [costs](https://openai.com/api/pricing/) for text generated via API.

            This value is now deprecated in favor of `max_completion_tokens`, and is not
            compatible with
            [o-series models](https://platform.openai.com/docs/guides/reasoning).

        n: How many chat completion choices to generate for each input message. Note that
            you will be charged based on the number of generated tokens across all of the
            choices. Keep `n` as `1` to minimize costs.

        presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.

        reasoning_effort: Constrains effort on reasoning for
            [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
            supported values are `minimal`, `low`, `medium`, and `high`. Reducing reasoning
            effort can result in faster responses and fewer tokens used on reasoning in a
            response.

        response_format: An object specifying the format that the model must output.

            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
            Outputs which ensures the model will match your supplied JSON schema. Learn more
            in the
            [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

            Setting to `{ "type": "json_object" }` enables the older JSON mode, which
            ensures the message the model generates is valid JSON. Using `json_schema` is
            preferred for models that support it.

        safety_identifier: A stable identifier used to help detect users of your application that may be
            violating OpenAI's usage policies. The IDs should be a string that uniquely
            identifies each user. We recommend hashing their username or email address, in
            order to avoid sending us any identifying information.
            [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

        seed: This feature is in Beta. If specified, our system will make a best effort to
            sample deterministically, such that repeated requests with the same `seed` and
            parameters should return the same result. Determinism is not guaranteed, and you
            should refer to the `system_fingerprint` response parameter to monitor changes
            in the backend.

        stop: Not supported with latest reasoning models `o3` and `o4-mini`.

            Up to 4 sequences where the API will stop generating further tokens. The
            returned text will not contain the stop sequence.

        stream: If set to true, the model response data will be streamed to the client as it is
            generated using
            [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).
            See the
            [Streaming section below](https://platform.openai.com/docs/api-reference/chat/streaming)
            for more information, along with the
            [streaming responses](https://platform.openai.com/docs/guides/streaming-responses)
            guide for more information on how to handle the streaming events.

        stream_options: Options for streaming response. Only set this when you set `stream: true`.

        temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic. We generally recommend altering this or `top_p` but
            not both.

        tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
            not call any tool and instead generates a message. `auto` means the model can
            pick between generating a message or calling one or more tools. `required` means
            the model must call one or more tools. Specifying a particular tool via
            `{"type": "function", "function": {"name": "my_function"}}` forces the model to
            call that tool.

            `none` is the default when no tools are present. `auto` is the default if tools
            are present.

        tools: A list of tools the model may call. You can provide either
            [custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools)
            or [function tools](https://platform.openai.com/docs/guides/function-calling).

        top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to
            return at each token position, each with an associated log probability.
            `logprobs` must be set to `true` if this parameter is used.

        extra_body: Additional JSON properties to the request, specific to vLLM OpenAI chat completions.

    """

    messages: Required[list[ChatCompletionMessageParam]]
    model: Optional[str]
    frequency_penalty: Optional[float]
    include_reasoning: Optional[bool]
    logit_bias: Optional[dict[str, int]]
    logprobs: Optional[bool]
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[float]
    reasoning_effort: Optional[ReasoningEffort]
    response_format: Optional[completion_create_params.ResponseFormat]
    safety_identifier: Optional[str]
    seed: Optional[int]
    stop: Optional[str] | list[str] | None
    stream: Optional[Literal[False]]
    stream_options: Optional[ChatCompletionStreamOptionsParam]
    temperature: Optional[float]
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam]
    tools: Optional[Iterable[ChatCompletionToolParam]]
    top_logprobs: Optional[int]
    extra_body: Optional[vLLMExtraBody]
