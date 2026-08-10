import getpass
import json
import pathlib
import time
from typing import Any
from snowflake.snowpark import Session
from openai import OpenAI
import os
from abc import ABC, abstractmethod
import re





class LLMClient(ABC):
    """Minimal interface the verifier needs. Implement for any backend."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Return the raw text completion for a single system+user exchange."""
        ...

    def complete_json(self, system: str, user: str, max_tokens: int = 4096, temperature: int = 0, response_schema: Any = None) -> dict:
        """Completion parsed as JSON, with fence-stripping and one retry."""
        suffix = "\nRespond ONLY with valid JSON. No markdown fences, no preamble."
        for attempt in range(2):
            user_attempt = user + "\nYour previous output was not valid JSON. Return ONLY valid JSON." if attempt > 0 else user
            kwargs: dict[str, Any] = dict(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_attempt+suffix},
                ],
                response_format=response_schema,
            )
            raw_content = self._client.chat.completions.create(**kwargs)

            text = raw_content.choices[0].message.content

            text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable")



class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", **client_kwargs):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(**client_kwargs)  # API key from env by default

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def completion_with_backoff(self, **kwargs):
        import anthropic
        is_ok = False
        retry_count = 0
        while not is_ok:
            retry_count += 1
            try:
                response = self._client.messages.create(**kwargs)
                is_ok = True
            except anthropic.RateLimitError as error:
                if retry_count <= 30:
                    if retry_count % 10 == 0:
                        print(f"Anthropic API retry for {retry_count} times ({error})")
                    time.sleep(10)
                    continue
                else:
                    return {}
            except anthropic.BadRequestError as error:
                msg = str(error)
                if 'maximum context length' in msg or 'max_tokens' in msg:
                    if retry_count <= 3:
                        print(f"reduce max_tokens by 500")
                        kwargs['max_tokens'] = kwargs['max_tokens'] - 500
                        continue
                    else:
                        print(error)
                        return {}
                else:
                    print(error)
                    return {}
            except Exception as error:
                print(error)
                return {}
        return response


class OpenAICompatibleClient(LLMClient):
    """Works with OpenAI's cloud API, vLLM, or any other OpenAI-compatible endpoint."""

    def __init__(self, model: str, model_host: str = None, base_url: str = None):
        from openai import OpenAI
        self.model = model
        if base_url is None:
            # No host given -> talk to OpenAI's cloud API. A host (e.g. "10.0.0.1:8000")
            # is assumed to be a local/self-hosted OpenAI-compatible server (vLLM, etc.).
            base_url = f"http://{model_host}/v1" if model_host else "https://api.openai.com/v1"
        self._client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def completion_with_backoff(self, **kwargs):
        import openai
        is_ok = False
        retry_count = 0
        while not is_ok:
            retry_count += 1
            try:
                response = self._client.chat.completions.create(**kwargs)
                is_ok = True
            except openai.RateLimitError as error:
                if retry_count <= 30:
                    if retry_count % 10 == 0:
                        print(f"OpenAI API retry for {retry_count} times ({error})")
                    time.sleep(10)
                    continue
                else:
                    return {}
            except openai.BadRequestError as error:
                msg = str(error)
                if 'maximum context length' in msg:
                    if retry_count <= 3:
                        print(f"reduce max_tokens by 500")
                        kwargs['max_tokens'] = kwargs['max_tokens'] - 500
                        continue
                    else:
                        print(error)
                        return {}
                else:
                    print(error)
                    return {}
            except Exception as error:
                print(error)
                return {}
        return response


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI, using deployment-based routing.

    Reads endpoint/key from OPENAI_API_BASE / OPENAI_API_KEY, matching the
    env vars run.py already used for Azure.
    """

    def __init__(self, model: str, deploy_name: str = None, api_version: str = "2023-05-15"):
        from openai import AzureOpenAI
        self.model = deploy_name or model
        self._client = AzureOpenAI(
            azure_endpoint=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=api_version,
        )

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def completion_with_backoff(self, **kwargs):
        import openai
        is_ok = False
        retry_count = 0
        while not is_ok:
            retry_count += 1
            try:
                response = self._client.chat.completions.create(**kwargs)
                is_ok = True
            except openai.RateLimitError as error:
                if retry_count <= 30:
                    if retry_count % 10 == 0:
                        print(f"Azure OpenAI API retry for {retry_count} times ({error})")
                    time.sleep(10)
                    continue
                else:
                    return {}
            except openai.BadRequestError as error:
                msg = str(error)
                if 'maximum context length' in msg:
                    if retry_count <= 3:
                        print(f"reduce max_tokens by 500")
                        kwargs['max_tokens'] = kwargs['max_tokens'] - 500
                        continue
                    else:
                        print(error)
                        return {}
                else:
                    print(error)
                    return {}
            except Exception as error:
                print(error)
                return {}
        return response


class SnowflakeCompatibleClient(LLMClient):
    def __init__(self, model):
        import snowflake.connector
        params = {
        "account": "moffitt.us-east-1.privatelink",
        "user": "shohreh.haddadan@moffitt.org",
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": pathlib.Path("~/.snowflake/snowflake_key.p8").expanduser(),
        "private_key_file_pwd": (
            os.getenv("Pass") or getpass.getpass(prompt="Passphrase to unlock Snowflake private key: ")
        ),
        "database": "MCAP_CDSC_PROD",
        "schema": "MCC23352_THIEU_THANH",
        "warehouse": "SNOWFLAKE_HB_WH",
        "role": "SNOWFLAKE_MCC23352",
        }
        self.snowflake_session = Session.builder.configs(params).create()
        self.model = model
        #self._client = snowflake.connector.connect(**client_kwargs)

    def complete(self, system: str, user: str, max_tokens: int = 4096 ,temperature =0) -> str:
        """Call Snowflake Cortex AI_COMPLETE using an already-open session."""
        
        prompt = system + "\\n" + user
        safe_prompt = prompt.replace("$$", "\\$\\$")
  
        model_params = {"temperature": temperature, "max_tokens": max_tokens}
        query = f"""
        SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
            '{self.model}',
            $${safe_prompt}$$,
            model_parameters => {model_params}
        )
        """
        row = self.snowflake_session.sql(query).collect()
        return row[0][0]
    def complete_json(self, system: str, user: str, max_tokens: int = 4096, temperature = 0 ,response_schema = None) -> dict:
        """Call Snowflake Cortex AI_COMPLETE using an already-open session."""
        prompt = system+"\\n"+user
        safe_prompt = prompt.replace("$$", "\\$\\$")
        model_params = {"temperature": temperature, "max_tokens": max_tokens}
        if response_schema is not None:
            query = f"""
            SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                '{self.model}',
                $${safe_prompt}$$,
                model_parameters => {model_params},
                response_format => {response_schema}
            )
            """
        else:
            query = f"""
            SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                '{self.model}',
                $${safe_prompt}$$,
                model_parameters => {model_params}
            )
            """
        row = self.snowflake_session.sql(query).collect()
        return row[0][0]

    def __del__(self):
    
        if self.snowflake_session:
            self.snowflake_session.close()
            print("Snowflake session closed.")

    def completion_with_backoff(self, **kwargs):
        """Retry wrapper around AI_COMPLETE. Accepts prompt, max_tokens, temperature, response_schema."""
        max_tokens = kwargs.get("max_tokens", 4096)
        temperature = kwargs.get("temperature", 0)
        prompt = kwargs.get("prompt", "")
        response_schema = kwargs.get("response_schema", None)
        is_ok = False
        retry_count = 0
        while not is_ok:
            retry_count += 1
            try:
                safe_prompt = prompt.replace("$$", "\\$\\$")
                model_params = {"temperature": temperature, "max_tokens": max_tokens}
                if response_schema is not None:
                    query = f"""
                    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                        '{self.model}',
                        $${safe_prompt}$$,
                        model_parameters => {model_params},
                        response_format => {response_schema}
                    )
                    """
                else:
                    query = f"""
                    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                        '{self.model}',
                        $${safe_prompt}$$,
                        model_parameters => {model_params}
                    )
                    """
                row = self.snowflake_session.sql(query).collect()
                response = row[0][0]
                is_ok = True
            except Exception as error:
                msg = str(error)
                if ("rate limit" in msg.lower() or "throttl" in msg.lower()) and retry_count <= 30:
                    if retry_count % 10 == 0:
                        print(f"Snowflake Cortex retry for {retry_count} times ({error})")
                    time.sleep(10)
                    continue
                if ("max tokens" in msg.lower() or "context length" in msg.lower()) and retry_count <= 3:
                    print(f"reduce max_tokens by 500")
                    max_tokens = max_tokens - 500
                    continue
                print(error)
                return {}
        return response


class snowflake_client():
    def __init__(self):
        params = {
        "account": "moffitt.us-east-1.privatelink",
        "user": "shohreh.haddadan@moffitt.org",
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": pathlib.Path("~/.snowflake/snowflake_key.p8").expanduser(),
        "private_key_file_pwd": (
            os.getenv("Pass") or getpass.getpass(prompt="Passphrase to unlock Snowflake private key: ")
        ),
        "database": "MCAP_CDSC_PROD",
        "schema": "MCC23352_THIEU_THANH",
        "warehouse": "SNOWFLAKE_HB_WH",
        "role": "SNOWFLAKE_MCC23352",
        }
        self.snowflake_session = Session.builder.configs(params).create()
    

    def call_model(self, model_name: str, prompt: str,temperature: float , response_format: str = None) -> str:
        """Call Snowflake Cortex AI_COMPLETE using an already-open session."""
        safe_prompt = prompt.replace("$$", "\\$\\$")
        model_params = {"temperature": temperature}
        if response_format is not None:
            query = f"""
            SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                '{model_name}',
                $${safe_prompt}$$,
                model_parameters => {model_params},
                response_format => {response_format}
            )
            """
        else:
            query = f"""
            SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                '{model_name}',
                $${safe_prompt}$$,
                model_parameters => {model_params}
            )
            """
        row = self.snowflake_session.sql(query).collect()
        return row[0][0]
    def __del__(self):
    
        if self.snowflake_session is not None:
            self.snowflake_session.close()
            print("Snowflake session closed.")

    def completion_with_backoff(self, **kwargs):
        """Retry wrapper around AI_COMPLETE. Accepts model_name, prompt, temperature, response_format, max_tokens."""
        model_name = kwargs.get("model_name")
        prompt = kwargs.get("prompt", "")
        temperature = kwargs.get("temperature", 0)
        response_format = kwargs.get("response_format", None)
        max_tokens = kwargs.get("max_tokens", 4096)
        is_ok = False
        retry_count = 0
        while not is_ok:
            retry_count += 1
            try:
                safe_prompt = prompt.replace("$$", "\\$\\$")
                model_params = {"temperature": temperature, "max_tokens": max_tokens}
                if response_format is not None:
                    query = f"""
                    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                        '{model_name}',
                        $${safe_prompt}$$,
                        model_parameters => {model_params},
                        response_format => {response_format}
                    )
                    """
                else:
                    query = f"""
                    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                        '{model_name}',
                        $${safe_prompt}$$,
                        model_parameters => {model_params}
                    )
                    """
                row = self.snowflake_session.sql(query).collect()
                response = row[0][0]
                is_ok = True
            except Exception as error:
                msg = str(error)
                if ("rate limit" in msg.lower() or "throttl" in msg.lower()) and retry_count <= 30:
                    if retry_count % 10 == 0:
                        print(f"Snowflake Cortex retry for {retry_count} times ({error})")
                    time.sleep(10)
                    continue
                if ("max tokens" in msg.lower() or "context length" in msg.lower()) and retry_count <= 3:
                    print(f"reduce max_tokens by 500")
                    max_tokens = max_tokens - 500
                    continue
                print(error)
                return {}
        return response
def build_client(llm_cfg: dict) -> LLMClient:
    """Build a client from a config section like {"provider": ..., "model": ..., ...}."""
    provider = llm_cfg.get("provider", "openai")
    kwargs = {k: v for k, v in llm_cfg.items()
              if k not in ("provider", "model") and v is not None}
    
    if provider in ("openai", "openai_compatible"):
        return OpenAICompatibleClient(model=llm_cfg["model"], **kwargs)
    elif provider == "azure":
        return AzureOpenAIClient(model=llm_cfg["model"], **kwargs)
    elif provider == "snowflake":
        return SnowflakeCompatibleClient(model=llm_cfg["model"], **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model=llm_cfg["model"], **kwargs)
    raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":

    system_prompt = "You are a one word answer bot."
    user_prompt = "How are you?"
    json_schema_snowflake = {
        "type": "json",
        "schema": {
        "type": "object",
        "additionalProperties":False,
        "required": ["status"],
        "properties": {
            "status": { "type": "string"}
        }
        }
    }
    json_schema_openai={
    "type": "json_schema",
    "json_schema": {
            "name": "sample",    # required
            "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                
            },
            "required": ["status"]
            }
        }
    
    }
    params = {

    }
    client = SnowflakeCompatibleClient("OPENAI-GPT-5-NANO")
    print(client.complete(system=system_prompt, user=user_prompt))
    print(client.complete_json(system=system_prompt, user=user_prompt, response_schema=json_schema_snowflake))
    print(client.completion_with_backoff())

    client = OpenAICompatibleClient(model="google/gemma-4-31B-it", model_host="10.14.29.33:11434")
    print(client.complete(system=system_prompt, user=user_prompt))
    print(client.complete_json(system=system_prompt, user=user_prompt , response_schema=json_schema_openai))

    