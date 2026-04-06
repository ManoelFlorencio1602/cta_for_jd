import json
import re
import time
from pathlib import Path

import requests
from typing import Generator, Iterator, Type, TypeVar

from openai import OpenAI
from ollama import Client
from pydantic import BaseModel

from abc import ABC, abstractmethod

from tenacity import wait_random_exponential, stop_after_attempt, retry

T = TypeVar("T", bound=BaseModel)

class LLMService(ABC):
    """Classe base abstrata para todos os serviços de LLM."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate(self, system_msg: str | None, user_msg: str) -> str:
        """Gera uma resposta completa (sem streaming) e retorna como string."""
        pass

    def generate_stream(self, system_msg: str | None, user_msg: str) -> Iterator[str]:
        """
        Gera a resposta em modo streaming, yield de cada chunk de texto
        conforme ele chega do modelo.

        Deve ser sobrescrito pelas subclasses que suportam streaming.
        Por padrão, faz fallback para `generate` e emite a resposta inteira de uma vez.
        """
        yield self.generate(system_msg, user_msg)

    def generate_structured(self, system_msg: str | None, user_msg: str, base_model: Type[T]) -> T:
        """Gera uma resposta estruturada e valida contra `base_model` (Pydantic)."""
        pass

    def generate_batch(self, prompts: list[tuple[str, str]], batch_name: str):
        """Envia um lote de prompts para inferência assíncrona (batch)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch inference.")


class OllamaService(LLMService):
    """
    Serviço para modelos servidos localmente via Ollama (http://localhost:11434).

    Parâmetros de config
    --------------------
    model    : str  – nome do modelo (default: "llama3.1:8b")
    base_url : str  – URL base do servidor Ollama (default: "http://localhost:11434")
    stream   : bool – habilita streaming por padrão em `generate` (default: False)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model    = config.get("model", "llama3.1:8b")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.stream   = config.get("stream", False)
        self.think    = config.get("think", False)

        self.client = Client(host=config.get('host'))

    def generate(self, system_msg: str | None, user_msg: str) -> str:
        """
        Gera uma resposta completa.
        Se `self.stream` for True, consome o stream internamente e retorna
        o texto concatenado.
        """
        if self.stream:
            return "".join(self.generate_stream(system_msg, user_msg))


        response = self.client.chat(
            model=self.model,
            messages=self._build_messages(system_msg, user_msg),
            stream=False,
            think=self.think
        )
        return response.message.content

    def generate_stream(self, system_msg: str | None, user_msg: str) -> Generator[str, None, None]:
        """
        Gera a resposta em streaming, fazendo yield de cada token/chunk
        conforme ele é recebido da API Ollama (`/api/generate` com stream=True).
        """
        prompt = system_msg + "\n" + user_msg if system_msg else user_msg
        with requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=300
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if token := chunk.get("response"):
                    yield token
                if chunk.get("done"):
                    break

    def generate_structured(self, system_msg: str | None, user_msg: str, base_model: Type[T]) -> T:
        """
        Tenta gerar JSON estruturado usando o modo nativo do Ollama.
        Se `think` estiver ativado, realiza uma chamada padrão e tenta parsear o JSON.
        """
        schema = base_model.model_json_schema()

        if self.think:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=self._build_messages(system_msg, user_msg),
                    think=self.think,
                    stream=False
                )

                raw_output = response.message.content
                print("RAW OUTPUT (think mode):", repr(raw_output))

                if raw_output:
                    return base_model.model_validate_json(raw_output)

                raise ValueError("Empty response from Ollama in think mode.")

            except Exception as e:
                print("\nThink mode failed.")
                print(e)
                print("Falling back to structured mode...\n")

        try:
            response = self.client.chat(
                model=self.model,
                messages=self._build_messages(system_msg, user_msg),
                think=self.think,
                stream=False,
                format=base_model.model_json_schema()
            )

            raw_output = response.message.content
            print("RAW OUTPUT (native):", repr(raw_output))

            if raw_output:
                return base_model.model_validate_json(raw_output)

            raise ValueError("Empty response from Ollama structured mode.")

        except Exception as e:

            print("\nStructured mode failed.")
            print(e)
            print("Falling back to prompt-based JSON generation...\n")

        # fallback via prompting
        schema_json = json.dumps(schema, indent=2)

        fallback_prompt = f"""
    You must respond ONLY with valid JSON.

    Follow this JSON schema exactly:

    {schema_json}

    Do not include explanations, markdown, or text outside the JSON.

    User request:
    {user_msg}
    """

        response = self.client.chat(
            model=self.model,
            messages=self._build_messages(system_msg, user_msg),
            think=self.think,
            stream=False
        )

        raw_output = response.message.content
        print("RAW OUTPUT (fallback):", repr(raw_output))

        if not raw_output:
            raise RuntimeError("Fallback JSON generation also returned empty output.")

        match = re.search(r"{.*}", raw_output, re.S)  # Corrige escape redundante

        if not match:
            raise RuntimeError(
                f"Could not extract JSON from model output:\n{raw_output}"
            )

        json_str = match.group(0)
        return base_model.model_validate_json(json_str)

    # Métodos Auxiliares

    def _build_messages(self, system_msg: str | None, user_msg: str) -> list[dict]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        return messages  # Ajusta para Iterable[ChatCompletion...]


class OpenAIService(LLMService):
    """
    Serviço para a API oficial da OpenAI (Responses API).

    Parâmetros de config
    --------------------
    base_url : str  – URL base da API (default: None, usa endpoint oficial)
    model    : str  – nome do modelo (default: "gpt-4o")
    stream   : bool – habilita streaming por padrão em `generate` (default: False)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(base_url=config.get("base_url"))
        self.model  = config.get("model", "gpt-4o")
        self.stream = config.get("stream", False)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, system_msg: str | None, user_msg: str) -> str:
        """
        Gera uma resposta completa via Responses API.
        Se `self.stream` for True, consome o stream internamente e retorna
        o texto concatenado.
        """
        if self.stream:
            return "".join(self.generate_stream(system_msg, user_msg))

        input_msgs = self._build_input(system_msg, user_msg)
        response = self.client.responses.create(model=self.model, input=input_msgs)
        return response.output_text

    def generate_stream(self, system_msg: str | None, user_msg: str) -> Generator[str, None, None]:
        """
        Gera a resposta em streaming via Responses API (stream=True).
        Faz yield de cada delta de texto conforme ele é recebido.
        """
        input_msgs = self._build_input(system_msg, user_msg)
        with self.client.responses.stream(model=self.model, input=input_msgs) as stream:
            for event in stream:
                # O SDK emite ResponseTextDeltaEvent com .delta contendo o chunk
                delta = getattr(event, "delta", None)
                if delta:
                    yield delta

    def generate_structured(self, system_msg: str | None, user_msg: str, base_model: Type[T]) -> T:
        """
        Gera uma resposta estruturada e valida contra `base_model` (Pydantic).
        Streaming não é aplicável aqui – sempre usa modo síncrono.
        """
        input_msgs = self._build_input(system_msg, user_msg)
        response = self.client.responses.parse(
            model=self.model,
            input=input_msgs,
            text_format=base_model
        )
        if (parsed := response.output_parsed) is not None:
            return parsed
        raise ValueError("Failed to parse structured response")

    def generate_batch(self, prompts: list[tuple[str, str]], batch_name: str):
        """Envia um lote de prompts para inferência assíncrona via Batch API."""
        requests_path = self._create_request_file(batch_name, prompts)
        file_id = self._upload_requests_file(requests_path)
        batch = self._create_batch(file_id)
        self._wait_and_download(batch, batch_name)

    def _create_request_file(self, filename: str, prompts: list[tuple[str, str]]):
        requests_dir = Path("requests")
        requests_dir.mkdir(exist_ok=True)

        requests_path = requests_dir / f"requests_{filename}.jsonl"

        with open(requests_path, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                item = {
                    "custom_id": f"req_{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": f"{prompt[0]}\n{prompt[1]}"
                    }
                }
                f.write(json.dumps(item) + "\n")

        print(f"Created {requests_path.name}.")
        return requests_path

    def _upload_requests_file(self, path: Path) -> str:
        uploaded = self.client.files.create(file=open(path, "rb"), purpose="batch")
        print("Uploaded file:", uploaded.id)
        return uploaded.id

    def _create_batch(self, file_id: str):
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h"
        )
        print("Created batch retrieve:", batch.id)
        return batch

    def _wait_and_download(self, batch, filename: str):
        last_status = batch.status
        print("Batch status:", last_status)

        while batch.status not in ["completed", "failed"]:
            time.sleep(60)
            batch = self.client.batches.retrieve(batch.id)
            if batch.status != last_status:
                print("Batch status:", batch.status)
                last_status = batch.status

        if batch.status == "completed":
            print("Batch completed.")
            output_file = self.client.files.content(batch.output_file_id)
            output_path = Path(f"requests/batch_results_{filename}.jsonl")
            with open(output_path, "wb") as f:
                f.write(output_file.read())
                print(f"Saved {output_path.name}")
        else:
            print("Batch failed.")

    # Métodos Auxiliares

    def _build_input(self, system_msg: str | None, user_msg: str) -> list[dict]:
        """Constrói a lista de mensagens no formato da Responses API."""
        msgs = []
        if system_msg is not None:
            msgs.append({"role": "system", "content": system_msg})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

class LocalOpenAILikeService(LLMService):
    """
    Serviço para modelos locais que expõem a API OpenAI-compatible
    (POST /v1/chat/completions).

    Gerencia seu próprio client OpenAI sem depender da variável de ambiente
    OPENAI_API_KEY. Usa a Chat Completions API (v1), não a Responses API.

    Parâmetros
    ----------
    model  : str  – nome do modelo a ser usado
    port   : int  – porta do servidor local
    host   : str  – hostname do servidor (default: "localhost")
    stream : bool – habilita streaming por padrão em `generate` (default: False)
    """

    def __init__(self, model: str, port: int, host: str = "localhost", stream: bool = False):
        super().__init__(config={"model": model})
        self.model  = model
        self.stream = stream
        self.client = OpenAI(
            api_key="local",  # valor obrigatório pelo SDK; ignorado pelo servidor local
            base_url=f"http://{host}:{port}/v1"
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, system_msg: str | None, user_msg: str) -> str:
        """
        Gera uma resposta completa via Chat Completions (/v1/chat/completions).
        Se `self.stream` for True, consome o stream internamente e retorna
        o texto concatenado.
        """
        if self.stream:
            return "".join(self.generate_stream(system_msg, user_msg))

        messages = self._build_messages(system_msg, user_msg)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def generate_stream(self, system_msg: str | None, user_msg: str) -> Generator[str, None, None]:
        """
        Gera a resposta em streaming via Chat Completions (stream=True).
        Faz yield de cada delta de conteúdo conforme ele é recebido.
        """
        messages = self._build_messages(system_msg, user_msg)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_structured(self, system_msg: str | None, user_msg: str, base_model: Type[T]) -> T:
        """
        Gera uma resposta estruturada usando a extensão beta de parsing do SDK OpenAI.
        Streaming não é aplicável aqui – sempre usa modo síncrono.
        """
        messages = self._build_messages(system_msg, user_msg)
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=base_model
        )
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed
        raise ValueError("Failed to parse structured response")

    # Métodos Auxiliares

    def _build_messages(self, system_msg: str | None, user_msg: str) -> list[dict]:
        """Constrói a lista de mensagens no formato Chat Completions."""
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        return messages

