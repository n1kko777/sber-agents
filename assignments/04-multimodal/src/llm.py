import logging
from openai import AsyncOpenAI
from openai import APIError, InternalServerError
from config import config
from models import TransactionResponse

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

async def get_transaction_response_text(
    last_message: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        response = await client.chat.completions.create(
            model=config.MODEL_TEXT,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT_TEXT},
                *message_history[-10:],  # последние 10 сообщений для контекста
                {"role": "user", "content": last_message}
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "transaction_response",
                "schema": TransactionResponse.model_json_schema(),
                "strict": True
            }}
        )
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response")
            raise ValueError("LLM returned empty response")
        
        try:
            # Парсим JSON ответ
            import json
            parsed_json = json.loads(raw_content)
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал ваше сообщение."
            
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response: {json_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse: {parse_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

async def get_transaction_response_image(
    image_base64: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        schema = TransactionResponse.model_json_schema()
        logger.info(f"Using model: {config.MODEL_IMAGE}, base_url: {config.OPENAI_BASE_URL}")
        
        # Логируем размер изображения в более понятном формате
        image_size_bytes = len(image_base64.encode('utf-8')) * 3 // 4  # примерная оценка
        image_size_kb = image_size_bytes / 1024
        logger.info(f"Image size: ~{image_size_kb:.1f} KB ({len(image_base64)} base64 chars)")
        logger.info(f"Message history length: {len(message_history)} messages")
        
        response = await client.chat.completions.create(
            model=config.MODEL_IMAGE,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE},
                *message_history[-10:],  # последние 10 сообщений для контекста
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "Извлеки транзакции из этого изображения"}
                    ]
                }
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "transaction_response",
                "schema": schema,
                "strict": True  # Используем strict для лучшего соответствия схеме
            }}
        )
        
        # Логируем информацию о response объекте
        logger.info(f"Response object: {response}")
        logger.info(f"Response choices count: {len(response.choices)}")
        if response.choices:
            logger.info(f"First choice finish_reason: {response.choices[0].finish_reason}")
            logger.info(f"First choice message role: {response.choices[0].message.role}")
        
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response for image (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response for image")
            logger.error(f"Response object details: {response}")
            logger.error(f"Finish reason: {response.choices[0].finish_reason if response.choices else 'no choices'}")
            raise ValueError("LLM returned empty response")
        
        try:
            # Парсим JSON ответ
            import json
            parsed_json = json.loads(raw_content)
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал изображение."
            
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse for image: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response for image: {json_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse for image: {parse_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

