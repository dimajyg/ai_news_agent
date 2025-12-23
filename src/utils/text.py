from typing import Optional

async def translate_to_english(llm, text: str) -> str:
    if not text:
        return ''
    try:
        prompt = f"Translate the following text to English. Return only the translated text, no explanations.\n\n{text}"
        res = await llm.ainvoke(prompt)
        if hasattr(res, 'content') and isinstance(res.content, str):
            out = res.content.strip()
        else:
            out = str(res).strip()
        return out or text
    except Exception:
        return text
