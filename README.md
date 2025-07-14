# 🎙️ Video Dubbing DLS Project

Проект по дублированию видео (**video-dubbing**) с использованием различных NLP и ML подходов.  
Разработан в рамках [Deep Learning School](https://dls.samcs.ru/).

## 📌 Описание

Основная цель проекта — создать **pipeline для автоматического дублирования видео на другой язык**.  
В текущей реализации выполняется перевод с **английского на русский** с помощью стэка моделей машинного обучения:

1. **VAD (Voice Activity Detection)** – детекция речевых сегментов в аудио  
2. **ASR (Automatic Speech Recognition)** – автоматическое распознавание речи  
3. **MT (Machine Translation)** – машинный перевод  
4. **TTS (Text-to-Speech)** – синтез речи на целевом языке  

### Основный отчетный ноутбук находится в `notebooks/report.ipynb`.

---

## 📁 Структура репозитория

```bash
video-dubbing-dls-project/
├── data/ # данные проекта (за исключением чекпоинтов, обучающих аудио)
├── config/ # Конфиги:
├── notebooks/ # Исследовательские ноутбуки:
│ ├── dataset.ipynb # Подготовка датасета
│ ├── report.ipynb # Главный отчет ноутбука
│ └── trainer.ipynb # Обучение и инференс
├── src/
│ ├── vad.py # Voice Activity Detection
│ ├── asr.py # Распознавание речи (ASR)
│ ├── mt.py # Машинный перевод (MT)
│ ├── tts.py # Синтез речи (TTS)
│ ├── pipeline.py # Скрипт для запуска полного пайплайна
│ ├── helpers.py # Отдельные вспомогательные методы
│ └── utils.py # Вспомогательные функции и утилиты по пайплайну
├── pyproject.toml # Конфигурация проекта
├── README.md # Описание проекта
```

---

## 🧠 Используемые модели

| Компонент | Модель                     | Источник |
|-----------|---------------------------|----------|
| **VAD**   | Silero VAD                 | [Silero](https://github.com/snakers4/silero-vad) |
| **ASR**   | Whisper            | [OpenAI](https://github.com/openai/whisper) |
| **MT**    | Helsinki-NLP/opus-mt-en-ru | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) |
| **TTS**   | xTTS v2                    | [Coqui TTS](https://github.com/coqui-ai/TTS) |