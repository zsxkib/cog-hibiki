# Hibiki: Real-Time Speech Translation

[[Paper]][hibiki] | [[Samples]](https://huggingface.co/spaces/kyutai/hibiki-samples) | [[HuggingFace Models]](https://huggingface.co/collections/kyutai/hibiki-fr-en-67a48835a3d50ee55d37c2b5)

Hibiki is a state-of-the-art model for **real-time speech-to-speech translation** that maintains voice characteristics while translating. It works with French-to-English translation and can run locally on consumer hardware.

## Quick Start

Run translation with a single command using Cog:

```bash
sudo cog predict -i audio_input=@sample_fr_hibiki_crepes.mp3
```

This will translate the sample French audio file to English while preserving voice characteristics. Replace with your own `.mp3` file for custom translations.

## Key Features
- 🎙️ **Voice preservation** through classifier-free guidance
- ⏱️ **Real-time processing** with 12.5Hz framerate
- 🔊 **Natural-sounding output** in target language
- 📜 Simultaneous text transcription

[hibiki]: https://arxiv.org/abs/2502.03382