[LLM](https://llm.datasette.io/) plugin for embedding audio files (music, speech, sounds) and text using [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap)

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-clap
```

## Usage

Once you have installed an embedding model you can use it to embed text like this:

```bash
llm embed -m clap -c 'Hello world'
```
Or an audio file like this:
```bash
llm embed -m clap --binary -i AUDIO_1431.wav
```

Embeddings are more useful if you store them in a database - see [the LLM documentation](https://llm.datasette.io/en/stable/embeddings/cli.html#storing-embeddings-in-sqlite) for details.

To embed every audio file in a folder and save them in a collection called "songs":

```bash
llm embed-multi songs -m clap --binary --files songs/ '*.wav'
```
You can then search for songs of specific things like this:
```bash
llm similar songs -c 'rap'
```
